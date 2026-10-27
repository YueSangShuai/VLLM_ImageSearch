#include "vlm_describer.h"
#include "util.h"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>

// 全流程参照 llama.cpp/tools/mtmd/mtmd-cli.cpp 的单轮生成路径：
//   chat template 格式化 -> mtmd_tokenize -> mtmd_helper_eval_chunks
//   -> common_sampler 循环采样直到 EOG / max_new_tokens
// 与之的差异仅在于：每次 describe 都是独立单轮（清 KV），图像直接以解码后的 RGB 传入
// （Python 端把 PIL 图重编码 JPEG q90 是为了走 llama-cpp-python 的 base64 通道，这里不需要）。

struct VlmDescriber::Impl {
    common_params params;
    common_init_result_ptr llama_init;
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_sampler * smpl = nullptr;
    common_chat_templates_ptr tmpls;
    mtmd::context_ptr vision;
    llama_batch batch;          // 单 token 续采样用
    int n_batch = 2048;
    bool ok_flag = false;
    double cur_temp = 0.0;
    double cur_rep = 1.0;
};

static bool g_common_inited = false;

VlmDescriber::VlmDescriber(const std::string & gguf_path, const std::string & mmproj_path,
                           const std::string & device, int n_gpu_layers, int n_ctx,
                           double temperature, double repetition_penalty)
    : impl_(new Impl()) {
    if (!g_common_inited) {
        common_init();
        ggml_backend_load_all();  // 动态注册 CUDA/CPU 后端
        g_common_inited = true;
    }

    common_params & params = impl_->params;
    params.model.path = gguf_path;
    params.mmproj.path = mmproj_path;
    params.n_ctx = n_ctx;
    params.verbosity = 2;  // 只保留 warn 及以上，避免刷屏

    bool use_gpu = device.rfind("cpu", 0) != 0;
    params.n_gpu_layers = use_gpu ? n_gpu_layers : 0;
    if (!use_gpu) params.mmproj_use_gpu = false;
    params.sampling.temp = temperature;
    params.sampling.penalty_repeat = (float)repetition_penalty;
    impl_->cur_temp = temperature;
    impl_->cur_rep = repetition_penalty;

    impl_->llama_init = common_init_from_params(params);
    impl_->model = impl_->llama_init->model();
    impl_->lctx = impl_->llama_init->context();
    if (!impl_->model || !impl_->lctx) {
        fprintf(stderr, "VLM 模型加载失败: %s\n", gguf_path.c_str());
        return;
    }
    impl_->vocab = llama_model_get_vocab(impl_->model);
    impl_->smpl = common_sampler_init(impl_->model, params.sampling);
    // GGUF 内置 tokenizer.chat_template（已验证齐全），use_jinja 默认 true
    impl_->tmpls = common_chat_templates_init(impl_->model, params.chat_template);

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = use_gpu && params.mmproj_use_gpu;
    mp.n_threads = params.cpuparams.n_threads;
    mp.print_timings = false;
    impl_->vision.reset(mtmd_init_from_file(mmproj_path.c_str(), impl_->model, mp));
    if (!impl_->vision) {
        fprintf(stderr, "mmproj 加载失败: %s\n", mmproj_path.c_str());
        return;
    }

    impl_->batch = llama_batch_init(1, 0, 1);
    impl_->n_batch = params.n_batch;
    impl_->ok_flag = true;
    printf("✅ VLM 加载完成: %s (gpu_layers=%d, n_ctx=%d)\n",
           gguf_path.c_str(), params.n_gpu_layers, params.n_ctx);
}

VlmDescriber::~VlmDescriber() {
    if (!impl_) return;
    if (impl_->smpl) common_sampler_free(impl_->smpl);
    if (impl_->ok_flag) llama_batch_free(impl_->batch);
    impl_->vision.reset();      // mtmd 持有 model 指针，必须先于 model 释放
    impl_->llama_init.reset();
}

bool VlmDescriber::ok() const { return impl_->ok_flag; }

std::string VlmDescriber::describe(const ImageRGB & img, const std::string & prompt,
                                   int max_new_tokens, double temperature,
                                   double repetition_penalty) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!impl_->ok_flag || img.w <= 0 || img.h <= 0) return "";

    // 调用方通常都传同一组全局采样参数；仅实际变化时重建 sampler（罕见路径）
    if (temperature != impl_->cur_temp || repetition_penalty != impl_->cur_rep) {
        common_sampler_free(impl_->smpl);
        impl_->params.sampling.temp = temperature;
        impl_->params.sampling.penalty_repeat = (float)repetition_penalty;
        impl_->smpl = common_sampler_init(impl_->model, impl_->params.sampling);
        impl_->cur_temp = temperature;
        impl_->cur_rep = repetition_penalty;
    }
    common_sampler_reset(impl_->smpl);

    // 1. 单轮 user 消息；marker 置于文首（与 mtmd-cli 单轮逻辑一致）
    std::string content = std::string(mtmd_default_marker()) + prompt;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = content;
    std::string formatted = common_chat_format_single(
        impl_->tmpls.get(), /*past=*/{}, msg, /*add_ass=*/true, /*use_jinja=*/true);

    // 2. 图像 -> bitmap（RGB 直通，不重编码）
    mtmd::bitmap bmp(img.w, img.h, img.rgb.data());
    if (!bmp.ptr) {
        fprintf(stderr, "mtmd_bitmap_init 失败: %dx%d\n", img.w, img.h);
        return "";
    }

    mtmd_input_text text;
    text.text = formatted.data();
    text.text_len = formatted.size();
    text.add_special = true;   // 空历史，首 chunk 需要 BOS
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    const mtmd_bitmap * bptrs[1] = { bmp.ptr.get() };
    int32_t res = mtmd_tokenize(impl_->vision.get(), chunks.ptr.get(), &text, bptrs, 1);
    if (res != 0) {
        fprintf(stderr, "mtmd_tokenize 失败: %d\n", res);
        return "";
    }

    // 3. 每次 describe 都是独立单轮：清空 KV 后从头评估
    llama_memory_clear(llama_get_memory(impl_->lctx), true);

    llama_pos n_past = 0;
    res = mtmd_helper_eval_chunks(impl_->vision.get(), impl_->lctx, chunks.ptr.get(),
                                  /*n_past=*/0, /*seq_id=*/0, impl_->n_batch,
                                  /*logits_last=*/true, &n_past);
    if (res != 0) {
        fprintf(stderr, "mtmd_helper_eval_chunks 失败: %d\n", res);
        return "";
    }

    // 4. 采样循环（temp=0 时 common_sampler 即贪心，等价 python do_sample=False）
    std::string out;
    for (int i = 0; i < max_new_tokens; i++) {
        llama_token tok = common_sampler_sample(impl_->smpl, impl_->lctx, -1);
        common_sampler_accept(impl_->smpl, tok, true);
        if (llama_vocab_is_eog(impl_->vocab, tok)) break;
        out += common_token_to_piece(impl_->lctx, tok);
        if (i + 1 >= max_new_tokens) break;
        common_batch_clear(impl_->batch);
        common_batch_add(impl_->batch, tok, n_past++, /*seq_id=*/{0}, /*logits=*/true);
        if (llama_decode(impl_->lctx, impl_->batch) != 0) {
            fprintf(stderr, "llama_decode 生成 token 失败\n");
            break;
        }
    }
    return trim(out);
}
