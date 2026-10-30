// RKLLM C API 封装实现。生成是回调流式的：rkllm_run 阻塞至生成结束，
// NORMAL 状态逐段回调 result->text，这里在 mu_ 持锁期间累加到 current_out_。
#include "vlm_describer.h"
#include "attr_head.h"
#include "rknn_image_enc.h"
#include "rkllm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Qwen3-VL 的 vision special token（rknn-llm demo 默认值，run_base/run_ft 实测正确）。
// 换模型时用环境变量覆盖。
const char * env_or(const char * key, const char * def) {
    const char * v = getenv(key);
    return (v && v[0]) ? v : def;
}

struct CallbackCtx {  // userdata -> 实例
    std::string * out;
};

int rkllm_text_callback(RKLLMResult * result, void * userdata, LLMCallState state) {
    auto * ctx = static_cast<CallbackCtx *>(userdata);
    if (state == RKLLM_RUN_NORMAL && result && result->text && ctx && ctx->out) {
        ctx->out->append(result->text);
    } else if (state == RKLLM_RUN_ERROR) {
        fprintf(stderr, "rkllm run error\n");
    }
    return 0;
}

}  // namespace

struct VlmDescriber::Impl {
    LLMHandle llm = nullptr;
    rknn_app_context_t vision{};
    bool vision_ready = false;
    bool ready = false;

    std::string vision_path, llm_path;
    int n_ctx = 4096;
    float temperature = 0.0f;
    float repeat_penalty = 1.2f;

    // 视觉 embedding 缓冲（init 时按模型维度分配，复用）
    std::vector<float> embed;
    size_t n_image_tokens = 0;
    int embed_len = 0;
    int model_w = 0, model_h = 0;

    // Qwen3-VL special token
    std::string img_start, img_end, img_content;

    // 属性线性头（可选，setAttrHead 加载）：启用时预处理改为 expand2square+resize（对齐训练端）
    AttrHead head;
    bool attr_pad = false;

    CallbackCtx cb_ctx;
    std::string * current_out = nullptr;

    ~Impl() {
        if (llm) rkllm_destroy(llm);
        if (vision_ready) release_imgenc(&vision);
    }
};

VlmDescriber::VlmDescriber(const std::string & vision_path, const std::string & llm_path,
                           const std::string & /*device*/, int /*n_gpu_layers*/, int n_ctx,
                           double temperature, double repetition_penalty)
    : impl_(new Impl) {
    impl_->vision_path = vision_path;
    impl_->llm_path = llm_path;
    impl_->n_ctx = n_ctx;
    impl_->temperature = (float)temperature;
    impl_->repeat_penalty = (float)repetition_penalty;

    impl_->img_start = env_or("VLM_IMG_START", " vision_start");
    impl_->img_end = env_or("VLM_IMG_END", " vision_end");
    impl_->img_content = env_or("VLM_IMG_CONTENT", " vision");

    const char * cores_env = getenv("RKLLM_NPU_CORE_NUM");
    int core_num = (cores_env && atoi(cores_env) > 0) ? atoi(cores_env) : 3;

    // ---- LLM ----
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = impl_->llm_path.c_str();
    param.max_context_len = impl_->n_ctx;
    param.max_new_tokens = 512;  // describe() 每次 run 会用 InferParam 覆盖
    param.top_k = 1;
    param.temperature = impl_->temperature;
    param.repeat_penalty = impl_->repeat_penalty;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;  // RK3588/RK3576

    RKLLMCallback rkllm_cb = {};
    rkllm_cb.result_callback = rkllm_text_callback;

    int ret = rkllm_init(&impl_->llm, &param, &rkllm_cb);
    if (ret != 0 || !impl_->llm) {
        fprintf(stderr, "rkllm_init fail! ret=%d path=%s\n", ret, impl_->llm_path.c_str());
        return;
    }

    // ---- Vision ----
    if (init_imgenc(impl_->vision_path.c_str(), &impl_->vision, core_num) != 0) {
        fprintf(stderr, "init_imgenc fail! path=%s\n", impl_->vision_path.c_str());
        return;
    }
    impl_->vision_ready = true;
    impl_->n_image_tokens = (size_t)impl_->vision.model_image_token;
    impl_->embed_len = impl_->vision.model_image_token *
                       (int)impl_->vision.io_num.n_output *
                       impl_->vision.model_embed_size;
    impl_->model_w = impl_->vision.model_width;
    impl_->model_h = impl_->vision.model_height;
    impl_->embed.assign((size_t)impl_->embed_len, 0.0f);

    impl_->cb_ctx.out = nullptr;
    impl_->ready = true;
    printf("VlmDescriber(RKLLM) ready: vision=%dx%d tokens=%zu llm=%s\n",
           impl_->model_w, impl_->model_h, impl_->n_image_tokens, impl_->llm_path.c_str());
}

VlmDescriber::~VlmDescriber() = default;

bool VlmDescriber::ok() const { return impl_->ready; }

void VlmDescriber::setAttrHead(const std::string & path) {
    if (path.empty() || !impl_->ready) return;
    if (impl_->head.load(path, impl_->vision.model_embed_size))
        impl_->attr_pad = true;  // 与训练端对齐：expand2square(127) -> resize 448
}

bool VlmDescriber::attrHeadReady() const { return impl_->head.ready(); }

std::string VlmDescriber::describe(const ImageRGB & img, const std::string & prompt,
                                   int max_new_tokens, double temperature,
                                   double repetition_penalty,
                                   std::vector<std::string> * attrs_out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (attrs_out) attrs_out->clear();
    if (!impl_->ready) return "";

    // ROI -> 固定分辨率（与导出时的 448x448 一致；双线性，对齐 demo 的 INTER_LINEAR）。
    // 属性头启用时先 expand2square 再加灰边（与 qwen_classifer 训练预处理逐像素对齐）。
    ImageRGB resized = impl_->attr_pad
                           ? resize_bilinear(expand2square(img), impl_->model_w, impl_->model_h)
                           : (img.w == impl_->model_w && img.h == impl_->model_h
                                  ? img
                                  : resize_bilinear(img, impl_->model_w, impl_->model_h));
    if (resized.rgb.size() != (size_t)impl_->model_w * impl_->model_h * 3) return "";

    if (run_imgenc(&impl_->vision, resized.rgb.data(), impl_->embed.data()) != 0) {
        fprintf(stderr, "run_imgenc fail\n");
        return "";
    }

    // 属性头：视觉 embedding 已在手，顺带算 54 属性（<1ms CPU），省一次 VLM 结构化生成
    if (attrs_out && impl_->head.ready())
        impl_->head.run(impl_->embed.data(), (int)impl_->n_image_tokens,
                        impl_->vision.io_num.n_output, impl_->vision.model_embed_size,
                        attrs_out);

    // <image> 占位符必须在 prompt 文首（demo 行为）
    std::string full_prompt = "<image>" + prompt;

    RKLLMInput input = {};
    input.input_type = RKLLM_INPUT_MULTIMODAL;
    input.role = "user";
    input.multimodal_input.prompt = (char *)full_prompt.c_str();
    input.multimodal_input.image.image_embed = impl_->embed.data();
    input.multimodal_input.image.n_image_tokens = impl_->n_image_tokens;
    input.multimodal_input.image.n_image = 1;
    input.multimodal_input.image.image_start = impl_->img_start.c_str();
    input.multimodal_input.image.image_end = impl_->img_end.c_str();
    input.multimodal_input.image.image_content = impl_->img_content.c_str();
    input.multimodal_input.image.image_height = (size_t)impl_->model_h;
    input.multimodal_input.image.image_width = (size_t)impl_->model_w;

    RKLLMSamplingParam sampling = {};
    sampling.top_k = 1;
    sampling.temperature = (float)temperature;
    sampling.repeat_penalty = (float)repetition_penalty;

    RKLLMInferParam infer = {};
    infer.mode = RKLLM_INFER_GENERATE;
    infer.keep_history = 0;
    infer.max_new_tokens = (int32_t)max_new_tokens;
    infer.sampling_params = &sampling;

    std::string out;
    impl_->cb_ctx.out = &out;
    impl_->current_out = &out;
    int ret = rkllm_run(impl_->llm, &input, &infer, &impl_->cb_ctx);
    impl_->cb_ctx.out = nullptr;
    impl_->current_out = nullptr;
    if (ret != 0) {
        fprintf(stderr, "rkllm_run fail! ret=%d\n", ret);
        return "";
    }
    return out;
}
