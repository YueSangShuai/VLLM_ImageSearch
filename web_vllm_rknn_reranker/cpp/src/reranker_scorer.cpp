// RerankerScorer 实现：骨架复用 vlm_describer.cpp（rkllm_init + rknn_image_enc），
// 差异仅在 rkllm_run 用 RKLLM_INFER_GET_LOGITS / GET_LAST_HIDDEN_LAYER 并从回调取输出。
// 回调里只拷 need 的 2 个 logit / 末 token hidden（rkllm.h 提示大缓冲随后可能被回收）。
#include "reranker_scorer.h"
#include "config.h"          // getenv_bool
#include "rknn_image_enc.h"
#include "rkllm.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <vector>

namespace {

const char * env_or(const char * key, const char * def) {
    const char * v = getenv(key);
    return (v && v[0]) ? v : def;
}

// 与 HF chat_template 渲染的 system 句逐字一致（官方 qwen3_vl_reranker.py 硬编码）
const char * kSystemPrompt =
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\".";

double sigmoidf_(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

struct RerankerScorer::Impl {
    LLMHandle llm = nullptr;
    rknn_app_context_t vision{};
    bool vision_ready = false;
    bool ready = false;
    bool use_hidden = false;

    std::string vision_path, llm_path, instruction;
    int n_ctx = 2048;
    long yes_id = 9693, no_id = 2152;
    std::string img_start, img_end, img_content;

    std::vector<float> embed;              // 视觉 embedding 缓冲（复用）
    std::vector<float> wdiff;              // W_yes - W_no（hidden 模式）
    size_t n_image_tokens = 0;
    int model_w = 0, model_h = 0;

    // 帧级视觉缓存：同帧多规则只跑一次视觉塔（比对 resize 后的 RGB）
    std::vector<uint8_t> last_rgb;
    bool cache_valid = false;

    // run 输出（回调填充，取末 token 行）
    double logit_yes = 0, logit_no = 0;    // logits 模式
    std::vector<float> hidden;             // hidden 模式：末 token
    bool out_ok = false;

    static int cb(RKLLMResult * result, void * userdata, LLMCallState state) {
        auto * self = static_cast<Impl *>(userdata);
        if (state == RKLLM_RUN_FINISH && result) {
            if (result->logits.logits && result->logits.vocab_size > 0) {
                int vocab = result->logits.vocab_size;
                int row = (result->logits.num_tokens > 0 ? result->logits.num_tokens : 1) - 1;
                const float * base = result->logits.logits + (size_t)row * vocab;
                if (self->yes_id < vocab && self->no_id < vocab) {
                    self->logit_yes = base[self->yes_id];
                    self->logit_no = base[self->no_id];
                    self->out_ok = true;
                }
            } else if (result->last_hidden_layer.hidden_states && self->use_hidden) {
                int d = result->last_hidden_layer.embd_size;
                int row = (result->last_hidden_layer.num_tokens > 0
                               ? result->last_hidden_layer.num_tokens : 1) - 1;
                const float * base = result->last_hidden_layer.hidden_states + (size_t)row * d;
                self->hidden.assign(base, base + d);
                self->out_ok = true;
            }
        } else if (state == RKLLM_RUN_ERROR) {
            fprintf(stderr, "reranker rkllm run error\n");
            self->out_ok = false;
        }
        return 0;
    }

    ~Impl() {
        if (llm) rkllm_destroy(llm);
        if (vision_ready) release_imgenc(&vision);
    }
};

RerankerScorer::RerankerScorer(const std::string & vision_path, const std::string & llm_path,
                               int n_ctx, const std::string & instruction,
                               const std::string & mode, const std::string & weight_bin,
                               long yes_id, long no_id)
    : impl_(new Impl) {
    impl_->vision_path = vision_path;
    impl_->llm_path = llm_path;
    impl_->n_ctx = n_ctx;
    impl_->instruction = instruction;
    impl_->yes_id = yes_id > 0 ? yes_id : 9693;
    impl_->no_id = no_id > 0 ? no_id : 2152;
    impl_->use_hidden = (mode == "hidden");

    impl_->img_start = env_or("VLM_IMG_START", " vision_start");
    impl_->img_end = env_or("VLM_IMG_END", " vision_end");
    impl_->img_content = env_or("VLM_IMG_CONTENT", " vision");

    // hidden 模式权重：显式参数 > RERANKER_WEIGHT_BIN > exe 同目录
    if (impl_->use_hidden) {
        std::vector<std::string> candidates;
        if (!weight_bin.empty()) candidates.push_back(weight_bin);
        const char * env_w = getenv("RERANKER_WEIGHT_BIN");
        if (env_w && env_w[0]) candidates.push_back(env_w);
        char buf[4096];
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            std::string exe_dir(buf);
            auto slash = exe_dir.rfind('/');
            if (slash != std::string::npos) exe_dir = exe_dir.substr(0, slash);
            candidates.push_back(exe_dir + "/rerank_yes_no_diff.bin");
        }
        for (const auto & p : candidates) {
            std::ifstream f(p, std::ios::binary | std::ios::ate);
            if (!f) continue;
            size_t bytes = (size_t)f.tellg();
            if (bytes >= 2048 * 4 && bytes % 4 == 0) {
                impl_->wdiff.resize(bytes / 4);
                f.seekg(0);
                f.read((char *)impl_->wdiff.data(), (std::streamsize)bytes);
                printf("RerankerScorer: W_yes-W_no <- %s (%zu floats)\n", p.c_str(), impl_->wdiff.size());
                break;
            }
        }
        if (impl_->wdiff.empty()) {
            fprintf(stderr, "RerankerScorer: hidden 模式缺少 W_yes-W_no 权重向量，初始化失败\n");
            return;
        }
    }

    const char * cores_env = getenv("RKLLM_NPU_CORE_NUM");
    int core_num = (cores_env && atoi(cores_env) > 0) ? atoi(cores_env) : 3;

    // ---- LLM ----
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = impl_->llm_path.c_str();
    param.max_context_len = impl_->n_ctx;
    param.max_new_tokens = 1;              // 不消费生成文本，仅取 logits/hidden
    param.top_k = 1;
    param.temperature = 0.0f;
    param.repeat_penalty = 1.0f;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;

    RKLLMCallback rkllm_cb = {};
    rkllm_cb.result_callback = &Impl::cb;

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
    int embed_len = impl_->vision.model_image_token *
                    (int)impl_->vision.io_num.n_output *
                    impl_->vision.model_embed_size;
    impl_->model_w = impl_->vision.model_width;
    impl_->model_h = impl_->vision.model_height;
    impl_->embed.assign((size_t)embed_len, 0.0f);
    impl_->ready = true;
    printf("RerankerScorer(%s) ready: vision=%dx%d tokens=%zu llm=%s yes=%ld no=%ld\n",
           impl_->use_hidden ? "hidden" : "logits",
           impl_->model_w, impl_->model_h, impl_->n_image_tokens,
           impl_->llm_path.c_str(), impl_->yes_id, impl_->no_id);
}

RerankerScorer::~RerankerScorer() = default;
bool RerankerScorer::ok() const { return impl_->ready; }

double RerankerScorer::score(const ImageRGB & img, const std::string & text) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!impl_->ready) return -1.0;

    // ROI -> 固定分辨率（与 VlmDescriber 一致，双线性）
    ImageRGB resized = (img.w == impl_->model_w && img.h == impl_->model_h)
                           ? img
                           : resize_bilinear(img, impl_->model_w, impl_->model_h);
    size_t rgb_bytes = resized.rgb.size();
    if (rgb_bytes != (size_t)impl_->model_w * impl_->model_h * 3) return -1.0;

    bool hit = impl_->cache_valid && impl_->last_rgb.size() == rgb_bytes &&
               memcmp(impl_->last_rgb.data(), resized.rgb.data(), rgb_bytes) == 0;
    if (!hit) {
        if (run_imgenc(&impl_->vision, resized.rgb.data(), impl_->embed.data()) != 0) {
            fprintf(stderr, "run_imgenc fail\n");
            return -1.0;
        }
        impl_->last_rgb.assign(resized.rgb.begin(), resized.rgb.end());
        impl_->cache_valid = true;
    }

    // HF 渲染顺序：system 句由 rkllm role=system 注入；user 侧 = Instruct/Query/Document。
    // 图片占位默认文首（vlm_describer 同款约定）；RERANKER_IMG_MID=1 时插到 <Document>: 之后。
    std::string body = "<Instruct>: " + impl_->instruction +
                       "<Query>:" + text + "\n<Document>:";
    bool img_mid = getenv_bool("RERANKER_IMG_MID", false);
    std::string full_prompt = img_mid ? body + "<image>" : "<image>" + body;

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

    RKLLMInferParam infer = {};
    infer.mode = impl_->use_hidden ? RKLLM_INFER_GET_LAST_HIDDEN_LAYER : RKLLM_INFER_GET_LOGITS;
    infer.keep_history = 0;
    infer.max_new_tokens = 1;

    impl_->out_ok = false;
    impl_->logit_yes = impl_->logit_no = 0;
    impl_->hidden.clear();
    int ret = rkllm_run(impl_->llm, &input, &infer, impl_.get());
    if (ret != 0 || !impl_->out_ok) {
        fprintf(stderr, "reranker rkllm_run fail! ret=%d out_ok=%d\n", ret, (int)impl_->out_ok);
        return -1.0;
    }

    if (!impl_->use_hidden) return sigmoidf_(impl_->logit_yes - impl_->logit_no);
    if (impl_->hidden.size() != impl_->wdiff.size()) {
        fprintf(stderr, "reranker hidden dim=%zu != wdiff=%zu\n",
                impl_->hidden.size(), impl_->wdiff.size());
        return -1.0;
    }
    double dot = 0;
    for (size_t i = 0; i < impl_->hidden.size(); i++) dot += impl_->hidden[i] * impl_->wdiff[i];
    return sigmoidf_(dot);
}
