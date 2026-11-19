#include "reranker.h"

#include "rkllm.h"

#include <cmath>
#include <cstdio>

// ChatML 特殊标记：源码拆段字面量（编译期拼接），完整标记会干扰文本处理工具链。
static const char *K_IM_START = "<" "|im_start|" ">";
static const char *K_IM_END   = "<" "|im_end|" ">";
static const char *K_TH_START = "<" "|think" ">";
static const char *K_TH_END   = "<" "|/think" ">";

// 复刻 Qwen3-Reranker-0.6B 自带 chat_template.jinja（纯静态布局）：
//   system 判定提示 / user 内 <Instruct>/<Query>/<Document> 三行 /
//   assistant 前带空 think 块（关思考）。实测与 TOKEN 输入同分才允许启用。
static std::string buildChatML(const std::string & instruction,
                               const std::string & query,
                               const std::string & document) {
    std::string s;
    s += K_IM_START; s += "system\n";
    s += "Judge whether the Document meets the requirements based on the "
         "Query and the Instruct provided. Note that the answer can only be "
         "\"yes\" or \"no\".";
    s += K_IM_END;   s += "\n";
    s += K_IM_START; s += "user\n";
    s += "<Instruct>: " + instruction + "\n";
    s += "<Query>: "    + query + "\n";
    s += "<Document>: "  + document;
    s += K_IM_END;   s += "\n";
    s += K_IM_START; s += "assistant\n";
    s += K_TH_START; s += "\n\n"; s += K_TH_END; s += "\n\n";
    return s;
}

// yes/no token id 由主机 tokenizer 固定（Qwen3 词表）；换模型须重验。
static const int YES_ID = 9693;
static const int NO_ID  = 2152;

namespace {
struct structCb {
    float p_yes;
    bool  got;
};

int rerank_cb(RKLLMResult *result, void *ud, LLMCallState state) {
    auto *cb = static_cast<structCb *>(ud);
    if (result && result->logits.logits && result->logits.vocab_size > 0) {
        int V = result->logits.vocab_size;
        const float *row = result->logits.logits +
                           (size_t)(result->logits.num_tokens - 1) * V;
        cb->p_yes = 1.0f / (1.0f + std::exp(row[NO_ID] - row[YES_ID]));
        cb->got = true;
    }
    return 0;
}
}  // namespace

struct Reranker::Impl {
    LLMHandle handle = nullptr;
    std::string chatml_instruction;   // init 后模板置空，prompt 原文进模型
};

Reranker::Reranker(const std::string & model_path, double default_threshold)
    : impl_(new Impl), threshold_(default_threshold) {
    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path = model_path.c_str();
    p.max_context_len = 512;         // 板端实测 prompt ~115 token
    p.max_new_tokens = 1;
    p.top_k = 1;
    p.temperature = 0.0f;
    p.repeat_penalty = 1.0f;
    p.skip_special_token = true;
    p.extend_param.base_domain_id = 1;   // 与 VlmDescriber 同域（monitor 约定）
    p.extend_param.embed_flash = 1;

    RKLLMCallback cbs = {};
    cbs.result_callback = rerank_cb;
    int ret = rkllm_init(&impl_->handle, &p, &cbs);
    if (ret != 0) {
        printf("Reranker: rkllm_init 失败 ret=%d（模型: %s），二次核验停用\n",
               ret, model_path.c_str());
        impl_->handle = nullptr;
        return;
    }
    // 置空 runtime 模板：该 checkpoint 模板按 query/document 角色取值，
    // runtime 的 user 包装会静默丢内容（主机实测 P 恒 0.0043），
    // 故 ChatML 全串由 buildChatML 自拼，runtime 只做分词+前向。
    int rt = rkllm_set_chat_template(impl_->handle, "", "", "");
    if (rt != 0)
        printf("Reranker: set_chat_template ret=%d（继续，若与基准分数不符须查）\n", rt);
    ok_ = true;
    printf("Reranker(RKLLM) ready: %s (默认阈值 %.2f)\n", model_path.c_str(), threshold_);
}

Reranker::~Reranker() {
    if (impl_ && impl_->handle) rkllm_destroy(impl_->handle);
}

bool Reranker::ok() const { return ok_; }
double Reranker::defaultThreshold() const { return threshold_; }

double Reranker::rerank(const std::string & query, const std::string & document) {
    if (!ok_) return -1.0;
    std::lock_guard<std::mutex> lk(mu_);

    std::string text = buildChatML(
        "Given a pedestrian rule, retrieve person descriptions that satisfy "
        "the rule. A description satisfies the rule if it contains all "
        "attributes required by the rule (extra attributes are allowed).",
        query, document);

    structCb cb = {-1.0f, false};

    RKLLMInput in = {};
    in.input_type = RKLLM_INPUT_PROMPT;
    in.role = "user";
    in.enable_thinking = false;
    in.prompt_input = text.c_str();

    RKLLMInferParam ip = {};
    ip.mode = RKLLM_INFER_GET_LOGITS;
    ip.keep_history = 0;
    ip.max_new_tokens = 0;           // 纯 prefill，不生成

    // userdata 由 rkllm_run 末参透传到 init 注册的 result_callback
    int ret = rkllm_run(impl_->handle, &in, &ip, &cb);
    if (ret != 0 || !cb.got) {
        printf("Reranker: run 失败 ret=%d，按放行处理\n", ret);
        return -1.0;
    }
    return cb.p_yes;
}
