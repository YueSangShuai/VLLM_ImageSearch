// rerank_bench 扩展：验证 C++ 服务可行的 PROMPT 路径
//  模式A (--tokens file): TOKEN 输入（已验证基准 P={0.046,0.9947,0.9989,0.576,0.0255,0.0009}）
//  模式B (--prompts file): PROMPT 输入 + set_chat_template("","","")，
//      文件内每行 = 主机拼好的完整 ChatML 文本（含 im_start/im_end，\n 用字面 \\n）
//      → 若与模式A一致，服务侧 C++ 纯字符串拼接即可，无需 tokenizer。
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "rkllm.h"

static float g_p_yes = -1.0f;
static float g_pre_ms = 0, g_gen_ms = 0;
static int g_yes_id = 9693, g_no_id = 2152;

static int cb(RKLLMResult* result, void* ud, LLMCallState state) {
    if (result && result->perf.prefill_tokens) {
        g_pre_ms = result->perf.prefill_time_ms;
    }
    if (result && result->logits.logits && result->logits.vocab_size) {
        int V = result->logits.vocab_size;
        const float* row = result->logits.logits +
                           (size_t)(result->logits.num_tokens - 1) * V;
        g_p_yes = 1.0f / (1.0f + expf(row[g_no_id] - row[g_yes_id]));
    }
    return 0;
}

static std::string unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 'n') { o += '\n'; ++i; }
        else o += s[i];
    return o;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.rkllm --tokens|--prompts file [yes no]\n", argv[0]);
        return 1;
    }
    bool token_mode = strcmp(argv[2], "--tokens") == 0;

    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path = argv[1];
    p.max_context_len = 512;
    p.max_new_tokens = 1;
    p.top_k = 1; p.temperature = 0; p.repeat_penalty = 1.0f;
    p.skip_special_token = true;
    p.extend_param.base_domain_id = 0;
    p.extend_param.embed_flash = 1;

    LLMHandle h = nullptr;
    RKLLMCallback cbs = {};
    cbs.result_callback = cb;
    if (rkllm_init(&h, &p, &cbs) != 0) { fprintf(stderr, "init failed\n"); return 2; }

    if (!token_mode) {
        int rt = rkllm_set_chat_template(h, "", "", "");
        printf("# set_chat_template ret=%d\n", rt);
    }

    RKLLMInferParam ip = {};
    ip.mode = RKLLM_INFER_GET_LOGITS;
    ip.keep_history = 0;
    ip.max_new_tokens = 0;

    std::ifstream fin(argv[3]);
    std::string line;
    int idx = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        g_p_yes = -1.0f; g_pre_ms = g_gen_ms = 0;
        RKLLMInput in = {};
        in.role = "user";
        in.enable_thinking = false;
        int ret;
        if (token_mode) {
            auto tab = line.find('\t');
            std::istringstream iss(line.substr(tab + 1));
            static std::vector<int32_t> ids; ids.clear();
            long v; while (iss >> v) ids.push_back((int32_t)v);
            in.input_type = RKLLM_INPUT_TOKEN;
            in.token_input.input_ids = ids.data();
            in.token_input.n_tokens = ids.size();
        } else {
            static std::string text;
            text = unescape(line);
            in.input_type = RKLLM_INPUT_PROMPT;
            in.prompt_input = text.c_str();
        }
        auto s0 = std::chrono::steady_clock::now();
        ret = rkllm_run(h, &in, &ip, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        printf("%-8s %d prefill_ms=%.1f wall_ms=%.1f P(yes)=%.4f ret=%d\n",
               token_mode ? "token" : "prompt", idx++,
               g_pre_ms, std::chrono::duration<double, std::milli>(s1 - s0).count(),
               g_p_yes, ret);
    }
    rkllm_destroy(h);
    return 0;
}
