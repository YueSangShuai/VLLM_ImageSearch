// 板端 Qwen3-Reranker-0.6B w8a8 速度+正确性 bench（RK3588 NPU, rkllm 1.3.1）
// 输入文件每行: label \t id1 id2 id3 ...   （主机用同款 chat template 预渲染+分词）
// 用法: ./rerank_bench model.rkllm tokens.txt [yes_id no_id]
// 口径: GET_LOGITS 模式，一次 prefill 出末位 logits，softmax(yes/no) 即 P(yes)。
// 计时: result->perf 的 prefill/generate ms + 墙钟。
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
static int g_ptok = 0, g_gtok = 0;
static float g_mem = 0;
static int g_yes_id = 9693, g_no_id = 2152;

static int cb(RKLLMResult* result, void* ud, LLMCallState state) {
    if (result && result->perf.prefill_tokens) {
        g_pre_ms = result->perf.prefill_time_ms;
        g_ptok = result->perf.prefill_tokens;
        g_gen_ms = result->perf.generate_time_ms;
        g_gtok = result->perf.generate_tokens;
        g_mem = result->perf.memory_usage_mb;
    }
    if (result && result->logits.logits && result->logits.vocab_size) {
        int V = result->logits.vocab_size;
        const float* row = result->logits.logits +
                           (size_t)(result->logits.num_tokens - 1) * V;
        float a = row[g_no_id], b = row[g_yes_id];
        g_p_yes = 1.0f / (1.0f + expf(a - b));
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.rkllm tokens.txt [yes no]\n", argv[0]); return 1; }
    if (argc > 4) { g_yes_id = atoi(argv[3]); g_no_id = atoi(argv[4]); }

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
    auto t0 = std::chrono::steady_clock::now();
    int ret = rkllm_init(&h, &p, &cbs);
    auto t1 = std::chrono::steady_clock::now();
    if (ret != 0) { fprintf(stderr, "rkllm_init failed %d\n", ret); return 2; }
    printf("# init_ms=%.0f\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    RKLLMInferParam ip = {};
    ip.mode = RKLLM_INFER_GET_LOGITS;
    ip.keep_history = 0;
    ip.max_new_tokens = 0;   // 不生成，纯 prefill

    std::ifstream fin(argv[2]);
    std::string line;
    double total_ms = 0; int n = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto tab = line.find('\t');
        std::string label = line.substr(0, tab);
        std::vector<int32_t> ids;
        std::istringstream iss(line.substr(tab + 1));
        long v;
        while (iss >> v) ids.push_back((int32_t)v);

        RKLLMInput in = {};
        in.input_type = RKLLM_INPUT_TOKEN;
        in.role = "user";
        in.enable_thinking = false;
        in.token_input.input_ids = ids.data();
        in.token_input.n_tokens = ids.size();

        g_p_yes = -1.0f; g_pre_ms = g_gen_ms = 0; g_ptok = g_gtok = 0;
        auto s0 = std::chrono::steady_clock::now();
        ret = rkllm_run(h, &in, &ip, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(s1 - s0).count();
        total_ms += wall; n++;
        printf("%-4s ntok=%d prefill_ms=%.1f gen_ms=%.1f wall_ms=%.1f P(yes)=%.4f ret=%d mem=%.0fMB\n",
               label.c_str(), (int)ids.size(), g_pre_ms, g_gen_ms, wall, g_p_yes, ret, g_mem);
    }
    if (n) printf("# mean_wall_ms=%.0f over %d pairs\n", total_ms / n, n);
    rkllm_destroy(h);
    return 0;
}
