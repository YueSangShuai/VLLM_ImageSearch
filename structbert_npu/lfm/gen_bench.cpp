// gen_bench: 同口径 LLM 塔生成测速（gecko-LM / LFM-LM / Qwen3-VL-2B 三个 .rkllm）。
// PROMPT 模式 + 空 chat template，文件内每行 = 主机预渲染好的完整 ChatML（\n 为字面 \\n）。
// 每行跑 n_runs 次 GENERATE（greedy, max_new_tokens 固定），回报 prefill/生成/内存。
// 编译（板上）：g++ -O2 gen_bench.cpp -o gen_bench -I<inc> -L<lib> -lrkllmrt
// 用法：./gen_bench model.rkllm prompts.txt [n_runs=3] [max_new=64] [max_ctx=1024]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include "rkllm.h"

static RKLLMPerfStat g_perf = {};
static int g_nruns = 3, g_max_new = 64;

static int cb(RKLLMResult* result, void* ud, LLMCallState state) {
    if (result) g_perf = result->perf;
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
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.rkllm prompts.txt [n_runs] [max_new] [max_ctx]\n", argv[0]);
        return 1;
    }
    if (argc > 3) g_nruns = atoi(argv[3]);
    if (argc > 4) g_max_new = atoi(argv[4]);
    int max_ctx = argc > 5 ? atoi(argv[5]) : 1024;

    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path = argv[1];
    p.max_context_len = max_ctx;
    p.max_new_tokens = g_max_new;
    p.top_k = 1; p.temperature = 0.0f; p.repeat_penalty = 1.0f;
    p.skip_special_token = true;
    p.extend_param.base_domain_id = 0;
    p.extend_param.embed_flash = 1;

    auto t0 = std::chrono::steady_clock::now();
    LLMHandle h = nullptr;
    RKLLMCallback cbs = {};
    cbs.result_callback = cb;
    if (rkllm_init(&h, &p, &cbs) != 0) { fprintf(stderr, "init failed\n"); return 2; }
    auto t1 = std::chrono::steady_clock::now();
    printf("# load_s=%.1f\n", std::chrono::duration<double>(t1 - t0).count());
    rkllm_set_chat_template(h, "", "", "");

    RKLLMInferParam ip = {};
    ip.mode = RKLLM_INFER_GENERATE;
    ip.keep_history = 0;
    ip.max_new_tokens = g_max_new;

    std::ifstream fin(argv[2]);
    std::string line;
    int idx = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        static std::string text;
        text = unescape(line);
        for (int r = 0; r < g_nruns; ++r) {
            memset(&g_perf, 0, sizeof(g_perf));
            auto s0 = std::chrono::steady_clock::now();
            RKLLMInput in = {};
            in.role = "user";
            in.enable_thinking = false;
            in.input_type = RKLLM_INPUT_PROMPT;
            in.prompt_input = text.c_str();
            int ret = rkllm_run(h, &in, &ip, nullptr);
            auto s1 = std::chrono::steady_clock::now();
            double wall = std::chrono::duration<double, std::milli>(s1 - s0).count();
            int gt = g_perf.generate_tokens > 0 ? g_perf.generate_tokens : g_max_new;
            printf("run %d prefill_tok=%d prefill_ms=%.1f gen_tok=%d gen_ms=%.1f "
                   "ms_per_tok=%.1f tok_per_s=%.1f wall_ms=%.1f mem=%.0fMB ret=%d\n",
                   r, g_perf.prefill_tokens, g_perf.prefill_time_ms, gt,
                   g_perf.generate_time_ms,
                   gt ? g_perf.generate_time_ms / gt : 0.0,
                   gt ? gt * 1000.0 / g_perf.generate_time_ms : 0.0,
                   wall, g_perf.memory_usage_mb, ret);
        }
        idx++;
    }
    rkllm_destroy(h);
    return 0;
}
