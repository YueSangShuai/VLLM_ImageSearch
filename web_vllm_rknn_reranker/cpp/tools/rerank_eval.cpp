// 板端离线打分：对 host_ref_rerank.py 产出的 host_scores.json 逐条重打分，
// 输出 board_scores.json（同字段 + board_score），用于主机/板端排序一致性与延迟评估。
// 用法: rerank_eval --list calib/host_scores.json [--out board_scores.json]
//       模型路径/参数走 RERANKER_* 环境变量（同在线服务）。
#include "config.h"
#include "frame.h"
#include "reranker_scorer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

static double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

int main(int argc, char ** argv) {
    std::string list_path, out_path = "board_scores.json";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list") && i + 1 < argc) list_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
    }
    if (list_path.empty()) {
        fprintf(stderr, "usage: rerank_eval --list host_scores.json [--out out.json]\n");
        return 1;
    }

    std::ifstream f(list_path);
    if (!f) { fprintf(stderr, "cannot read %s\n", list_path.c_str()); return 1; }
    json hj = json::parse(f);
    fs::path base = fs::path(list_path).parent_path();

    Config cfg = Config::load();
    // list 里记录的 instruction 优先（保证与主机参考同一 prompt）
    std::string instruction = hj.contains("instruction") && hj["instruction"].is_string()
                                  ? hj["instruction"].get<std::string>()
                                  : cfg.reranker_instruction;
    RerankerScorer scorer(cfg.reranker_vision_path, cfg.reranker_llm_path, cfg.n_ctx,
                          instruction, cfg.reranker_mode, cfg.reranker_weight_bin,
                          cfg.reranker_yes_id, cfg.reranker_no_id);
    if (!scorer.ok()) { fprintf(stderr, "RerankerScorer init failed\n"); return 2; }

    // 同图多条文本：解码一次、连续打分（吃满 scorer 的视觉缓存）
    json out = json::array();
    const auto & entries = hj["entries"];
    double total_ms = 0;
    size_t done = 0;
    size_t i = 0;
    while (i < entries.size()) {
        std::string img_rel = entries[i]["image"].get<std::string>();
        size_t j = i;
        while (j < entries.size() && entries[j]["image"] == img_rel) j++;

        fs::path img_path = base / img_rel;
        std::ifstream im(img_path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(im)),
                                   std::istreambuf_iterator<char>());
        ImageRGB img;
        if (!im || !decode_jpeg(bytes, img)) {
            fprintf(stderr, "SKIP(decode) %s\n", img_path.c_str());
            i = j;
            continue;
        }
        for (size_t k = i; k < j; k++) {
            auto t0 = std::chrono::steady_clock::now();
            double s = scorer.score(img, entries[k]["query_text"].get<std::string>());
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            if (k != i) total_ms += ms;   // 首条含视觉塔耗时，延迟只记纯 LLM prefill
            json e = entries[k];
            e["board_score"] = s < 0 ? json(nullptr) : json(round4(s));
            e["board_ms"] = ms;
            out.push_back(e);
            done++;
        }
        printf("[%zu/%zu] %s\n", done, entries.size(), img_rel.c_str());
        fflush(stdout);
        i = j;
    }

    double per = done ? total_ms / done : 0;
    printf("done: %zu scores, LLM per-call avg %.0f ms\n", done, per);
    json wrap;
    wrap["instruction"] = instruction;
    wrap["mode"] = cfg.reranker_mode;
    wrap["avg_llm_ms"] = per;
    wrap["entries"] = out;
    std::ofstream of(out_path);
    of << wrap << "\n";
    printf("wrote %s\n", out_path.c_str());
    return 0;
}
