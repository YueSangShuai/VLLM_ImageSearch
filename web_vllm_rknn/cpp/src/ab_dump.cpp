// ab_dump：批量生成工具（RK3588），与在线服务完全同一 VlmDescriber/预处理/采样链路。
// 对目录下每张图跑 prompts.json 里的每一组 (tag, prompt, penalty, max_tokens)，
// 逐行输出 JSONL：{"file":..., "tag":..., "raw":..., "head_attrs":[...], "sec":...}
// 用于 W8A8 vs host bf16 的 A/B 评测（scripts/eval/w8a8_ab/）。
// 环境变量与 run_web_rk.sh 同源（Config::load），建议 env -i 后手动 export 或经 run_bench 风格脚本启动。
//
// 用法: ab_dump --images DIR --out out.jsonl --prompts prompts.json [--limit N]
// prompts.json: [{"tag":"structured","penalty":1.4,"max_tokens":128,"prompt":"..."}]
#include "config.h"
#include "frame.h"
#include "util.h"
#include "vlm_describer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    std::string images_dir, out_path, prompts_path;
    int limit = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s 缺少参数\n", name); exit(2); }
            return argv[++i];
        };
        if      (a == "--images")  images_dir = next("--images");
        else if (a == "--out")     out_path = next("--out");
        else if (a == "--prompts") prompts_path = next("--prompts");
        else if (a == "--limit")   limit = atoi(next("--limit").c_str());
        else { fprintf(stderr, "未知参数: %s\n", a.c_str()); return 2; }
    }
    if (images_dir.empty() || out_path.empty() || prompts_path.empty()) {
        fprintf(stderr, "用法: ab_dump --images DIR --out out.jsonl --prompts prompts.json [--limit N]\n");
        return 2;
    }

    std::vector<fs::path> files;
    for (auto &e : fs::directory_iterator(images_dir)) {
        std::string ext = e.path().extension().string();
        for (auto &c : ext) c = (char)tolower(c);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (limit > 0 && (size_t)limit < files.size()) files.resize((size_t)limit);
    if (files.empty()) { fprintf(stderr, "目录无可用图像: %s\n", images_dir.c_str()); return 1; }

    json prompts;
    { std::ifstream f(prompts_path); if (!f) { fprintf(stderr, "无法读 %s\n", prompts_path.c_str()); return 1; }
      f >> prompts; }

    Config cfg = Config::load();
    printf("ab_dump: %zu 张图 x %zu 组 prompt | vision=%s llm=%s\n",
           files.size(), prompts.size(), cfg.vlm_model_path.c_str(), cfg.mmproj_path.c_str());

    VlmDescriber describer(cfg.vlm_model_path, cfg.mmproj_path, cfg.n_ctx,
                           cfg.vlm_temperature, cfg.vlm_repetition_penalty);
    if (!describer.ok()) { fprintf(stderr, "VlmDescriber 初始化失败\n"); return 1; }
    // 与线上一致：头启用时预处理走 expand2square+resize，同时顺带产出头属性
    describer.setAttrHead(cfg.attr_head_path);
    const bool head_on = describer.attrHeadReady();

    std::ofstream out(out_path, std::ios::app);
    for (size_t k = 0; k < files.size(); k++) {
        ImageRGB img;
        {   std::ifstream f(files[k].string(), std::ios::binary);
            std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                       std::istreambuf_iterator<char>());
            if (!decode_jpeg(bytes, img)) { printf("跳过（解码失败）%s\n", files[k].c_str()); continue; } }
        for (auto &p : prompts) {
            std::string tag = p.value("tag", "p");
            std::string prompt = p.value("prompt", "");
            double penalty = p.value("penalty", cfg.vlm_repetition_penalty);
            int max_tokens = p.value("max_tokens", cfg.vlm_max_new_tokens);
            double a = now_ms();
            std::vector<std::string> head_attrs;
            std::string raw = describer.describe(img, prompt, max_tokens,
                                                 cfg.vlm_temperature, penalty,
                                                 head_on ? &head_attrs : nullptr);
            json rec = {{"file", files[k].filename().string()},
                        {"tag", tag}, {"penalty", penalty},
                        {"raw", raw}, {"head_attrs", head_attrs},
                        {"sec", (now_ms() - a) / 1000.0}};
            out << rec.dump() << "\n";
            out.flush();
            printf("[%zu/%zu] %s tag=%s raw_len=%zu head=%zu %.1fs\n",
                   k + 1, files.size(), files[k].filename().c_str(), tag.c_str(),
                   utf8_length(raw), head_attrs.size(), rec["sec"].get<double>());
            fflush(stdout);
        }
    }
    printf("DONE -> %s\n", out_path.c_str());
    return 0;
}
