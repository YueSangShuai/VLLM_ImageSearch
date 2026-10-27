// bench_models：独立基准测试程序，测量本框架全部模型在当前平台上的加载与推理耗时：
//   1) Detector      YOLOv8n 行人检测   （ONNX Runtime）
//   2) TextEmbedder  BGE 文本向量       （ONNX Runtime）
//   3) VlmDescriber  Qwen-VL 图像描述   （llama.cpp + libmtmd）
// 模型路径/参数与在线服务完全同源（Config::load 读同一批环境变量），
// 推荐用部署目录下的 run_bench.sh 启动（env 与 run_web.sh 对齐）。
//
// 用法：bench_models [--image 文件或目录] [--iters N] [--warmup N]
//                    [--vlm-max-tokens N] [--skip yolo,bge,vlm] [--help]
// 未指定 --image 时使用合成图：YOLO 用 1280x720 全帧，VLM 用 256x256（接近线上 ROI 裁剪尺寸）。
#include "config.h"
#include "detector.h"
#include "frame.h"
#include "text_embedder.h"
#include "util.h"
#include "vlm_describer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

// ==================== 计时统计 ====================

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stats {
    std::vector<double> samples;  // 每次迭代耗时 ms（不含 warmup）
    void add(double ms) { samples.push_back(ms); }
    size_t n() const { return samples.size(); }
    double pct(double p) const {  // p ∈ [0,1]
        if (samples.empty()) return 0;
        std::vector<double> s = samples;
        std::sort(s.begin(), s.end());
        size_t i = (size_t)(p * (s.size() - 1) + 0.5);
        return s[i];
    }
    double mean() const {
        if (samples.empty()) return 0;
        double t = 0; for (double v : samples) t += v;
        return t / samples.size();
    }
    double min_ms() const { return pct(0.0); }
    double max_ms() const { return pct(1.0); }
};

// 汇总表格用
struct SummaryRow {
    std::string model;
    double load_ms = 0;
    double infer_avg_ms = 0;
    std::string extra;
};
static std::vector<SummaryRow> g_summary;

static void print_stats(const char *label, const Stats &st, const char *unit_extra = "") {
    if (st.n() == 0) return;
    printf("  %-22s avg=%9.1f ms  min=%9.1f  max=%9.1f  p50=%9.1f%s\n",
           label, st.mean(), st.min_ms(), st.max_ms(), st.pct(0.5), unit_extra);
}

// ==================== 输入图像 ====================

// 合成渐变图 + 若干色块：保证像素有信息量（避免全黑图带来的误导性缓存效应）
static ImageRGB make_synthetic(int w, int h) {
    ImageRGB img;
    img.w = w; img.h = h;
    img.rgb.assign((size_t)w * h * 3, 0);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = ((size_t)y * w + x) * 3;
            img.rgb[i]     = (uint8_t)((x * 255) / (w > 1 ? w - 1 : 1));
            img.rgb[i + 1] = (uint8_t)((y * 255) / (h > 1 ? h - 1 : 1));
            img.rgb[i + 2] = (uint8_t)(((x + y) * 255) / (w + h > 2 ? w + h - 2 : 1));
        }
    }
    // 固定种子撒几个"行人"大小的色块
    srand(42);
    for (int b = 0; b < 6; b++) {
        int bx = rand() % (w > 40 ? w - 40 : 1), by = rand() % (h > 100 ? h - 100 : 1);
        int bw = 30 + rand() % 60, bh = 90 + rand() % 100;
        uint8_t r = (uint8_t)(rand() % 256), g = (uint8_t)(rand() % 256), bl = (uint8_t)(rand() % 256);
        for (int y = by; y < std::min(by + bh, h); y++)
            for (int x = bx; x < std::min(bx + bw, w); x++) {
                size_t i = ((size_t)y * img.w + x) * 3;
                img.rgb[i] = r; img.rgb[i + 1] = g; img.rgb[i + 2] = bl;
            }
    }
    return img;
}

static bool load_image_file(const std::string &path, ImageRGB &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    if (bytes.empty()) return false;
    return decode_jpeg(bytes, out);  // stb 解码，同时支持 jpg/png/bmp
}

// --image 为文件/目录 -> 图像列表；空 -> 用合成图
static std::vector<ImageRGB> collect_images(const std::string &arg, int def_w, int def_h) {
    std::vector<ImageRGB> imgs;
    if (arg.empty()) {
        imgs.push_back(make_synthetic(def_w, def_h));
        return imgs;
    }
    fs::path p(arg);
    std::vector<fs::path> files;
    if (fs::is_directory(p)) {
        for (auto &e : fs::directory_iterator(p)) {
            std::string ext = e.path().extension().string();
            for (auto &c : ext) c = (char)tolower(c);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(p);
    }
    for (auto &f : files) {
        ImageRGB img;
        if (load_image_file(f.string(), img)) {
            imgs.push_back(std::move(img));
        } else {
            printf("  ⚠️ 无法解码图像，已跳过: %s\n", f.string().c_str());
        }
    }
    if (imgs.empty()) throw std::runtime_error("--image 指定的路径下没有可用图像: " + arg);
    return imgs;
}

// ==================== 各模型基准 ====================

static void require_file(const std::string &path, const char *what) {
    if (path.empty() || !fs::exists(path))
        throw std::runtime_error(std::string(what) + " 不存在: " + (path.empty() ? "(未配置)" : path));
}

// YOLOv8n：全帧 letterbox + 推理 + NMS
static bool bench_yolo(const Config &cfg, const std::vector<ImageRGB> &imgs,
                       int iters, int warmup) {
    printf("\n[1/3] YOLOv8n 行人检测 (%s)\n", cfg.yolo_onnx_path.c_str());
    require_file(cfg.yolo_onnx_path, "YOLO ONNX");
    auto t0 = now_ms();
    Detector detector(cfg.yolo_onnx_path, cfg.det_conf, cfg.det_iou);
    double ms = now_ms() - t0;
    printf("  模型加载: %.1f ms\n", ms);

    Stats st;
    size_t last_count = 0;
    size_t total_iter = (size_t)warmup + iters;
    for (size_t i = 0; i < total_iter; i++) {
        const ImageRGB &img = imgs[i % imgs.size()];
        double a = now_ms();
        auto dets = detector.detect(img);
        double dt = now_ms() - a;
        if ((int)i >= warmup) { st.add(dt); last_count = dets.size(); }
    }
    char extra[128];
    snprintf(extra, sizeof(extra), "   (%zu 张图轮测, 末次检出 %zu 个目标)", imgs.size(), last_count);
    print_stats("单帧检测", st, extra);
    printf("  吞吐: %.2f 帧/秒（单线程串行）\n", 1000.0 / st.mean());
    g_summary.push_back({"YOLOv8n 检测", ms, st.mean(),
                         std::to_string(1000.0 / st.mean()).substr(0, 6) + " fps"});
    return true;
}

// BGE：文本 -> 归一化向量（线上每帧对若干条描述句编码）
static bool bench_bge(const Config &cfg, int iters, int warmup) {
    printf("\n[2/3] BGE 文本向量 (%s)\n", cfg.bge_onnx_path.c_str());
    require_file(cfg.bge_onnx_path, "BGE ONNX");
    require_file(cfg.bge_vocab_path, "BGE vocab");
    auto t0 = now_ms();
    TextEmbedder embedder(cfg.bge_onnx_path, cfg.bge_vocab_path, cfg.bge_max_length);
    double ms = now_ms() - t0;
    printf("  模型加载: %.1f ms\n", ms);

    // 与线上一致：单条 clothing 句子批量编码
    std::vector<std::string> texts = {
        "上身穿着白色短袖T恤",
        "下身是一条黑色长裤，脚穿白色运动鞋",
        "背着一个红色的双肩背包",
        "上身为蓝色拉链外套，下摆有反光条",
        "老年男性，灰色夹克，双手插在口袋里",
    };

    Stats batch, single;
    for (int i = 0; i < warmup + iters; i++) {
        double a = now_ms();
        embedder.encode(texts);
        double dt = now_ms() - a;
        if (i >= warmup) batch.add(dt);
    }
    char extra[64];
    snprintf(extra, sizeof(extra), "   (batch=%zu)", texts.size());
    print_stats("批量编码", batch, extra);
    printf("  单条平均: %.1f ms/条\n", batch.mean() / texts.size());

    for (int i = 0; i < warmup + iters; i++) {
        double a = now_ms();
        embedder.encode_one("上身穿着白色短袖T恤");
        double dt = now_ms() - a;
        if (i >= warmup) single.add(dt);
    }
    print_stats("单条编码", single);
    printf("  向量维度: %d\n", embedder.dim());
    g_summary.push_back({"BGE 文本向量", ms, single.mean(),
                         "batch " + std::to_string((int)batch.mean()) + " ms"});
    return true;
}

// Qwen-VL：图像 + 提示词 -> 描述文本（一次调用 = 视觉编码 + prefill + decode）
static bool bench_vlm(const Config &cfg, const std::vector<ImageRGB> &imgs,
                      int iters, int warmup, int max_tokens) {
    printf("\n[3/3] Qwen-VL 图像描述 (%s)\n  mmproj: %s\n",
           cfg.vlm_model_path.c_str(), cfg.mmproj_path.c_str());
    require_file(cfg.vlm_model_path, "VLM GGUF");
    require_file(cfg.mmproj_path, "mmproj GGUF");
    auto t0 = now_ms();
    VlmDescriber describer(cfg.vlm_model_path, cfg.mmproj_path, cfg.device,
                           cfg.n_gpu_layers, cfg.n_ctx,
                           cfg.vlm_temperature, cfg.vlm_repetition_penalty);
    double ms = now_ms() - t0;
    printf("  模型加载: %.1f ms (device=%s, n_ctx=%d)\n", ms, cfg.device.c_str(), cfg.n_ctx);
    if (!describer.ok()) throw std::runtime_error("VlmDescriber 初始化失败");

    std::string prompt = cfg.description_prompt;
    if (prompt.empty()) prompt = "Describe the pedestrian in the image.";

    Stats st;
    size_t chars_sum = 0;
    for (int i = 0; i < warmup + iters; i++) {
        const ImageRGB &img = imgs[i % imgs.size()];
        double a = now_ms();
        std::string out = describer.describe(img, prompt, max_tokens,
                                             cfg.vlm_temperature, cfg.vlm_repetition_penalty);
        double dt = now_ms() - a;
        if (i >= warmup) { st.add(dt); chars_sum += utf8_length(out); }
        if (i == warmup + iters - 1)
            printf("  末次输出(%zu字): %s\n", utf8_length(out),
                   utf8_substr(out, 80).c_str());
    }
    char extra[128];
    snprintf(extra, sizeof(extra), "   (max_tokens=%d, 输入%zu张图轮测)", max_tokens, imgs.size());
    print_stats("单帧生成", st, extra);
    double chars_per_s = chars_sum * 1000.0 / (st.mean() * st.n());
    printf("  平均输出 %.1f 字/次, 约 %.1f 字/秒（含视觉编码+prefill+decode）\n",
           (double)chars_sum / st.n(), chars_per_s);
    g_summary.push_back({"Qwen-VL 生成", ms, st.mean(),
                         std::to_string((int)chars_per_s) + " 字/秒"});
    return true;
}

// ==================== main ====================

static void usage(const char *argv0) {
    printf(R"(模型推理耗时基准测试（与在线服务同一套环境变量配置）

用法: %s [选项]
  --image PATH        测试图（文件或目录，jpg/png/bmp 轮测）；缺省用合成图
                      （YOLO 1280x720 全帧；VLM 256x256，接近线上 ROI 尺寸）
  --iters N           每个模型的正式计时次数（默认 3）
  --warmup N          预热次数，不计入统计（默认 1）
  --vlm-max-tokens N  VLM 计时用最大生成 token（默认取环境变量 VLM_MAX_NEW_TOKENS）
  --skip LIST         跳过模型，逗号分隔: yolo,bge,vlm（如 --skip vlm）
  --help              显示本帮助

模型路径由环境变量指定（同 run_web.sh）：DETECTOR_ONNX_PATH / EMBEDDING_ONNX_PATH /
EMBEDDING_VOCAB_PATH / VLM_MODEL_NAME / VLM_MMPROJ_PATH / VLM_DEVICE 等。
推荐直接在部署目录运行 ./run_bench.sh（自动对齐 run_web.sh 的全部环境变量）。
)", argv0);
}

int main(int argc, char **argv) {
    std::string image_arg, skip_list;
    int iters = 3, warmup = 1, vlm_max_tokens = -1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s 缺少参数\n", name); exit(2); }
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--image")           image_arg = next("--image");
        else if (a == "--skip")            skip_list = next("--skip");
        else if (a == "--iters")           iters = atoi(next("--iters").c_str());
        else if (a == "--warmup")          warmup = atoi(next("--warmup").c_str());
        else if (a == "--vlm-max-tokens")  vlm_max_tokens = atoi(next("--vlm-max-tokens").c_str());
        else { fprintf(stderr, "未知参数: %s（--help 查看用法）\n", a.c_str()); return 2; }
    }
    iters = std::max(1, iters);
    warmup = std::max(0, warmup);
    auto skip_items = split_any(skip_list, ",;");
    auto skipped = [&](const char *name) {
        for (auto &s : skip_items) if (s == name) return true;
        return false;
    };

    Config cfg = Config::load();
    if (vlm_max_tokens <= 0) vlm_max_tokens = cfg.vlm_max_new_tokens;

    printf("======== web_vllm_cpp 模型推理耗时基准 ========\n");
    printf("平台: %ld 核 | device=%s | match_mode=%s\n",
           sysconf(_SC_NPROCESSORS_ONLN), cfg.device.c_str(), cfg.match_mode.c_str());

    bool failed = false;
    double t_start = now_ms();

    // 输入图像分别取默认尺寸：检测用全帧，VLM 用接近线上 ROI 的小图
    std::vector<ImageRGB> det_imgs, vlm_imgs;
    try {
        if (!skipped("yolo"))
            det_imgs = collect_images(image_arg, 1280, 720);
        if (!skipped("vlm"))
            vlm_imgs = image_arg.empty() ? collect_images("", 256, 256) : det_imgs;
    } catch (const std::exception &e) {
        fprintf(stderr, "图像输入失败: %s\n", e.what());
        return 1;
    }

    // 各模型独立 try：单模型失败不影响其余计时
    if (!skipped("yolo")) {
        try { bench_yolo(cfg, det_imgs, iters, warmup); }
        catch (const std::exception &e) { fprintf(stderr, "\n❌ YOLO 基准失败: %s\n", e.what()); failed = true; }
    } else printf("\n[1/3] YOLO —— 已跳过\n");

    if (!skipped("bge")) {
        try { bench_bge(cfg, iters, warmup); }
        catch (const std::exception &e) { fprintf(stderr, "\n❌ BGE 基准失败: %s\n", e.what()); failed = true; }
    } else printf("\n[2/3] BGE —— 已跳过\n");

    if (!skipped("vlm")) {
        try { bench_vlm(cfg, vlm_imgs, iters, warmup, vlm_max_tokens); }
        catch (const std::exception &e) { fprintf(stderr, "\n❌ VLM 基准失败: %s\n", e.what()); failed = true; }
    } else printf("\n[3/3] Qwen-VL —— 已跳过\n");

    printf("\n======== 汇总 ========\n");
    printf("%-16s %12s %12s  %s\n", "模型", "加载(ms)", "单次推理(ms)", "备注");
    for (auto &r : g_summary)
        printf("%-16s %12.0f %12.1f  %s\n", r.model.c_str(), r.load_ms, r.infer_avg_ms, r.extra.c_str());
    printf("总耗时 %.1f 秒\n", (now_ms() - t_start) / 1000.0);
    printf("注: VLM 单次含视觉编码+prefill+decode；线上 combined 模式每帧约 2 次生成调用，\n"
           "    rule_direct 约 1~2 次，实际帧耗时 ≈ 检测 + 向量 + 描述提示词对应的生成次数 × 上表值。\n");
    return failed ? 1 : 0;
}
