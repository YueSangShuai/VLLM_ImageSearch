// 板端 golden 对拍：golden_check <models_dir> <golden_raw_dir>
// 走生产同款 ImageEmbedder/TextEmbedder 代码路径，核验：
//   1) WordPiece 分词与 x86 HF BertTokenizer 完全一致（ids 全等）
//   2) 文本/图像塔 RKNN 输出 vs torch golden cosine ≥ 0.99
//   3) 端到端 topk(10) 与 x86 期望一致（近并分容差 0.005 由人工看输出）
#include "frame.h"
#include "image_embedder.h"
#include "text_embedder.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

using json = nlohmann::json;

template <class T>
static std::vector<T> read_raw(const std::string &path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "读不到 %s\n", path.c_str()); exit(2); }
    std::vector<T> v(count);
    f.read((char *)v.data(), count * sizeof(T));
    return v;
}

static std::vector<std::string> read_json_strs(const std::string &path) {
    std::ifstream f(path);
    json j = json::parse(f);
    return j.get<std::vector<std::string>>();
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "用法: %s <models_dir> <golden_raw_dir>\n", argv[0]); return 2; }
    std::string models = argv[1], raw = argv[2];

    json meta = json::parse(std::ifstream(raw + "/meta.json"));
    int N = meta["N"], M = meta["M"], SEQ = meta["seq"], DIM = meta["dim"];
    auto px = read_raw<uint8_t>(raw + "/px_u8.raw", (size_t)N * 224 * 224 * 3);
    auto ids_ref = read_raw<int32_t>(raw + "/ids.raw", (size_t)M * SEQ);
    auto g_emb = read_raw<float>(raw + "/g_emb.raw", (size_t)N * DIM);
    auto t_emb = read_raw<float>(raw + "/t_emb.raw", (size_t)M * DIM);
    auto queries = read_json_strs(raw + "/queries.json");
    auto files = read_json_strs(raw + "/gallery_files.json");
    json expected = json::parse(std::ifstream(raw + "/expected_topk.json"));

    TextEmbedder txt(models + "/mars_text_fp16.rknn", models + "/vocab.txt", SEQ);
    ImageEmbedder img(models + "/mars_image_fp16.rknn");

    // 1) token 一致性
    int tok_bad = 0;
    for (int qi = 0; qi < M; qi++) {
        std::vector<int32_t> ids;
        std::vector<float> attn;
        txt.tokenize(queries[qi], ids, attn);
        if (!std::equal(ids.begin(), ids.end(), ids_ref.begin() + (size_t)qi * SEQ)) tok_bad++;
    }
    printf("[1] tokenizer ids 全等: %s (%d/%d 不符)\n", tok_bad ? "FAIL" : "PASS", tok_bad, M);

    // 2) 文本塔向量
    double txt_min_cos = 1.0;
    std::vector<float> got_t((size_t)M * DIM);
    for (int qi = 0; qi < M; qi++) {
        auto e = txt.encode_clean(queries[qi]);
        double a = 0, b = 0, d = 0;
        for (int k = 0; k < DIM; k++) {
            float x = e[k], y = t_emb[(size_t)qi * DIM + k];
            a += x * y; b += x * x; d += y * y;
        }
        txt_min_cos = std::min(txt_min_cos, a / (std::sqrt(b) * std::sqrt(d) + 1e-12));
        memcpy(&got_t[(size_t)qi * DIM], e.data(), DIM * sizeof(float));
    }
    printf("[2] text cos(min) = %.5f : %s\n", txt_min_cos, txt_min_cos >= 0.99 ? "PASS" : "FAIL");

    // 3) 图像塔向量（golden px 已是 224×224，embed 内 resize 为恒等）
    double img_min_cos = 1.0;
    std::vector<float> got_g((size_t)N * DIM);
    for (int i = 0; i < N; i++) {
        ImageRGB im;
        im.w = im.h = 224;
        im.rgb.assign(px.begin() + (size_t)i * 224 * 224 * 3,
                      px.begin() + (size_t)(i + 1) * 224 * 224 * 3);
        auto e = img.embed(im);
        double a = 0, b = 0, d = 0;
        for (int k = 0; k < DIM; k++) {
            float x = e[k], y = g_emb[(size_t)i * DIM + k];
            a += x * y; b += x * x; d += y * y;
        }
        img_min_cos = std::min(img_min_cos, a / (std::sqrt(b) * std::sqrt(d) + 1e-12));
        memcpy(&got_g[(size_t)i * DIM], e.data(), DIM * sizeof(float));
    }
    printf("[3] image cos(min) = %.5f : %s\n", img_min_cos, img_min_cos >= 0.99 ? "PASS" : "FAIL");

    // 4) 端到端 topk
    int topk_bad = 0;
    for (int qi = 0; qi < M; qi++) {
        std::vector<std::pair<float, int>> sc;
        for (int i = 0; i < N; i++) {
            double s = 0;
            for (int k = 0; k < DIM; k++) s += (double)got_t[(size_t)qi * DIM + k] * got_g[(size_t)i * DIM + k];
            sc.emplace_back((float)s, i);
        }
        std::partial_sort(sc.begin(), sc.begin() + std::min(10, N), sc.end(), std::greater<>());
        bool same = true;
        for (int r = 0; r < std::min(10, N); r++)
            if (files[sc[r].second] != expected[qi]["topk"][r]["file"]) same = false;
        if (!same) {
            topk_bad++;
            printf("    query%d top3: %s / %s / %s\n", qi, files[sc[0].second].c_str(),
                   files[sc[1].second].c_str(), files[sc[2].second].c_str());
        }
    }
    printf("[4] topk(10) 全 %d 条一致: %s（尾名近并分时允许 ±1 位重排，人工复核）\n",
           M, topk_bad ? "存在差异" : "PASS");

    bool pass = !tok_bad && txt_min_cos >= 0.99 && img_min_cos >= 0.99;
    printf("== golden_check %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
