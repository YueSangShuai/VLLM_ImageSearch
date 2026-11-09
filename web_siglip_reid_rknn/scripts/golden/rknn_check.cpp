// 板端 golden 对拍：RKNN 视觉塔输出 vs x86 ONNX/torch 期望（scripts/golden/raw/）。
// 与在线服务同一 SiglipVision 代码路径（预处理归一化口径一并验证）。
// 构建（板上或交叉）：见 build_check.sh；用法：rknn_check <rknn路径> <raw目录>
#include "siglip_vision.h"
#include "frame.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static ImageRGB load_raw(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("打不开 " + path);
    ImageRGB img;
    img.w = img.h = 224;
    img.rgb.resize((size_t)224 * 224 * 3);
    f.read((char *)img.rgb.data(), (std::streamsize)img.rgb.size());
    if (!f) throw std::runtime_error("读不完整 " + path);
    return img;
}

static std::vector<float> load_f32(const std::string &path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("打不开 " + path);
    std::vector<float> v(n);
    f.read((char *)v.data(), (std::streamsize)(n * sizeof(float)));
    return v;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <siglip_vision.rknn> <raw目录>\n", argv[0]);
        return 2;
    }
    std::string raw = argv[2];
    std::ifstream mf(raw + "/meta.txt");
    if (!mf) { fprintf(stderr, "缺 %s/meta.txt\n", raw.c_str()); return 2; }

    try {
        SiglipVision vision(argv[1]);
        std::string line;
        int pass = 0, fail = 0;
        double worst = 1.0;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream is(line);
            int idx; double th;
            is >> idx >> th;
            auto img = load_raw(raw + "/input" + std::to_string(idx) + ".raw");
            auto expect = load_f32(raw + "/expect" + std::to_string(idx) + ".raw", 768);
            auto got = vision.embed(img);
            if (got.size() != expect.size()) {
                printf("case %d: 维度不一致 got=%zu expect=%zu FAIL\n", idx, got.size(), expect.size());
                fail++; continue;
            }
            double cos = 0;
            for (size_t k = 0; k < got.size(); k++) cos += (double)got[k] * expect[k];
            bool ok = cos >= th;
            worst = std::min(worst, cos);
            printf("case %d: cos=%.5f (th=%.3f) %s\n", idx, cos, th, ok ? "PASS" : "FAIL");
            ok ? pass++ : fail++;
        }
        printf("== %d PASS / %d FAIL，最差 cos=%.5f ==\n", pass, fail, worst);
        return fail ? 1 : 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "❌ %s\n", e.what());
        return 1;
    }
}
