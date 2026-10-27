// yolo_probe：对图片目录逐张跑 Detector，打印检测框（golden 对齐用）
// 用法：yolo_probe <model.rknn|.onnx> <conf> <iou> <img|dir>...
#include "detector.h"
#include "frame.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static bool load_image(const std::string &path, ImageRGB &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    return decode_jpeg(bytes, out);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model> <conf> <iou> <img|dir>...\n", argv[0]);
        return 2;
    }
    float conf = atof(argv[2]), iou = atof(argv[3]);
    Detector det(argv[1], conf, iou);
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    for (int i = 4; i < argc; i++) {
        fs::path p(argv[i]);
        if (fs::is_directory(p)) {
            for (auto &e : fs::directory_iterator(p))
                if (e.path().extension() == ".jpg" || e.path().extension() == ".jpeg" ||
                    e.path().extension() == ".png")
                    files.push_back(e.path().string());
        } else
            files.push_back(argv[i]);
    }
    for (auto &f : files) {
        ImageRGB img;
        if (!load_image(f, img)) { printf("SKIP(load) %s\n", f.c_str()); continue; }
        auto dets = det.detect(img);
        printf("%s %dx%d\n", f.c_str(), img.w, img.h);
        for (auto &d : dets)
            printf("  [%d,%d,%d,%d] %.4f\n", d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                   d.confidence);
    }
    return 0;
}
