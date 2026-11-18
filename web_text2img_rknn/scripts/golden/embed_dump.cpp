// 排障工具：embed_dump <models_dir> <jpg_path> [resized_out.raw]
// 打印图像塔 embedding 前 6 维；给了第三参则把 stb 解码+resize 后的 224×224 RGB 落 raw
#include "frame.h"
#include "image_embedder.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: %s <models_dir> <jpg> [resized.raw]\n", argv[0]); return 2; }
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ImageRGB img;
    if (!decode_jpeg(bytes, img)) { fprintf(stderr, "decode 失败\n"); return 1; }
    fprintf(stderr, "decoded %dx%d\n", img.w, img.h);
    ImageEmbedder ie(std::string(argv[1]) + "/mars_image_fp16.rknn");
    auto e = ie.embed(img);
    printf("emb[0:6]: ");
    for (int i = 0; i < 6; i++) printf("%.4f ", e[i]);
    printf("\n");
    if (argc > 3) {
        ImageRGB r224 = resize_bilinear(img, 224, 224);
        std::ofstream o(argv[3], std::ios::binary);
        o.write((const char *)r224.rgb.data(), r224.rgb.size());
        fprintf(stderr, "resized raw -> %s\n", argv[3]);
    }
    return 0;
}
