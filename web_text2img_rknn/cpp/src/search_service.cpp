#include "search_service.h"

#include "frame.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

SearchService::SearchService(const std::string &image_rknn, const std::string &text_rknn,
                             const std::string &vocab_path, int text_max_len,
                             const std::string &gallery_dir, const std::string &cache_dir)
    : gallery_(gallery_dir, cache_dir) {
    img_ = std::make_unique<ImageEmbedder>(image_rknn);
    txt_ = std::make_unique<TextEmbedder>(text_rknn, vocab_path, text_max_len);
}

std::vector<float> SearchService::encode_file(const std::string &abs_path) {
    std::ifstream f(abs_path, std::ios::binary);
    if (!f) throw std::runtime_error("读不到文件: " + abs_path);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    ImageRGB img;
    if (!decode_jpeg(bytes, img))   // frame.h 的 stb 解码，png/jpg 通吃
        throw std::runtime_error("解码失败: " + abs_path);
    return img_->embed(img);
}

void SearchService::build() {
    std::lock_guard<std::mutex> lk(build_mu_);   // 串行化并发触发
    building_ = true;
    gallery_.sync([this](const std::string &p) { return encode_file(p); });
    building_ = false;
}

std::vector<std::pair<std::string, float>> SearchService::search(const std::string &text,
                                                                 int topk) {
    auto q = txt_->encode(text);
    return gallery_.search(q, topk);
}

std::string SearchService::enroll(const std::string &abs_path) {
    auto emb = encode_file(abs_path);
    int idx = gallery_.add(abs_path, emb);
    (void)idx;
    // 返回实际入库的相对路径（add 内不改名；重名覆盖由调用方生成唯一名保证）
    std::lock_guard<std::mutex> lk(build_mu_);
    return abs_path;
}
