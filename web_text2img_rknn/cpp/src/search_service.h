#pragma once
// 检索服务核心：双塔常驻 + 图库索引 + 建库/检索/上传三个动作。
// main 启动序列：构造（加载 rknn 双塔）→ 后台线程 build() → server 立即可查
// （查到的是已入库部分，边建边查）。
#include "gallery.h"
#include "image_embedder.h"
#include "text_embedder.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class SearchService {
public:
    SearchService(const std::string &image_rknn, const std::string &text_rknn,
                  const std::string &vocab_path, int text_max_len,
                  const std::string &gallery_dir, const std::string &cache_dir);

    // 目录扫描 + 增量编码（后台线程调用；重入安全，同一时刻只跑一个）
    void build();
    bool building() const { return building_; }

    // text → topk {file(相对), score}
    std::vector<std::pair<std::string, float>> search(const std::string &text, int topk);

    // 上传图入库：abs_path 已落盘；返回 {file, score_self_match?}（重复名自动改名）
    std::string enroll(const std::string &abs_path);

    int gallery_size() const { return gallery_.size(); }
    int gallery_total() const { return gallery_.total(); }

private:
    std::vector<float> encode_file(const std::string &abs_path);

    std::unique_ptr<ImageEmbedder> img_;
    std::unique_ptr<TextEmbedder> txt_;
    Gallery gallery_;
    std::atomic<bool> building_{false};
    std::mutex build_mu_;
};
