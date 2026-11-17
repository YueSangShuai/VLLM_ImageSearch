#pragma once
// 图库向量索引：目录扫描 → 增量编码 → append-only 向量库落盘缓存。
// data/gallery/vectors.bin（float32 追加写）+ manifest.json（文件→向量偏移），
// 删除的文件打 alive=false 墓碑（避免频繁重写），/api/rebuild 或启动时全量整理。
// 板端 10 万张全量编码约 2h，启动时后台增量、服务不等建库完成。
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

struct GalleryEntry {
    std::string file;      // 相对 GALLERY_DIR 的路径（响应/URL 用）
    int64_t mtime = 0;     // 秒级修改时间，变更检测
    int64_t off = -1;      // vectors.bin 中 float 偏移
    bool alive = true;
};

class Gallery {
public:
    using EncodeFn = std::function<std::vector<float>(const std::string &abs_path)>;

    Gallery(const std::string &gallery_dir, const std::string &cache_dir)
        : dir_(gallery_dir), cache_dir_(cache_dir) {}

    // 加载缓存 → 扫描目录增量编码新图/变更图。encode 回调可能耗时（NPU），
    // 编码在锁外做，仅追加时加锁。progress_cb(done,total) 可选。
    void sync(const EncodeFn &encode, const std::function<void(int, int)> &progress = nullptr);

    // 单张新图入库（/api/enroll）。返回 entry 下标；重复文件名自动加时间戳后缀。
    int add(const std::string &abs_path, const std::vector<float> &emb);

    // topk 检索：q（已归一化）× 全库点积。锁内拷贝结果快照，返回 (相对路径, 分数) 降序。
    std::vector<std::pair<std::string, float>> search(const std::vector<float> &q, int topk) const;

    int size() const { int n = 0; for (auto &e : entries_) if (e.alive) n++; return n; }
    int total() const { return (int)entries_.size(); }
    int dim() const { return dim_; }

private:
    void load_cache();
    void save_cache();

    std::string dir_, cache_dir_;
    int dim_ = 0;
    std::vector<GalleryEntry> entries_;
    std::vector<float> vecs_;                 // 展平 N×dim（append-only）
    mutable std::mutex mu_;
};
