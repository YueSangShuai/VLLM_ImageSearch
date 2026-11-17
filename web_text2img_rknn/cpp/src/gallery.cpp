#include "gallery.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

using json = nlohmann::json;
namespace fs = std::filesystem;

static int64_t file_mtime(const fs::path &p) {
    auto t = fs::last_write_time(p);
    // file_time → unix 秒（C++20 clock_cast 不普及，用 now 校准一次即可，只做变更比较）
    return std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}

static bool is_image(const std::string &name) {
    auto dot = name.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = name.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "jpg" || ext == "jpeg" || ext == "png";
}

void Gallery::load_cache() {
    std::ifstream mf(cache_dir_ + "/manifest.json");
    if (!mf) return;
    json j;
    try { mf >> j; } catch (...) { return; }
    dim_ = j.value("dim", 0);
    entries_.clear();
    for (auto &e : j["entries"]) {
        GalleryEntry g;
        g.file = e["file"];
        g.mtime = e["mtime"];
        g.off = e["off"];
        g.alive = e.value("alive", true);
        entries_.push_back(std::move(g));
    }
    std::ifstream vf(cache_dir_ + "/vectors.bin", std::ios::binary);
    if (!vf || dim_ <= 0) { entries_.clear(); return; }
    vf.seekg(0, std::ios::end);
    size_t bytes = (size_t)vf.tellg();
    vf.seekg(0, std::ios::beg);
    vecs_.resize(bytes / sizeof(float));
    vf.read((char *)vecs_.data(), (std::streamsize)bytes);
    fprintf(stderr, "gallery 缓存加载: %zu 条 dim=%d\n", entries_.size(), dim_);
}

void Gallery::save_cache() {
    fs::create_directories(cache_dir_);
    json entries = json::array();
    for (auto &e : entries_)
        entries.push_back({{"file", e.file}, {"mtime", e.mtime}, {"off", e.off},
                           {"alive", e.alive}});
    json j = {{"dim", dim_}, {"entries", entries}};
    std::ofstream(cache_dir_ + "/manifest.json") << j.dump(1);
    std::ofstream vf(cache_dir_ + "/vectors.bin", std::ios::binary);
    vf.write((const char *)vecs_.data(), (std::streamsize)(vecs_.size() * sizeof(float)));
}

void Gallery::sync(const EncodeFn &encode, const std::function<void(int, int)> &progress) {
    std::vector<GalleryEntry> new_items;
    {
        std::lock_guard<std::mutex> lk(mu_);
        load_cache();
        // 扫描目录：新文件/变更文件 → 重编码；消失的文件 → 墓碑
        std::vector<std::pair<std::string, int64_t>> found;  // rel, mtime
        if (fs::exists(dir_)) {
            for (auto &p : fs::recursive_directory_iterator(dir_)) {
                if (!p.is_regular_file() || !is_image(p.path().filename().string())) continue;
                std::string rel = fs::relative(p.path(), dir_).string();
                found.emplace_back(rel, file_mtime(p.path()));
            }
        }
        std::sort(found.begin(), found.end());
        std::map<std::string, size_t> by_name;
        for (size_t i = 0; i < entries_.size(); i++)
            if (entries_[i].alive) by_name[entries_[i].file] = i;
        std::map<std::string, int64_t> found_map(found.begin(), found.end());
        for (auto &[rel, mt] : found) {
            auto it = by_name.find(rel);
            if (it == by_name.end()) new_items.push_back({rel, mt, -1, true});
            else if (entries_[it->second].mtime != mt) {
                entries_[it->second].alive = false;   // 变更：旧向量作废，重新入 new_items
                new_items.push_back({rel, mt, -1, true});
            }
        }
        for (auto &[rel, idx] : by_name)
            if (!found_map.count(rel)) entries_[idx].alive = false;
    }

    // 锁外逐张编码，结果攒批后一次追加落盘
    int done = 0, n_total = (int)new_items.size();
    fprintf(stderr, "gallery 增量编码 %d 张（目录 %s）...\n", n_total, dir_.c_str());
    for (auto &item : new_items) {
        std::vector<float> emb;
        try {
            emb = encode((fs::path(dir_) / item.file).string());
        } catch (const std::exception &e) {
            fprintf(stderr, "  编码失败 %s: %s\n", item.file.c_str(), e.what());
            if (progress) progress(++done, n_total);
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (dim_ == 0) dim_ = (int)emb.size();
            if ((int)emb.size() != dim_) {
                fprintf(stderr, "  维度不符 %s: %zu vs %d\n", item.file.c_str(), emb.size(), dim_);
                continue;
            }
            GalleryEntry g;
            g.file = item.file;
            g.mtime = item.mtime;
            g.off = (int64_t)(vecs_.size() / dim_);
            vecs_.insert(vecs_.end(), emb.begin(), emb.end());
            entries_.push_back(std::move(g));
        }
        if (progress) progress(++done, n_total);
        else if (done % 50 == 0 || done == n_total) {
            fprintf(stderr, "  gallery %d/%d\n", done, n_total);
            save_cache();   // 定期落盘，中断可续
        }
    }
    if (n_total > 0) save_cache();
    fprintf(stderr, "gallery 就绪: alive=%d 总数=%d dim=%d\n", size(), total(), dim_);
}

int Gallery::add(const std::string &abs_path, const std::vector<float> &emb) {
    std::lock_guard<std::mutex> lk(mu_);
    std::string rel = fs::relative(fs::path(abs_path), fs::path(dir_)).string();
    for (size_t i = 0; i < entries_.size(); i++)
        if (entries_[i].alive && entries_[i].file == rel) {
            // 同名覆盖：旧向量作废
            entries_[i].alive = false;
        }
    if (dim_ == 0) dim_ = (int)emb.size();
    if ((int)emb.size() != dim_) throw std::runtime_error("enroll 向量维度不符");
    GalleryEntry g;
    g.file = rel;
    g.mtime = file_mtime(abs_path);
    g.off = (int64_t)(vecs_.size() / dim_);
    vecs_.insert(vecs_.end(), emb.begin(), emb.end());
    entries_.push_back(g);
    save_cache();
    return (int)entries_.size() - 1;
}

std::vector<std::pair<std::string, float>> Gallery::search(const std::vector<float> &q,
                                                           int topk) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(entries_.size());
    for (size_t i = 0; i < entries_.size(); i++) {
        const auto &e = entries_[i];
        if (!e.alive || e.off < 0) continue;
        const float *v = &vecs_[(size_t)e.off * dim_];
        double s = 0;
        for (int d = 0; d < dim_; d++) s += (double)q[d] * v[d];
        scored.emplace_back((float)s, i);
    }
    if (topk < (int)scored.size()) {
        std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                          std::greater<>());
        scored.resize(topk);
    } else {
        std::sort(scored.begin(), scored.end(), std::greater<>());
    }
    std::vector<std::pair<std::string, float>> out;
    for (auto &[s, i] : scored) out.emplace_back(entries_[i].file, s);
    return out;
}
