// attr_head.bin 加载 + CPU 推理（线性头 ~11 万 MACs，A55 上 <1ms）。
#include "attr_head.h"
#include "attributes.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace {
const char MAGIC[6] = {'A', 'H', 'E', 'A', 'D', '1'};
const uint32_t VERSION = 1;

bool read_u32(std::ifstream & f, uint32_t & v) {
    return !!f.read(reinterpret_cast<char *>(&v), 4);
}
}  // namespace

bool AttrHead::load(const std::string & path, int expect_in_dim) {
    ready_ = false;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        printf("[attr_head] 未加载（文件不存在）: %s\n", path.c_str());
        return false;
    }
    char magic[6] = {};
    uint32_t version = 0, in_dim = 0, n_attrs = 0;
    if (!f.read(magic, 6) || memcmp(magic, MAGIC, 6) != 0 ||
        !read_u32(f, version) || version != VERSION ||
        !read_u32(f, in_dim) || !read_u32(f, n_attrs)) {
        fprintf(stderr, "[attr_head] %s: magic/version/头部读取失败\n", path.c_str());
        return false;
    }
    if (in_dim == 0 || n_attrs == 0 ||
        (expect_in_dim > 0 && (int)in_dim != expect_in_dim)) {
        fprintf(stderr, "[attr_head] %s: in_dim=%u 与视觉 embed_size=%d 不匹配，禁用\n",
                path.c_str(), in_dim, expect_in_dim);
        return false;
    }
    thr_.assign(n_attrs, 0.5f);
    if (!f.read(reinterpret_cast<char *>(thr_.data()), 4 * n_attrs)) {
        fprintf(stderr, "[attr_head] %s: 阈值段截断\n", path.c_str());
        return false;
    }
    // 属性名逐字校验：训练侧属性表必须与 attributes.h 顺序一致
    const auto & table = pedestrianAttributes();
    if ((size_t)n_attrs != table.size()) {
        fprintf(stderr, "[attr_head] %s: num_attrs=%u != 板端属性表 %zu，禁用\n",
                path.c_str(), n_attrs, table.size());
        return false;
    }
    for (uint32_t j = 0; j < n_attrs; j++) {
        uint32_t len = 0;
        if (!read_u32(f, len) || len == 0 || len > 128) {
            fprintf(stderr, "[attr_head] %s: 第 %u 个属性名长度异常\n", path.c_str(), j);
            return false;
        }
        std::string name(len, '\0');
        if (!f.read(&name[0], len) || name != table[j]) {
            fprintf(stderr, "[attr_head] %s: 第 %u 个属性名不匹配（bin=%s 板端=%s），禁用\n",
                    path.c_str(), j, name.c_str(), table[j].c_str());
            return false;
        }
    }
    w_.resize((size_t)in_dim * n_attrs);
    b_.resize(n_attrs);
    if (!f.read(reinterpret_cast<char *>(w_.data()), 4 * (size_t)in_dim * n_attrs) ||
        !f.read(reinterpret_cast<char *>(b_.data()), 4 * n_attrs)) {
        fprintf(stderr, "[attr_head] %s: 权重段截断\n", path.c_str());
        return false;
    }
    char extra = 0;
    if (f.read(&extra, 1)) {
        fprintf(stderr, "[attr_head] %s: 尾部多余字节，格式版本不符，禁用\n", path.c_str());
        return false;
    }
    in_dim_ = (int)in_dim;
    n_attrs_ = (int)n_attrs;
    ready_ = true;
    printf("[attr_head] 已加载 %s: %d->%d（%s）\n",
           path.c_str(), in_dim_, n_attrs_, getenv("VLM_ATTR_HEAD_DUMP") ? "DUMP=on" : "DUMP=off");
    return true;
}

void AttrHead::run(const float * embed, int n_tokens, int n_output, int embed_size,
                   std::vector<std::string> * attrs_out,
                   std::vector<float> * probs_out) const {
    if (attrs_out) attrs_out->clear();
    if (!ready_ || n_tokens <= 0 || n_output <= 0 || embed_size != in_dim_) return;

    const auto & table = pedestrianAttributes();
    std::vector<float> pooled(in_dim_, 0.0f);
    // 段 0 = main（导出 ONNX graph.output[0] 即 merger 主输出），token 步长 n_output*embed_size
    for (int t = 0; t < n_tokens; t++) {
        const float * seg = embed + ((size_t)t * n_output) * embed_size;
        for (int d = 0; d < in_dim_; d++) pooled[d] += seg[d];
    }
    const float inv = 1.0f / (float)n_tokens;
    for (auto & v : pooled) v *= inv;

    if (attrs_out || probs_out) {
        for (int j = 0; j < n_attrs_; j++) {
            const float * w = w_.data() + (size_t)j * in_dim_;
            float acc = b_[j];
            for (int d = 0; d < in_dim_; d++) acc += w[d] * pooled[d];
            float prob = 1.0f / (1.0f + std::exp(-acc));
            if (probs_out) probs_out->push_back(prob);
            if (attrs_out && prob >= thr_[j] && j < (int)table.size())
                attrs_out->push_back(table[j]);
        }
    }

    // 特征对齐调试：每帧追加一行 "pool:...,probs:..." 到指定文件
    const char * dump = getenv("VLM_ATTR_HEAD_DUMP");
    if (dump && dump[0]) {
        std::ofstream out(dump, std::ios::app);
        if (out) {
            out << "pool";
            for (int d = 0; d < in_dim_; d++) out << ',' << pooled[d];
            out << '\n';
        }
    }
}
