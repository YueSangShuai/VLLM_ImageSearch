#include "detector.h"
#include "frame.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
const Ort::MemoryInfo g_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
constexpr int kPersonClass = 0;

float iou(const float *a, const float *b) {
    float ix1 = std::max(a[0], b[0]), iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]), iy2 = std::min(a[3], b[3]);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return ua > 0 ? inter / ua : 0;
}

int round_half_up(double v) { return (int)std::lround(v); }  // 近 round()
}  // namespace

Detector::Detector(const std::string &onnx_path, float conf_th, float iou_th)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"), conf_th_(conf_th), iou_th_(iou_th) {
    opts_.SetIntraOpNumThreads(2);
    opts_.SetInterOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), opts_);
}

Detector::Letterboxed Detector::letterbox(const ImageRGB &frame) const {
    // 对齐 Python PersonDetector.detect 的预缩放 + ultralytics LetterBox(auto=True)
    Letterboxed out;
    int w = frame.w, h = frame.h;
    double scale1 = std::min((double)max_side_ / std::max(w, h), 1.0);
    ImageRGB f1 = frame;
    if (scale1 < 1.0) {
        f1 = resize_bilinear(frame, (int)(w * scale1), (int)(h * scale1));
    }

    double r = std::min((double)max_side_ / f1.w, (double)max_side_ / f1.h);
    int new_w_un = (int)std::lround(f1.w * r);
    int new_h_un = (int)std::lround(f1.h * r);
    int new_w = ((new_w_un + stride_ - 1) / stride_) * stride_;
    int new_h = ((new_h_un + stride_ - 1) / stride_) * stride_;
    double dw = (new_w - new_w_un) / 2.0, db = (new_h - new_h_un) / 2.0;
    int left = round_half_up(dw - 0.1), right = round_half_up(dw + 0.1);
    int top = round_half_up(db - 0.1), bottom = round_half_up(db + 0.1);

    ImageRGB resized = resize_bilinear(f1, new_w_un, new_h_un);
    out.img.w = new_w; out.img.h = new_h;
    out.img.rgb.assign((size_t)new_w * new_h * 3, 114);
    for (int y = 0; y < new_h_un; y++) {
        memcpy(&out.img.rgb[((size_t)(y + top) * new_w + left) * 3],
               &resized.rgb[(size_t)y * new_w_un * 3], (size_t)new_w_un * 3);
    }
    out.ratio = (float)r;
    out.pad_l = left; out.pad_t = top;
    (void)bottom; (void)right; (void)scale1;
    return out;
}

std::vector<Detection> Detector::detect(const ImageRGB &frame) {
    std::lock_guard<std::mutex> lk(mu_);
    auto lb = letterbox(frame);
    int H = lb.img.h, W = lb.img.w;

    // uint8 RGB -> float32 NCHW /255
    std::vector<float> input((size_t)3 * H * W);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const uint8_t *p = &lb.img.rgb[((size_t)y * W + x) * 3];
            input[((size_t)0 * H + y) * W + x] = p[0] / 255.0f;
            input[((size_t)1 * H + y) * W + x] = p[1] / 255.0f;
            input[((size_t)2 * H + y) * W + x] = p[2] / 255.0f;
        }

    std::array<int64_t, 4> shape{1, 3, H, W};
    auto tensor = Ort::Value::CreateTensor<float>(g_mem, input.data(), input.size(),
                                                  shape.data(), 4);
    const char *in_names[] = {"images"};
    const char *out_names[] = {"output0"};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    auto oshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (oshape.size() != 3) throw std::runtime_error("YOLO 输出形状异常");
    int N = (int)oshape[2]; (void)oshape;
    const float *o = outputs[0].GetTensorData<float>();

    // 预缩放比例（PersonDetector 的第一次 resize）
    double scale1 = std::min((double)max_side_ / std::max(frame.w, frame.h), 1.0);

    struct Cand { float box[4]; float conf; };
    std::vector<Cand> cands;

    // 展平索引：stride 0/1/2 的网格按行优先依次排列
    int base = 0;
    for (int s_idx = 0; s_idx < 3; s_idx++) {
        int stride = 8 << s_idx;
        int gh = H / stride, gw = W / stride;
        for (int gy = 0; gy < gh; gy++)
            for (int gx = 0; gx < gw; gx++) {
                int idx = base + gx + gy * gw;
                float conf = o[(size_t)(4 + kPersonClass) * N + idx];
                if (conf < conf_th_) continue;
                // 该导出已内置 DFL，前 4 通道即 letterbox 像素坐标系的绝对 xywh
                // （黄金样例实测：直接用像素解出的框与 golden [279,1,318,113] 对齐）
                float bx = o[(size_t)0 * N + idx], by = o[(size_t)1 * N + idx];
                float bw = o[(size_t)2 * N + idx], bh = o[(size_t)3 * N + idx];
                float x1 = bx - bw / 2, y1 = by - bh / 2;
                float x2 = bx + bw / 2, y2 = by + bh / 2;
                (void)gx; (void)gy; (void)stride;
                // 去 letterbox + 去预缩放（对应 Python 端 int(box/scale) 截断）
                cands.push_back({{(x1 - lb.pad_l) / lb.ratio / (float)scale1,
                                  (y1 - lb.pad_t) / lb.ratio / (float)scale1,
                                  (x2 - lb.pad_l) / lb.ratio / (float)scale1,
                                  (y2 - lb.pad_t) / lb.ratio / (float)scale1}, conf});
            }
        base += gw * gh;
    }

    // NMS（贪心，IoU > 阈值丢弃），对齐 ultralytics max_det=300
    std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
        return a.conf > b.conf;
    });
    std::vector<int> keep;
    for (size_t i = 0; i < cands.size() && keep.size() < 300; i++) {
        bool ok = true;
        for (int j : keep) {
            if (iou(cands[i].box, cands[j].box) > iou_th_) { ok = false; break; }
        }
        if (ok) keep.push_back((int)i);
    }

    std::vector<Detection> dets;
    for (int i : keep) {
        Detection d;
        d.bbox[0] = (int)cands[i].box[0];
        d.bbox[1] = (int)cands[i].box[1];
        d.bbox[2] = (int)cands[i].box[2];
        d.bbox[3] = (int)cands[i].box[3];
        d.confidence = cands[i].conf;
        dets.push_back(d);
    }
    return dets;
}
