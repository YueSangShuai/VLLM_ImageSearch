#include "detector.h"
#include "frame.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {
#ifndef WEBVLLM_RKNN
const Ort::MemoryInfo g_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
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

#ifdef WEBVLLM_RKNN
// IEEE-754 binary16 -> float32
float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {  // subnormal
            int e = -1;
            do { mant <<= 1; e++; } while (!(mant & 0x400));
            mant &= 0x3ff;
            out = sign | ((uint32_t)(127 - 15 - e + 1) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000 | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}
#endif
}  // namespace

Detector::Detector(const std::string &onnx_path, float conf_th, float iou_th)
    : conf_th_(conf_th), iou_th_(iou_th) {
#ifdef WEBVLLM_RKNN
    int ret = rknn_init(&rknn_ctx_, (void *)onnx_path.c_str(), 0, 0, NULL);
    if (ret < 0) throw std::runtime_error("rknn_init YOLO 失败 ret=" + std::to_string(ret));
    const char *cores = getenv("RKNN_YOLO_CORE_NUM");
    int core = (cores && atoi(cores) > 0) ? atoi(cores) : 1;  // 默认单核，给 LLM 留核
    ret = rknn_set_core_mask(rknn_ctx_, core == 3 ? RKNN_NPU_CORE_0_1_2
                             : core == 2 ? RKNN_NPU_CORE_0_1
                                         : RKNN_NPU_CORE_0);
    if (ret < 0) throw std::runtime_error("rknn_set_core_mask YOLO 失败");
    rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    in_attr_.index = 0;
    rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_));
    out_attr_.index = 0;
    rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr_, sizeof(out_attr_));
    fprintf(stderr, "YOLO rknn in: dims=[%d,%d,%d,%d] type=%d size=%d n_out=%d\n",
            in_attr_.dims[0], in_attr_.dims[1], in_attr_.dims[2], in_attr_.dims[3],
            (int)in_attr_.type, in_attr_.size, (int)io_num_.n_output);
    fprintf(stderr, "YOLO rknn out: dims=[%d,%d,%d,%d] type=%d size=%d n_elems=%d\n",
            out_attr_.dims[0], out_attr_.dims[1], out_attr_.dims[2], out_attr_.dims[3],
            (int)out_attr_.type, out_attr_.size, out_attr_.n_elems);
#else
    opts_.SetIntraOpNumThreads(2);
    opts_.SetInterOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), opts_);
#endif
}

Detector::~Detector() {
#ifdef WEBVLLM_RKNN
    if (rknn_ctx_) rknn_destroy(rknn_ctx_);
#endif
}

Detector::Letterboxed Detector::letterbox(const ImageRGB &frame) const {
    // 对齐 Python PersonDetector.detect 的预缩放 + ultralytics LetterBox(auto=True)
    // RKNN 后端：固定 640x640 方形（auto 语义等价于 padding 到正方形）
    Letterboxed out;
    int w = frame.w, h = frame.h;
    double scale1 = std::min((double)max_side_ / std::max(w, h), 1.0);
    ImageRGB f1 = frame;
    if (scale1 < 1.0) {
        f1 = resize_bilinear(frame, (int)(w * scale1), (int)(h * scale1));
    }

#ifdef WEBVLLM_RKNN
    int new_w = max_side_, new_h = max_side_;
    double r = std::min((double)max_side_ / f1.w, (double)max_side_ / f1.h);
    int new_w_un = (int)std::lround(f1.w * r);
    int new_h_un = (int)std::lround(f1.h * r);
    double dw = (new_w - new_w_un) / 2.0, db = (new_h - new_h_un) / 2.0;
    int left = round_half_up(dw), top = round_half_up(db);
#else
    double r = std::min((double)max_side_ / f1.w, (double)max_side_ / f1.h);
    int new_w_un = (int)std::lround(f1.w * r);
    int new_h_un = (int)std::lround(f1.h * r);
    int new_w = ((new_w_un + stride_ - 1) / stride_) * stride_;
    int new_h = ((new_h_un + stride_ - 1) / stride_) * stride_;
    double dw = (new_w - new_w_un) / 2.0, db = (new_h - new_h_un) / 2.0;
    int left = round_half_up(dw - 0.1), top = round_half_up(db - 0.1);
#endif

    ImageRGB resized = resize_bilinear(f1, new_w_un, new_h_un);
    out.img.w = new_w; out.img.h = new_h;
    out.img.rgb.assign((size_t)new_w * new_h * 3, 114);
    for (int y = 0; y < new_h_un; y++) {
        memcpy(&out.img.rgb[((size_t)(y + top) * new_w + left) * 3],
               &resized.rgb[(size_t)y * new_w_un * 3], (size_t)new_w_un * 3);
    }
    out.ratio = (float)r;
    out.pad_l = left; out.pad_t = top;
    return out;
}

std::vector<Detection> Detector::detect(const ImageRGB &frame) {
    std::lock_guard<std::mutex> lk(mu_);
    auto lb = letterbox(frame);
    int H = lb.img.h, W = lb.img.w;

    std::vector<float> input;   // ORT 路径：float NCHW /255
    std::vector<float> outbuf;  // 展平后的 [C*N] 输出
#ifdef WEBVLLM_RKNN
    // uint8 NHWC 原图直喂：/255 已折进模型归一化（convert_yolo_rknn.py mean/std）
    // float NCHW 在 NHWC 模型上触发 "unsupported src layout for normalize"
    {
        rknn_input in = {};
        in.index = 0;
        in.type = RKNN_TENSOR_UINT8;
        in.fmt = RKNN_TENSOR_NHWC;
        in.size = (uint32_t)lb.img.rgb.size();
        in.pass_through = 0;
        in.buf = lb.img.rgb.data();
        int ret = rknn_inputs_set(rknn_ctx_, 1, &in);
        if (ret < 0) throw std::runtime_error("YOLO rknn_inputs_set 失败");
        ret = rknn_run(rknn_ctx_, nullptr);
        if (ret < 0) throw std::runtime_error("YOLO rknn_run 失败");
        // 模型带 6 个 graph output（5 个中间张量残留；trim 反而触发 toolkit 融合崩溃，
        // 见 convert_yolo_rknn.py），index 0 即 output0 [1,84,8400]。
        // 必须 want_float=1（官方 demo 姿势）：want_float=0 的 raw 路径在本库版本上
        // 对含中间张量输出的模型报 "output index(127) out of range(6)"
        // 本库版本要求请求数 == n_output（单取 index 0 会报 "output index(127) out of range(6)"，
        // 127 是库内部对中间张量输出的原生 id），全部请求后只用第 0 个
        std::vector<rknn_output> os(io_num_.n_output);
        for (uint32_t j = 0; j < io_num_.n_output; j++) {
            os[j] = rknn_output{};
            os[j].index = j;
            os[j].want_float = 1;
        }
        ret = rknn_outputs_get(rknn_ctx_, io_num_.n_output, os.data(), NULL);
        if (ret < 0) throw std::runtime_error("YOLO rknn_outputs_get 失败 ret=" + std::to_string(ret));
        size_t n = out_attr_.n_elems;
        outbuf.resize(n);
        memcpy(outbuf.data(), os[0].buf, n * sizeof(float));  // want_float=1 → 恒为 float32
        rknn_outputs_release(rknn_ctx_, io_num_.n_output, os.data());
    }
    const float *o = outbuf.data();
    // dims: [1, C, N]
    int C = out_attr_.dims[1];
    int N = out_attr_.dims[2];
    if (outbuf.size() != (size_t)C * N)
        throw std::runtime_error("YOLO RKNN 输出大小异常");
#else
    // uint8 RGB -> float32 NCHW /255
    input.resize((size_t)3 * H * W);
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
    int N = (int)oshape[2];
    const float *o = outputs[0].GetTensorData<float>();
#endif

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
