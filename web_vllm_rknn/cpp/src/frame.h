#pragma once
// 图像 I/O：ffmpeg 子进程取 RTSP 帧；stb 负责 JPEG/PNG 编解码与缩放（不引入 OpenCV）
#include <string>
#include <vector>
#include <cstdint>

struct ImageRGB {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // w*h*3
};

// 从 RTSP 拉 1 帧，返回 JPEG 字节；失败返回空（命令与 processor.py 对齐）
std::vector<uint8_t> grab_frame_jpeg(const std::string &rtsp_url, int timeout_sec = 10);

bool decode_jpeg(const std::vector<uint8_t> &jpeg, ImageRGB &out);
std::vector<uint8_t> encode_jpeg(const ImageRGB &img, int quality = 95);

// 双线性缩放到 (nw,nh)（对应 cv2.resize）
ImageRGB resize_bilinear(const ImageRGB &src, int nw, int nh);

// 等比放置到正方形画布、空白填 bg（对齐训练端 PIL expand2square：偏移 (side-d)/2 向下取整）
ImageRGB expand2square(const ImageRGB &src, uint8_t bg = 127);

// 在原图 RGB 上画绿色矩形（对应 cv2.rectangle thickness=2 color=(0,255,0) BGR）
void draw_rect(ImageRGB &img, int x1, int y1, int x2, int y2);

// 占位图（640x480 纯黑 JPEG，对应 serve_placeholder 无文字版）
std::vector<uint8_t> placeholder_jpeg();

// 构建 RTSP URL（对齐 DeviceManager.get_rtsp_url，用户名/密码 URL 编码）
std::string build_rtsp_url(const std::string &username, const std::string &password,
                           const std::string &ip, const std::string &port,
                           const std::string &path);
std::string url_encode(const std::string &s);
