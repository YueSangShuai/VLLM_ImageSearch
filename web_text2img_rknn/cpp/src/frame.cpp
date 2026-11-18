#include "frame.h"

#include <poll.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <csignal>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

std::string url_encode(const std::string &s) {
    // 对齐 urllib.parse.quote(safe='')：仅保留 alnum 与 _.-~
    std::string out;
    char buf[4];
    for (unsigned char c : s) {
        if (isalnum(c) || c == '_' || c == '.' || c == '-' || c == '~') {
            out += (char)c;
        } else {
            snprintf(buf, sizeof(buf), "%%%02X", c);
            out += buf;
        }
    }
    return out;
}

std::string build_rtsp_url(const std::string &username, const std::string &password,
                           const std::string &ip, const std::string &port,
                           const std::string &path) {
    return "rtsp://" + url_encode(username) + ":" + url_encode(password) +
           "@" + ip + ":" + port + path;
}

std::vector<uint8_t> grab_frame_jpeg(const std::string &rtsp_url, int timeout_sec) {
    // ffmpeg -rtsp_transport tcp -stimeout 5000000 -i URL -vframes 1 -q:v 5
    //        -f image2pipe -vcodec mjpeg -
    int fds[2];
    if (pipe(fds) != 0) return {};
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return {}; }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]); close(fds[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execlp("ffmpeg", "ffmpeg", "-rtsp_transport", "tcp",
               "-stimeout", "5000000", "-i", rtsp_url.c_str(),
               "-vframes", "1", "-q:v", "5",
               "-f", "image2pipe", "-vcodec", "mjpeg", "-",
               (char *)nullptr);
        _exit(127);
    }
    close(fds[1]);

    std::vector<uint8_t> data;
    uint8_t buf[65536];
    int deadline_ms = timeout_sec * 1000;
    bool timed_out = false;
    while (true) {
        struct pollfd p{fds[0], POLLIN, 0};
        int pr = poll(&p, 1, 500);
        if (pr < 0 && errno == EINTR) continue;
        if (pr == 0) {
            if ((deadline_ms -= 500) <= 0) { timed_out = true; break; }
            continue;
        }
        ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n == 0) break;
        if (n < 0) { if (errno == EINTR) continue; break; }
        data.insert(data.end(), buf, buf + n);
    }
    close(fds[0]);
    if (timed_out) kill(pid, SIGKILL);
    int status;
    waitpid(pid, &status, 0);
    if (timed_out || data.empty()) return {};
    return data;
}

bool decode_jpeg(const std::vector<uint8_t> &jpeg, ImageRGB &out) {
    if (jpeg.empty()) return false;
    int w = 0, h = 0, ch = 0;
    stbi_uc *px = stbi_load_from_memory(jpeg.data(), (int)jpeg.size(), &w, &h, &ch, 3);
    if (!px) return false;
    out.w = w; out.h = h;
    out.rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    return true;
}

static void jpeg_writer(void *ctx, void *data, int size) {
    auto *vec = (std::vector<uint8_t> *)ctx;
    const uint8_t *p = (const uint8_t *)data;
    vec->insert(vec->end(), p, p + size);
}

std::vector<uint8_t> encode_jpeg(const ImageRGB &img, int quality) {
    std::vector<uint8_t> out;
    stbi_write_jpg_to_func(jpeg_writer, &out, img.w, img.h, 3, img.rgb.data(), quality);
    return out;
}

ImageRGB resize_bilinear(const ImageRGB &src, int nw, int nh) {
    ImageRGB dst;
    dst.w = nw; dst.h = nh;
    dst.rgb.resize((size_t)nw * nh * 3);
    if (src.w == nw && src.h == nh) { dst.rgb = src.rgb; return dst; }
    // ⚠ OpenCV 4.12 8U INTER_LINEAR 的【逐字节】复刻（实测 13 种尺寸+真实 JPEG 全零差异）：
    //   · 系数按 resize.cpp hal 路径：float32 fx=(dx+0.5)*scale-0.5 → 截断取 sx →
    //     独立权重 round((1-fx)*2048) / round(fx*2048)（nearbyint=银行家舍入，勿用 lround）
    //   · 横向精确 int32：S = p[sx]*a0 + p[sx+chan]*a1
    //   · 纵向不是教科书 (S0*b0+S1*b1+2^21)>>22！是 VResizeLinearVec_32s8u 截断律：
    //       ((S0>>4)*b0>>16) + ((S1>>4)*b1>>16) + 2 >> 2
    //     （平均偏 -0.13/像素，是该版本固有偏置，是排序对齐的关键，勿“修正”）
    //   · y 越界权重不置零（invoker clamp 行号、权重原样），仅 x 边界把 fx 清零
    //   · 尾元素（宽*通道 % 32）走标量 (S0*b0+S1*b1+2^21)>>22；服务恒为 224*3=672 无尾
    constexpr int S = 1 << 11;  // INTER_RESIZE_COEF_SCALE
    const double scale_x = (double)src.w / nw, scale_y = (double)src.h / nh;
    const int chan = 3;
    std::vector<int> ix0(nw), a0(nw), a1(nw);
    for (int x = 0; x < nw; x++) {
        float fx = (float)((x + 0.5) * scale_x - 0.5);
        int sx = (int)std::floor(fx);
        fx -= (float)sx;
        if (sx < 0)     { fx = 0.f; sx = 0; }
        if (sx >= src.w - 1) { fx = 0.f; sx = src.w - 1; }
        ix0[x] = sx;
        a1[x] = (int)std::nearbyintf(fx * 2048.f);
        a0[x] = (int)std::nearbyintf((1.f - fx) * 2048.f);
    }
    std::vector<int32_t> hb0((size_t)nw * chan), hb1((size_t)nw * chan);
    for (int y = 0; y < nh; y++) {
        float fy = (float)((y + 0.5) * scale_y - 0.5);
        int sy = (int)std::floor(fy);
        fy -= (float)sy;
        int r0 = std::min(std::max(sy, 0), src.h - 1);
        int r1 = std::min(std::max(sy + 1, 0), src.h - 1);
        int b1 = (int)std::nearbyintf(fy * 2048.f);
        int b0 = (int)std::nearbyintf((1.f - fy) * 2048.f);
        const uint8_t *row0 = src.rgb.data() + (size_t)r0 * src.w * chan;
        const uint8_t *row1 = src.rgb.data() + (size_t)r1 * src.w * chan;
        for (int x = 0; x < nw; x++)
            for (int c = 0; c < chan; c++) {
                int i0 = ix0[x] * chan + c, i1 = i0 + chan;
                hb0[x * chan + c] = (int32_t)row0[i0] * a0[x] + (int32_t)row0[i1] * a1[x];
                hb1[x * chan + c] = (int32_t)row1[i0] * a0[x] + (int32_t)row1[i1] * a1[x];
            }
        uint8_t *out = dst.rgb.data() + (size_t)y * nw * chan;
        int w = nw * chan, i = 0;
        for (; i <= w - 16; i += 16) {  // 向量主体（32/16  lane 律相同）
            for (int k = 0; k < 16; k++) {
                int t0 = ((hb0[i + k] >> 4) * b0) >> 16;
                int t1 = ((hb1[i + k] >> 4) * b1) >> 16;
                out[i + k] = (uint8_t)((t0 + t1 + 2) >> 2);
            }
        }
        for (; i < w; i++) {            // 标量尾：FixedPtCast
            int64_t v = (int64_t)hb0[i] * b0 + (int64_t)hb1[i] * b1;
            out[i] = (uint8_t)((v + ((int64_t)1 << 21)) >> 22);
        }
    }
    return dst;
}

ImageRGB expand2square(const ImageRGB &src, uint8_t bg) {
    // 对齐训练端 PIL expand2square：画布边长 max(w,h)，粘贴偏移 (side-d)/2（整数除法即向下取整）
    int side = std::max(src.w, src.h);
    ImageRGB dst;
    dst.w = dst.h = side;
    dst.rgb.assign((size_t)side * side * 3, bg);
    if (side == src.w && side == src.h) { dst.rgb = src.rgb; return dst; }
    int ox = (side - src.w) / 2, oy = (side - src.h) / 2;
    const size_t row = (size_t)src.w * 3;
    for (int y = 0; y < src.h; y++)
        memcpy(&dst.rgb[((size_t)(y + oy) * side + ox) * 3],
               &src.rgb[(size_t)y * row], row);
    return dst;
}

void draw_rect(ImageRGB &img, int x1, int y1, int x2, int y2) {
    auto put = [&](int x, int y) {
        // BGR (0,255,0) -> RGB (0,255,0)
        if (x < 0 || y < 0 || x >= img.w || y >= img.h) return;
        uint8_t *p = &img.rgb[((size_t)y * img.w + x) * 3];
        p[0] = 0; p[1] = 255; p[2] = 0;
    };
    int xa = std::min(x1, x2), xb = std::max(x1, x2);
    int ya = std::min(y1, y2), yb = std::max(y1, y2);
    for (int t = 0; t < 2; t++) {
        for (int x = xa; x <= xb; x++) { put(x, ya + t); put(x, yb - t); }
        for (int y = ya; y <= yb; y++) { put(xa + t, y); put(xb - t, y); }
    }
}

std::vector<uint8_t> placeholder_jpeg() {
    ImageRGB img;
    img.w = 640; img.h = 480;
    img.rgb.assign((size_t)640 * 480 * 3, 0);
    return encode_jpeg(img, 95);
}
