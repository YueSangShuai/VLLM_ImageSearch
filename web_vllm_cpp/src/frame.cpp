#include "frame.h"

#include <poll.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <csignal>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

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
    // stb_image_resize v2 的 medium API：STBIR_TYPE_UINT8 即线性色彩空间、无 alpha 加权，
    // 等价于 v1 stbir_resize_uint8_generic(..., STBIR_NO_ALPHA, ..., STBIR_COLORSPACE_LINEAR, ...)
    stbir_resize(src.rgb.data(), src.w, src.h, 0,
                 dst.rgb.data(), nw, nh, 0,
                 STBIR_RGB, STBIR_TYPE_UINT8,
                 STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE);
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
