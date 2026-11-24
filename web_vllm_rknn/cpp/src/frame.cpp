#include "frame.h"
#include "config.h"

#include <poll.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <cstdlib>
#include <thread>

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

void draw_polyline(ImageRGB &img, const std::vector<std::pair<int, int>> &pts, bool close) {
    if (pts.size() < 2) return;
    auto put = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= img.w || y >= img.h) return;
        uint8_t *p = &img.rgb[((size_t)y * img.w + x) * 3];
        p[0] = 0; p[1] = 200; p[2] = 255;
    };
    size_t n = pts.size() + (close ? 1 : 0);
    for (size_t i = 0; i + 1 < n; i++) {
        auto [x0, y0] = pts[i];
        auto [x1, y1] = pts[(i + 1) % pts.size()];
        // Bresenham：逐像素连线，越界点由 put 自行丢弃
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        for (;;) {
            // 1px 细线经 JPEG q90 压缩后基本看不见（区域边界就是给人看的证据图），补到 2px
            put(x0, y0); put(x0 + 1, y0); put(x0, y0 + 1);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }
}

std::vector<uint8_t> placeholder_jpeg() {
    ImageRGB img;
    img.w = 640; img.h = 480;
    img.rgb.assign((size_t)640 * 480 * 3, 0);
    return encode_jpeg(img, 95);
}

// ==================== 常驻拉流：最新帧槽 + PersistentPuller ====================

// 帧龄/看门狗用单调钟：系统时间被 NTP 回拨不该触发「流挂起」误杀或「旧帧」误丢
static int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

void LatestFrameSlot::publish(std::vector<uint8_t> jpeg, int64_t stamp_ms) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) return;
        seq_++;
        jpeg_ = std::move(jpeg);
        stamp_ms_ = stamp_ms;
    }
    cv_.notify_all();
}

bool LatestFrameSlot::waitNew(uint64_t last_seq, const std::atomic<bool> & running) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !running.load() || closed_ || seq_ != last_seq; });
    return running.load() && !closed_ && seq_ != last_seq;
}

uint64_t LatestFrameSlot::latest(std::vector<uint8_t> & jpeg, int64_t & age_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    jpeg = jpeg_;                     // 只拷字节，解码留在锁外
    age_ms = stamp_ms_ ? steady_ms() - stamp_ms_ : 0;
    return seq_;
}

int64_t LatestFrameSlot::ageMs() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stamp_ms_ ? steady_ms() - stamp_ms_ : -1;
}

void LatestFrameSlot::close() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
    }
    cv_.notify_all();
}

void PersistentPuller::reap() {
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
    if (pid_ > 0) {
        // 子进程 setpgid(0,0) 自成进程组：杀 -pgid 连它可能带的子孙一起收（PLAN 危险清单 #3）
        kill(-pid_, SIGKILL);
        kill(pid_, SIGKILL);          // setpgid 万一没生效的兜底
        int status;
        waitpid(pid_, &status, 0);    // SIGKILL 后即刻回收，不留僵尸
        pid_ = -1;
    }
}

void PersistentPuller::loop(const std::string & rtsp_url, const std::string & tag,
                            std::atomic<bool> & running) {
    const int64_t stale_ms = getenv_int("VLM_PULL_STALE_MS", 10000);
    int backoff = 1;
    std::vector<uint8_t> buf;
    while (running) {
        int fds[2];
        if (pipe(fds) != 0) {
            for (int i = 0; i < 10 && running; i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        pid_t pid = fork();
        if (pid < 0) {
            close(fds[0]); close(fds[1]);
            for (int i = 0; i < 10 && running; i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (pid == 0) {
            setpgid(0, 0);            // 自成进程组，重连时整组收割
            dup2(fds[1], STDOUT_FILENO);
            close(fds[0]); close(fds[1]);
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
            // 与一次性取帧同款参数，只去掉 -vframes 1：常驻解码、JPEG 连续出到 stdout
            execlp("ffmpeg", "ffmpeg", "-loglevel", "error",
                   "-rtsp_transport", "tcp", "-stimeout", "5000000",
                   "-i", rtsp_url.c_str(),
                   "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
                   (char *)nullptr);
            _exit(127);
        }
        setpgid(pid, 0);              // 父进程并设一次，消除子进程 exec 前进组的窗口
        close(fds[1]);
        pid_ = pid;
        fd_ = fds[0];
        buf.clear();

        bool got_frame = false;
        int64_t last_out_ms = steady_ms();
        while (running) {
            struct pollfd p{fd_, POLLIN, 0};
            int pr = poll(&p, 1, 200);
            if (pr < 0) { if (errno == EINTR) continue; break; }
            if (pr == 0) {
                // 看门狗：TCP 半开时 ffmpeg 挂着不退出也不吐字节 → 主动杀了重连，
                // 否则帧槽停在最后一帧（PLAN 危险清单 #1②）
                if (steady_ms() - last_out_ms > stale_ms) {
                    printf("[%s] 拉流 %lldms 无数据，杀掉 ffmpeg 重连\n",
                           tag.c_str(), (long long)stale_ms);
                    break;
                }
                continue;
            }
            uint8_t chunk[65536];
            ssize_t n = read(fd_, chunk, sizeof(chunk));
            if (n == 0) {
                printf("[%s] 拉流中断（ffmpeg 退出），等待重连...\n", tag.c_str());
                break;
            }
            if (n < 0) { if (errno == EINTR) continue; break; }
            last_out_ms = steady_ms();
            buf.insert(buf.end(), chunk, chunk + n);

            // 按 JPEG 整包切帧 FFD8..FFD9：scan 数据里 0xFF 必被填充为 0xFF00，
            // FFD9 只可能是真 EOI，不会在流中间误切
            size_t pos = 0;
            while (true) {
                size_t soi = buf.size();
                for (size_t i = pos + 1; i < buf.size(); i++)
                    if (buf[i] == 0xD8 && buf[i - 1] == 0xFF) { soi = i - 1; break; }
                if (soi == buf.size()) break;                       // 没有帧头，等更多数据
                size_t eoi = buf.size();
                for (size_t i = soi + 3; i < buf.size(); i++)
                    if (buf[i] == 0xD9 && buf[i - 1] == 0xFF) { eoi = i; break; }
                if (eoi == buf.size()) { pos = soi; break; }        // 帧没收完，留在缓冲
                std::vector<uint8_t> jpeg(buf.begin() + soi, buf.begin() + eoi + 1);
                slot_->publish(std::move(jpeg), steady_ms());
                got_frame = true;
                pos = eoi + 1;
            }
            if (pos > 0) buf.erase(buf.begin(), buf.begin() + pos);
            if (buf.size() > (16u << 20)) buf.clear();              // 切帧始终不成帧的兜底：丢缓存防涨爆
        }

        reap();
        if (!running) break;
        for (int i = 0; i < backoff * 10 && running; i++)            // 退避可被 stop 打断
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        backoff = got_frame ? 1 : std::min(backoff * 2, 30);
    }
}
