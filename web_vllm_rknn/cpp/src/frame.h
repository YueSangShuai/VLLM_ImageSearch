#pragma once
// 图像 I/O：ffmpeg 子进程取 RTSP 帧；stb 负责 JPEG/PNG 编解码与缩放（不引入 OpenCV）
// 常驻拉流（VLM_PULL_MODE=persistent）：PersistentPuller 读流线程 + LatestFrameSlot 最新帧槽
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <sys/types.h>

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

// 青色折线/闭合多边形（RGB 0,200,255），与绿色人框区分。close=true 时末点连回首点。
void draw_polyline(ImageRGB &img, const std::vector<std::pair<int, int>> &pts, bool close);

// 占位图（640x480 纯黑 JPEG，对应 serve_placeholder 无文字版）
std::vector<uint8_t> placeholder_jpeg();

// 构建 RTSP URL（对齐 DeviceManager.get_rtsp_url，用户名/密码 URL 编码）
std::string build_rtsp_url(const std::string &username, const std::string &password,
                           const std::string &ip, const std::string &port,
                           const std::string &path);
std::string url_encode(const std::string &s);

// ==================== 常驻拉流（VLM_PULL_MODE=persistent，PLAN「双线程」方案） ====================

// 单格「最新帧槽」，不是队列：读流线程覆盖旧帧、处理慢时中间帧直接丢——latest-wins，
// 杜绝积压过期画面（处理到人已走出画面的告警图）。槽内只放 JPEG 字节：
// 处理线程拷出后在锁外解码，否则读流被解码拖住 → 反压 ffmpeg 管道 → 整设备卡死。
class LatestFrameSlot {
public:
    void publish(std::vector<uint8_t> jpeg, int64_t stamp_ms);
    // 阻塞至 seq != last_seq 的新帧出现，或 close()/running=false。true=来了新帧
    bool waitNew(uint64_t last_seq, const std::atomic<bool> &running);
    // 锁内拷出最新帧，返回其序号（调用方更新 last_seq）与帧龄(ms)
    uint64_t latest(std::vector<uint8_t> &jpeg, int64_t &age_ms);
    int64_t ageMs() const;   // 距上一帧写入(ms，单调钟)；-1=从未有帧（online 判定回退用）
    void close();            // 唤醒全部 waitNew 等待者（stop 时）
private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    uint64_t seq_ = 0;
    std::vector<uint8_t> jpeg_;
    int64_t stamp_ms_ = 0;   // steady 单调钟毫秒；只与同钟的 now 相减，不做跨源比较
    bool closed_ = false;
};

// 常驻 ffmpeg 拉流句柄（RAII）：解码出的 JPEG 写入帧槽；poll 看门狗在 VLM_PULL_STALE_MS
// （默认 10s）无数据时判流挂起，杀进程组重连；重连退避 1→30s（成功过帧即复位 1s）。
class PersistentPuller {
public:
    explicit PersistentPuller(LatestFrameSlot *slot) : slot_(slot) {}
    ~PersistentPuller() { reap(); }
    // 在专门的读流线程里跑；running=false 时返回。生命周期收进本句柄，
    // 析构兜底杀进程组（防 ffmpeg 僵尸/挂起，PLAN 危险清单 #3）。
    void loop(const std::string &rtsp_url, const std::string &tag, std::atomic<bool> &running);

private:
    void reap();   // kill(-pgid,SIGKILL)+waitpid 回收；可重复调用
    LatestFrameSlot *slot_;
    pid_t pid_ = -1;
    int fd_ = -1;
};
