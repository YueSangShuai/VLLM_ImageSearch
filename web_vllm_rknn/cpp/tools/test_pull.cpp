// 常驻拉流机制的宿主端自测（不碰真 RTSP）：PATH 注入假 ffmpeg 脚本喂帧，验证
//   ① JPEG 流按整包切帧（chunk 与帧边界不对齐的场景）
//   ② LatestFrameSlot seq 单调 + latest-wins（处理慢时中间帧被覆盖即丢）
//   ③ 看门狗：假 ffmpeg 吐完就挂起 → VLM_PULL_STALE_MS 超时被杀 → 退避重连再收帧
//   ④ stop：running=false + close() 后 loop 快速返回，假 ffmpeg 进程组无残留
// 编译（宿主 x86，不进 CMake）：
//   g++ -std=c++17 -I../src -I<web_vllm_cpp>/third_party \
//       test_pull.cpp ../src/frame.cpp ../src/config.cpp -lpthread -o /tmp/test_pull
#include "frame.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

static const char * kBin = "/tmp/wv_pull_test.bin";
static const char * kDir = "/tmp/wv_pull_fake_marker";
static const char * kFakePath = "/tmp/wv_pull_fake_marker/ffmpeg";   // execlp("ffmpeg") 命中它
static const char * kFakePy = "/tmp/wv_pull_fake_marker/pull_fake_marker.py";

static int64_t ms_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool writeFile(const char * path, const std::string & s) {
    std::ofstream f(path, std::ios::binary);
    f << s;
    return f.good();
}

// 真实流探针：argv = <rtsp_url> [秒数，默认 8]。用线上同款 ffmpeg 参数常驻拉流，
// 统计帧数/帧间隔/分辨率（验证 image2pipe 常驻链路），不跑检测。
int main(int argc, char ** argv) {
    if (argc >= 2) {
        int secs = argc >= 3 ? atoi(argv[2]) : 8;
        LatestFrameSlot slot;
        PersistentPuller puller(&slot);
        std::atomic<bool> running{true};
        std::thread th([&] { puller.loop(argv[1], "probe", running); });
        std::thread deadline([&] {   // 到点收口（流全死时 waitNew 不会有帧，靠这路退出）
            std::this_thread::sleep_for(std::chrono::seconds(secs));
            running = false;
            slot.close();
        });
        uint64_t last = 0;
        int frames = 0;
        auto t_start = std::chrono::steady_clock::now(), t_prev = t_start;
        while (true) {
            if (!slot.waitNew(last, running)) break;
            std::vector<uint8_t> jpg;
            int64_t age = 0;
            last = slot.latest(jpg, age);
            auto now = std::chrono::steady_clock::now();
            int dt = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - t_prev).count();
            t_prev = now;
            ImageRGB im;
            bool dec = decode_jpeg(jpg, im);
            printf("frame #%llu 间隔 %dms %dx%d 解码=%d 帧龄=%lldms\n",
                   (unsigned long long)last, dt, im.w, im.h, (int)dec, (long long)age);
            frames++;
            if (std::chrono::duration_cast<std::chrono::seconds>(now - t_start).count() >= secs) break;
        }
        running = false;
        slot.close();
        th.join();
        deadline.join();
        printf("共收到 %d 帧\n", frames);
        return frames > 0 ? 0 : 1;
    }

    int failures = 0;
#define CHECK(cond, msg)                                                \
    do {                                                                \
        if (cond) printf("  [OK] %s\n", msg);                           \
        else { printf("  [FAIL] %s\n", msg); failures++; }              \
    } while (0)

    // ---------- 环境：3 张尺寸不同的 JPEG 拼成一路「流」；假 ffmpeg 分块吐 + 挂起 ----------
    std::vector<uint8_t> stream;
    int want_w[3] = {64, 100, 128};
    for (int s : want_w) {
        ImageRGB img;
        img.w = img.h = s;
        img.rgb.assign((size_t)s * s * 3, (uint8_t)(s + 7));
        auto jpg = encode_jpeg(img, 90);
        stream.insert(stream.end(), jpg.begin(), jpg.end());
    }
    { std::ofstream f(kBin, std::ios::binary); f.write((char *)stream.data(), stream.size()); }
    if (system((std::string("rm -rf ") + kDir + " /tmp/wv_pull_fake.pids && mkdir -p " + kDir)
                   .c_str()) != 0) return 2;
    // 假 ffmpeg：忽略参数，997B 一块吐完整帧（997 与帧长不对齐 → chunk 必跨帧边界），
    // 块间 50ms 让收帧线程来得及逐帧取走；随后挂 60s 供看门狗误杀演练。
    // 自报 pid 到文件（pgrep -f 会连「执行检查的 shell 自身」一起匹配，不能用）
    if (!writeFile(kFakePy,
               "#!/usr/bin/env python3\n"
               "import os,sys,time\n"
               "open('/tmp/wv_pull_fake.pids','a').write(str(os.getpid())+'\\n')\n"
               "d=open('/tmp/wv_pull_test.bin','rb').read()\n"
               "for i in range(0,len(d),997):\n"
               "    os.write(1,d[i:i+997])\n"
               "    time.sleep(0.05)\n"
               "time.sleep(60)\n")) return 2;
    if (!writeFile(kFakePath, "#!/bin/sh\nexec python3 " + std::string(kFakePy) + " \"$@\"\n"))
        return 2;
    chmod(kFakePath, 0755);
    chmod(kFakePy, 0755);
    setenv("PATH", (std::string(kDir) + ":" + getenv("PATH")).c_str(), 1);
    setenv("VLM_PULL_STALE_MS", "800", 1);

    // ---------- ① 帧槽原语义 + waitNew 对停止的响应 ----------
    printf("[1] 帧槽基础语义\n");
    LatestFrameSlot slot;
    CHECK(slot.ageMs() == -1, "无帧时 ageMs()=-1（Server 据此回退 RTSP 探测）");
    std::atomic<bool> dead{false};
    int64_t t0 = ms_now();
    CHECK(!slot.waitNew(0, dead) && ms_now() - t0 < 100, "running=false 时 waitNew 立即返回");

    // ---------- ② 常驻拉流收帧（切帧正确性） ----------
    printf("[2] 常驻拉流：切帧 + 解码\n");
    PersistentPuller puller(&slot);
    std::atomic<bool> running{true};
    std::thread th([&] { puller.loop("rtsp://fake/test", "test", running); });

    uint64_t last = 0;
    int round = 0, seq_changes = 0;
    bool framing_ok = true;
    t0 = ms_now();
    while (round < 3 && ms_now() - t0 < 8000) {
        if (!slot.waitNew(last, running)) break;
        std::vector<uint8_t> jpg;
        int64_t age = 0;
        last = slot.latest(jpg, age);
        ImageRGB im;
        if (!decode_jpeg(jpg, im)) { framing_ok = false; break; }
        if (im.w != want_w[round]) framing_ok = false;   // 帧内容必须成帧完整、顺序不乱
        round++;
        seq_changes = (int)last;
    }
    CHECK(round == 3 && framing_ok && seq_changes == 3,
          "997B 跨边界 chunk 下切出 3 帧完整 JPEG，seq=1..3 单调");

    // ---------- ③ 看门狗：假 ffmpeg 挂起 → 杀 → 退避重连 → 第二轮帧 ----------
    printf("[3] 看门狗重连（假 ffmpeg 吐完挂 60s）\n");
    t0 = ms_now();
    while (ms_now() - t0 < 10000) {
        std::vector<uint8_t> jpg;
        int64_t age = 0;
        uint64_t s2 = slot.latest(jpg, age);   // 不等待，轮询 seq 是否推进
        if (s2 > last) { last = s2; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(last > 3, "挂起 ~0.8s 后被看门狗杀掉重连，第二轮帧到达（latest-wins 槽继续推进）");
    {   // 帧龄合理性：刚到的帧 age 应该是小正数
        std::vector<uint8_t> jpg;
        int64_t age = 0;
        slot.latest(jpg, age);
        CHECK(age >= 0 && age < 3000, "latest() 报出的帧龄合理");
    }
    auto readPids = [] {
        std::vector<int> pids;
        std::ifstream f("/tmp/wv_pull_fake.pids");
        int p;
        while (f >> p) pids.push_back(p);
        return pids;
    };
    CHECK(readPids().size() >= 2, "假 ffmpeg 被拉起过 ≥2 次（证明确实发生了杀掉+重连）");

    // ---------- ④ stop：快速退出 + 进程组回收 ----------
    printf("[4] stop 回收\n");
    t0 = ms_now();
    running = false;
    slot.close();
    th.join();
    int64_t quit_ms = ms_now() - t0;
    CHECK(quit_ms < 1500, "running=false 后 loop 在 1.5s 内返回（poll/退避可打断）");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    bool all_reaped = !readPids().empty();
    for (int p : readPids())
        if (kill((pid_t)p, 0) == 0) all_reaped = false;   // ESRCH=已回收
    CHECK(all_reaped, "全部假 ffmpeg 实例（含挂起中的）都被 waitpid 收干净，无残留/僵尸");

    printf("\n%s (failures=%d)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures;
}
