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

int main(int argc, char ** argv) {
    // ---------- 背压压测（PLAN 回归缺口 #8）：处理端停摆 12s「模拟 VLM」，
    // 期间读流持续进帧；醒来必须直接拿到「最新一帧」（seq 大跳、帧龄小），
    // 绝不能把积压的几百帧逐条吐出来（latest-wins 真生效） ----------
    if (argc >= 2 && std::string(argv[1]) == "stress") {
        const char * sdir = "/tmp/wv_pull_stress";
        if (system((std::string("rm -rf ") + sdir + " /tmp/wv_pull_stress.pids && mkdir -p " + sdir)
                       .c_str()) != 0) return 2;
        // 假 ffmpeg：每 100ms 吐一整包（3 张拼接 JPEG=3 帧），持续 15s ≈ 450 帧
        std::ofstream py(std::string(sdir) + "/stream.py");
        py << "#!/usr/bin/env python3\n"
              "import os,time\n"
              "open('/tmp/wv_pull_stress.pids','a').write(str(os.getpid())+'\\n')\n"
              "d=open('/tmp/wv_pull_test.bin','rb').read()\n"
              "end=time.time()+15\n"
              "while time.time()<end:\n"
              "    os.write(1,d)\n"
              "    time.sleep(0.1)\n";
        std::ofstream sh(std::string(sdir) + "/ffmpeg");
        sh << "#!/bin/sh\nexec python3 " << sdir << "/stream.py \"$@\"\n";
        py.close(); sh.close();
        chmod((std::string(sdir) + "/stream.py").c_str(), 0755);
        chmod((std::string(sdir) + "/ffmpeg").c_str(), 0755);
        setenv("PATH", (std::string(sdir) + ":" + getenv("PATH")).c_str(), 1);

        // 复用 [主测试] 的 3 帧样本文件（先确保存在：内容无所谓，只要都是合法 JPEG）
        {
            std::vector<uint8_t> s;
            ImageRGB img; img.w = img.h = 96; img.rgb.assign((size_t)96 * 96 * 3, 123);
            auto j = encode_jpeg(img, 90);
            for (int i = 0; i < 3; i++) s.insert(s.end(), j.begin(), j.end());
            std::ofstream f(kBin, std::ios::binary);
            f.write((char *)s.data(), s.size());
        }

        int failures = 0;
        printf("[stress] 处理停摆 12s vs 读流 ~30fps\n");
        LatestFrameSlot slot;
        PersistentPuller puller(&slot);
        std::atomic<bool> running{true};
        std::thread th([&] { puller.loop("rtsp://fake/stress", "stress", running); });

        uint64_t last = 0;
        std::vector<uint8_t> jpg;
        int64_t age = 0;
        if (!slot.waitNew(last, running)) { printf("  [FAIL] 无首帧\n"); return 1; }
        last = slot.latest(jpg, age);
        printf("  停摆前 seq=%llu\n", (unsigned long long)last);

        std::this_thread::sleep_for(std::chrono::seconds(12));   // 模拟 VLM 占用处理线程
        int64_t t0 = ms_now();
        if (!slot.waitNew(last, running)) { printf("  [FAIL] 停摆后等不到新帧\n"); return 1; }
        uint64_t seq2 = slot.latest(jpg, age);
        int64_t wake_ms = ms_now() - t0;
        uint64_t jump = seq2 - last;
        printf("  停摆 12s 后: seq %llu→%llu（跳过 %llu 帧）帧龄=%lldms 唤醒耗时=%lldms\n",
               (unsigned long long)last, (unsigned long long)seq2,
               (unsigned long long)(jump - 1), (long long)age, (long long)wake_ms);
        if (jump < 100) { printf("  [FAIL] seq 跳变过小（latest-wins 未生效？）\n"); failures++; }
        else printf("  [OK] 积压 %llu 帧被覆盖，只拿最新一帧\n", (unsigned long long)(jump - 1));
        if (age > 500) { printf("  [FAIL] 醒来拿到的是旧帧（age=%lld）\n", (long long)age); failures++; }
        else printf("  [OK] 醒来即拿新鲜帧（帧龄 %lldms）\n", (long long)age);
        if (wake_ms > 300) { printf("  [FAIL] 停摆后唤醒过慢 %lldms\n", (long long)wake_ms); failures++; }
        else printf("  [OK] 停摆结束 %lldms 内即唤醒（无 sleep 节拍）\n", (long long)wake_ms);

        running = false;
        slot.close();
        th.join();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        bool all_reaped = true;
        {
            std::ifstream f("/tmp/wv_pull_stress.pids");
            int p;
            while (f >> p) if (kill((pid_t)p, 0) == 0) all_reaped = false;
        }
        if (all_reaped) printf("  [OK] 流线程退出，压测假 ffmpeg 已回收\n");
        else { printf("  [FAIL] 有残留进程\n"); failures++; }
        printf("\n%s (failures=%d)\n", failures ? "STRESS FAILED" : "STRESS PASS", failures);
        return failures;
    }

    // 真实流探针：argv = <rtsp_url> [秒数，默认 8]。用线上同款 ffmpeg 参数常驻拉流，
    // 统计帧数/帧间隔/分辨率（验证 image2pipe 常驻链路），不跑检测。
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
