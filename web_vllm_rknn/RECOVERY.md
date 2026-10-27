# 救砖预案（原地换内核失败时用）

换新内核后若重启不回来（SSH/ping 不通超过 3 分钟），按下面顺序处理。

## 方案 A：eMMC 里旧 boot-chain 回滚（需要板子还能进系统才用不上——boot 挂了进不去就用 B）

## 方案 B：USB-C maskrom 重刷（推荐，无需 SD 卡）
1. 板子断电，拔掉电源线。
2. 用 **USB-C 线**连接 NanoPi R6S 的 **Type-C OTG 口**（不是 PD 供电口，见官方 Wiki 丝印）和主机（这台 x86）。
3. 按住板上 **MASKROM 按键**（或短接 MASK 焊盘）同时上电，进入 maskrom 模式。
4. 主机确认：`lsusb` 应出现 `2207:350c` (Rockchip USB bootloader)。
5. 主机安装 rkdeveloptool：`sudo apt install rkdeveloptool`
6. 全量刷机（文件用新官方镜像解包后的全套）：
   ```
   rkdeveloptool ld
   rkdeveloptool wl 0x0 sd-img  # 或用 upgrade_tool/parameter 方式
   ```
   最简方式：把 `firmware/ubuntu-jammy-minimal-arm64/` 全套（MiniLoaderAll.bin, parameter.txt,
   uboot/misc/dtbo/resource/kernel/boot/recovery/rootfs/userdata.img）
   用 rkdeveloptool 按 parameter.txt 逐分区 `wl` 写入。
   （也可以先只 `wl 0x8000 dtbo.img` 等回滚旧分区——旧的四个分区备份在板子 /data/part-backup/
   以及主机侧，若取回过请核对 sha256：见本文件末尾。）

## 备份位置
- 板上：/data/part-backup/{dtbo,resource,kernel,boot}.img（旧系统，root 属主）
- 主机：/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/firmware/backup/（若已取回）

## 验证命令（刷成功后）
- `uname -r` → 6.1.141
- `dmesg | grep -iE "rknpu|rknn"` → driver_version ≥ 0.9.x
- `./run_base.sh` → 交互输入 1
