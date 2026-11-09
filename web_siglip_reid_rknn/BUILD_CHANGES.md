# 变更日志

- 2026-11-09: 整包推送 output/（769M，含 web_siglip_reid 二进制 Nov5 11:47 版 + models）到板 pi@10.8.18.118:/home/pi/output，板上校验 models/ 通过。新增 scripts/deploy/push_auto.py（push_full.sh 免交互版，pexpect 自动输密码）。注意 push_full.sh 末尾提示的 `cd ~//home/pi/output` 是路径拼接显示 bug，实际路径 /home/pi/output。
- 2026-11-09: 推送 output/data/{devices.json,tasks.json} → 板 /home/pi/output/data/（只推这两个，alerts_*/detection_log 运行态不动）。tasks.json md5 一致；devices.json 推后会被运行中的服务回写 status 字段（offline/online 属运行时状态，预期行为）。服务在跑（pid 377215），新任务是否生效需看是否热载，必要时重启 run_web_rk.sh。
- 2026-11-09: output/data/tasks.json 规则 positive/negative 全部改写为英文属性短句（沿用板上 demo 风格 "person, black upper clothing"；规则 name/设备名保留中文仅作 UI 显示，模型不用）。同时 threshold 0.55→0.15：0.55 是从中文 CLIP 项目拷来的，SigLIP 尺度正确属性只有 ~0.15–0.22（见 config.cpp:67、matcher.h），不改阈值规则永远不会命中。已推送板上。
