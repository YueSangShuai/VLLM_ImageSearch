# web_vllm_rknn_q35 —— Qwen3.5-0.8B 版多模态文搜图平台（RK3588）

与 ../web_vllm_rknn（Qwen3-VL-2B-ft 线上版）**功能完全一致**的并行平台，唯一差异是 VLM 换成
Qwen3.5-0.8B（w8a8，未微调基座）。2026-10-26 实测基座即反超 ft-2B（13.5s vs 21.5s/轮、零环境句、
属性更细），故单独立项并行运行、为后续微调版铺路。

## 与线上版的差异（其余同源同配置）
| 项 | 线上 web_vllm_rknn | 本项目 |
|---|---|---|
| VLM | qwen3vl-ft_vision.rknn + ft-merged_w8a8.rkllm | **qwen3.5_vision.rknn + qwen3.5-0.8b_w8a8.rkllm** |
| 端口 | 5003 | **5004** |
| 板目录 | /home/web_vllm_rknn | **/home/web_vllm_rknn_q35** |
| 进程名 | web_vllm_rknn | **web_vllm_rknn_q35**（⚠️ comm 截断成 `web_vllm_rknn_q`，pgrep/pkill 必须 `-f "[.]/web_vllm_rknn_q35"`）|
| data | 板端现行 18 规则（阈值已现场调优 0.44~0.6）| 同配置种子（2026-10-26 从板上拉取），运行后各自独立累积 |

C++ 源码与线上版零改动同源：`rknn_image_enc.cc` 本就支持 n_output==1（q35 视觉单输出 [196,1024]），
vlm_describer 通用。**换模型只需改 run_web_rk.sh 的模型路径 + 端口。**

## 使用
- 重编：`bash cpp/build_scripts/build_aarch64.sh`（rknn-llm 为软链 → ../web_vllm_rknn/rknn-llm；
  data 种子只在缺 devices.json 时覆盖，不会冲运行数据）
- 推板：`python scripts/deploy/push_full_py.py`（整包 1.6G 到 /home/web_vllm_rknn_q35；板盘 ~2.6G free，紧张）
- 启停：板上 `bash /home/web_vllm_rknn_q35/start_board_q35.sh`（只动 q35 进程，绝不误伤线上）
- 查看：`http://10.8.18.118:5004/`（与 5003 同 UI）

## 已知问题 / 待办
- [ ] 微调 Qwen3.5-0.8B（LoRA on 现有 ft 行人数据集 → 合并 → rkllm 导出；目标：压弃权长文/环境段、
      「他/她」混用、对齐短句风格）。语言塔 ~0.5B，导出比 2B 快。
- [ ] q35 基座在行人不清晰时会输出大段环境描述（诚实弃权但污染 BGE 输入，靠 isNoiseClause 兜底）。
- [ ] 双实例共 NPU（各 3 核 + base_domain_id=1）当前共存稳定，但帧吞吐互有拖累；板上盘 2.6G，
      两套告警图都会涨，注意清理旧 alerts_images。
- [ ] 线上 web_vllm_rknn 曾于 10-26 06:30 前后无 OOM 迹象地死过一次（q35 上线前已发生，原因未定位），观察中。
