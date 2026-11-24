# vit_align 变更日志（新条目追加在底部）

- 2026-11-24 框架初建：双塔并联（Qwen3-VL-2B 视觉塔冻结 + ProjMLP 可训，BGE-small-zh-v1.5
  冻结），对称 InfoNCE（可选同 id 多正例），两阶段离线缓存训练；CUHK-PEDES 数据管道；
  eval R@K/mAP（无前缀 + 加前缀双口径）；export_head.py 导出 vproj_head.bin
  （magic VPROJ1，仿 attr_head.bin，带 --verify numpy 复刻零差校验）。
- 2026-11-24 端到端跑通并出首轮结果：BGE HF vs 生产 ONNX selfcheck cos=1.0；提取
  train15070/test8335；仅 MLP i2t R@1 0.105；扫参发现文侧恒等初始化轻投影是大头
  （text_proj 开：R@1 0.2169 / R@5 0.446 / mAP 0.171，configs/tp600.yaml），MARS 锚点 0.25。
  加前缀英文查询掉到 0.115（需中文查询实测/中文描述重训）。verify --ckpt 缺省误用踩坑已在
  注释标明；eval.sh 支持 CFG= 透传。bin 零差 max|Δ|≈2e-7。
