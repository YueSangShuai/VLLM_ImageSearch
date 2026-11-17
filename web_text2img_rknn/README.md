# web_text2img_rknn — MARS 双塔文搜图服务（RK3588）

输入英文描述 → 返回图库中 topk 最匹配的行人图像。模型为 MARS 框架（ALBEF 双塔）
checkpoint `/data/yuesang/LLM/260320.pth`，移植到板端 NPU（RKNN fp16）。

## 模型与转换链

| 塔 | 结构 | 输入 | 输出 |
|---|---|---|---|
| 图像塔 | EfficientNet-V2-S + classifier + vision_proj | uint8 RGB NHWC 224×224（mean/std=127.5 烘焙）| 256-d L2 归一化（图内） |
| 文本塔 | xbert 前 6 层（mode='text'）+ text_proj | input_ids i32 [1,128] + attention_mask f32 [1,128] | 256-d L2 归一化 |

检索 = 两向量点积（归一化后即 cosine）。x86 → 板端流程：

```bash
# 1. 导出双塔 ONNX + golden（MARS env；导出 wrapper 与原始 forward 全等校验内置）
cd /data/yuesang/LLM/MARS && PYTHONPATH=/data/yuesang/LLM/MARS \
  /home/yuesang/miniconda3/envs/MARS/bin/python \
  /data/yuesang/LLM/FG-CLIP/web_text2img_rknn/export/onnx_export.py

# 2. ONNX → RKNN fp16 + x86 模拟器对拍（rknn env）
cd /data/yuesang/LLM/FG-CLIP/web_text2img_rknn && PYTHONNOUSERSITE=1 \
  /home/yuesang/miniconda3/envs/rknn/bin/python export/rknn_convert.py
# 已验证：双塔模拟器 row-cos≥0.99997，topk(10) 18 条查询与 torch 全对（0.005 平局容差）

# 3. golden raw（板端对拍数据）
python3 scripts/golden/export_golden.py

# 4. 交叉编译（复用 web_vllm_cpp 工具链 + web_vllm_rknn rknn 头库）
bash cpp/build_scripts/build_aarch64.sh

# 5. 部署（板子 pi@10.8.18.118，目录 /home/web_text2img_rknn，端口 5005）
bash scripts/deploy/push_board.sh          # 推 output/ 到板上 ~/web_text2img_rknn/
bash scripts/deploy/start_svc_board.sh     # 板上纯净重启（经 ssh）
# （start_svc_board.sh 需在板上执行：先 push 再 python3 scripts/deploy/ssh_pi.py 30 'bash ~/web_text2img_rknn/../start_svc_board.sh' 或手动 nohup）

# 6. 板端 golden 对拍 + E2E
python3 scripts/deploy/ssh_pi.py 120 "cd /home/web_text2img_rknn && ./run_golden_rk.sh"
curl -s localhost:5005/api/search -d '{"text":"a woman in white short-sleeve shirt","topk":5}'
```

## 关键口径（改动前必读）

- **文本必须先过 pre_caption**（小写、去 `,.'"!?()*#:;~`、`-` `/` 换空格、50 词截断）：
  实测不清洗排序翻转。C++ 复刻在 `text_embedder.cpp::pre_caption`，与
  `MARS/dataset/utils.py` 逐行为对齐；golden 查询走清洗后文本。
- 文本塔只导出前 6 层（`xbert.py` mode='text' 即 fusion_layer=6）；扩展掩码
  `(1-mask)*-10000` 图内计算（-10000 在 fp16 安全；勿改回 finfo.min，RKNN BERT 老坑）。
- 双塔均 **fp16 不量化**（w8a8 对 BERT/EfficientNet 均不保排序）。
- 图像 resize：板端 stb 双线性 vs 训练端 cv2 INTER_LINEAR 有像素级差，
  golden 对拍（cos≥0.999）已量化其影响可忽略。
- 向量缓存：`data/gallery/{manifest.json,vectors.bin}` append-only；改图自动重编，
  删图打墓碑；`POST /api/rebuild` 手动触发增量。

## API

| 路由 | 说明 |
|---|---|
| `GET /` | 检索页（英文输入 + topk 网格 + 上传）|
| `POST /api/search {text, topk}` | → `{results:[{file,score,url}]}` |
| `POST /api/enroll` (multipart `image`) | 上传入库（`gallery/uploads/`，即时可检索）|
| `POST /api/rebuild` | 目录新图增量建库 |
| `GET /api/status` | 建库进度/图库规模 |
| `GET /img?file=<rel>` | 结果缩略图回读 |

## 配置（env，见 run_web_rk.sh.template）

`WEB_VLLM_PORT`(5005) `GALLERY_DIR` `MARS_IMAGE_RKNN` `MARS_TEXT_RKNN` `MARS_VOCAB`
`RKNN_MARS_CORE_NUM`(1) `URL_PREFIX` `MARS_TEXT_MAX_LEN`(128) `DEFAULT_TOPK`(10)

初始图库：`/data/yuesang/person_data/pa100k/images` 的拷贝放板上 `gallery/`
（GALLERY_DIR 可指任意目录，递归 jpg/png）。全量 10 万张首建 ~2h（后台增量，不挡服务）。

## 已知边界

- 查询语言：英文（文本塔 bert-base-uncased 训练于 CUHK-PEDES）；中文需上游翻译。
- 板端与 web_vllm_rknn(5003)/siglip(5004) 共存：本服务 5005、NPU 单核分时。
