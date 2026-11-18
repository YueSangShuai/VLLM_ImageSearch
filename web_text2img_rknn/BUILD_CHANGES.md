# BUILD_CHANGES

## 2026-11-17 立项：MARS 260320.pth 双塔 → RK3588 文搜图服务

- 立项：从 web_vllm_rknn/web_siglip_reid_rknn 复刻骨架（板端 C++/httplib，端口 5005）。
- 导出：`export/onnx_export.py`（MARS env）双塔 ONNX。checkpoint 用内嵌 `ck['config']`
  建模，0 missing/0 unexpected；导出 wrapper 与原始 forward maxdiff=0（图像塔
  v[:,0,:]→vision_proj；文本塔 embeddings+前 6 层+CLS+text_proj）。ORT 对拍 3e-7。
- 转换：`export/rknn_convert.py`（rknn env）fp16 双塔。模拟器对拍
  image/text row-cos min 0.99998/0.99997；topk(10)×18 查询与 torch 全对（0.005 平局容差，
  尾名 -0.1781 vs -0.1781 级别的重排属 fp16 必然，无检索语义差）。
- 坑：`fusion_layer` 在 bert_config json 里而非顶层 config（取 `model.text_encoder.config`）；
  pa100k 图库混有坏图需跳过。
- golden：`golden/`（torch 向量 + token ids + 期望 topk）；板端 `golden_check` 走
  生产同款 ImageEmbedder/TextEmbedder 路径（分词全等 + cosine + topk 四步核验）。
- C++：text_embedder 手写复刻 pre_caption + HF BertTokenizer(uncased)（lower→去音典→
  标点/CJK 隔离→WordPiece→定长 128 pad=0）；gallery 为 append-only 向量缓存 +
  增量扫描；服务路由 search/enroll/rebuild/status/img。
- 交叉编译通过（复用 web_vllm_cpp 工具链，无 ORT/sentencepiece 依赖）。
  编译期坑：本仓 httplib 的 multipart 类型是 `MultipartFormData`（非 *Content）且
  `get_file_value` 按值返回；`find_package(Threads)` 必须置于 target 之前；
  gallery.cpp 局部变量 `total` 遮蔽成员 `total()`。
- 板端 E2E 物料：`scripts/eval/make_gallery_subset.py`（pa100k 前 K 张确定性拷贝）、
  `scripts/eval/baseline_topk.py`（x86 torch 基线 topk）、`scripts/eval/compare_board.py`
  （平局容差排序对比）。golden raw 由 `scripts/golden/export_golden.py` 生成，板端
  走 `output/run_golden_rk.sh`。
- 待办：板上 golden → E2E 与 x86 基线 topk 对齐。

## 2026-11-17 板端验证收尾：golden + E2E 基线对齐 全部 PASS

- 板上目录实为 `/home/pi/web_text2img_rknn`（README 已修正）。500 张 pa100k 子图库建库完成。
- 板上 golden：分词 ids 全等（18/18）、text cos≥0.99981、image cos≥0.99968；
  topk 差异均为尾名平局重排，golden_check PASS。
- 清理上次测试残留 `gallery/uploads/*test_enroll.jpg`（rebuild 后 alive=500，墓碑保留）。
- E2E vs x86 torch 基线（`compare_board.py`，18 查询 × top20）：平局外排序不符 **0**、
  max|score diff|=0.0119（<0.03 阈值）、top20 重合 19.39/20、top3 集合 17/18 —— 唯一
  不一致为 0.6220/0.6215（差 0.0005）平局带换位，语义等价。→ **RESULT: PASS**
- 剩余（非阻塞）：全量 10 万张建库上线；查询仅英文，中文需上游翻译。

## 2026-11-17 全量 10 万张上线 + 全量对拍暴露 resize 漂移 → cv2 定点复刻修复

- 全量建库：pa100k 10 万张（526MB tar，mtime 判新使已有 500 张免重编），~68 分钟完成，
  服务在线增量不中断；冒烟检索 71ms 正常。查询仅英文（BERT uncased）。
- 全量对拍（18 查询×top20 vs x86 torch 基线）：**平局外排序不符 2、max|Δ|=0.0426**（500 张时 0）。
  逐图归因：JPEG 解码（stb vs libjpeg）cos≥0.9996 可忽略；**漂移全部来自
  stb_image_resize vs cv2 INTER_LINEAR**（小图 55×150→224 上采样，emb cos 低至 0.9925）。
  10 万库 top20 相邻分差 ~0.005 级，0.005 漂移足以换尾名；top1/top2 全稳。
- 修复：`frame.cpp::resize_bilinear` 改为复刻 cv2 remap 定点（11-bit 权重、半像素对齐、
  clamp 边界、(v+1<<21)>>22 SIMD 同序舍入），x86 probe 与 numpy 逐像素对齐后交叉编译上线。
  离群图 emb cos 0.9925~0.9991 → **0.9951~0.9997**。残余：OpenCV 4.12 8U 路径有 ~0.13/像素
  系统性偏置无法从 8U 输出反推，字节级不可复刻（区间交集法验证权重律 11-bit-rint 成立）。
- 部署换新二进制后清空向量缓存全量重编（~90 分钟，26/s→被 web_vllm_rknn 挤至 21/s）。
- 坑：ssh 一行流 pkill 与含路径命令必须分离（已按记忆执行）；板上 /tmp 大包用完即删。
- 待收尾：重编完成后 golden 复跑 + compare_board 全量复拍，结果补录本条。

## 2026-11-17 (晚) 根因终极突破：OpenCV 4.12 8U resize 逐字节复刻

- 11-bit 近似版重编后 compare 仍 CHECK-ABOVE（同样 2 条、max|Δ|0.0377）→ 说明残余漂移大头
  不是 stb，而是 **cv2 4.12 自身不是教科书双线性**。黑盒区间交集法定位 + 直读
  OpenCV 4.12.0 resize.cpp 源码确认三处私有行为：
  1. 系数用 **float32** fx、两权重**独立**银行家舍入（nearbyint，勿 lround）；
  2. 纵向 8U 走 `VResizeLinearVec_32s8u` SIMD **截断律**
     `((S0>>4)*b0>>16)+((S1>>4)*b1>>16)+2>>2`（平均偏 **-0.13/像素**，全局单向 −1 的来源）；
  3. y 越界权重不置零（invoker 只 clamp 行号），x 边界才 fx 清零。
- `frame.cpp::resize_bilinear` 改为逐字节复刻（含标量尾 (v+2^21)>>22；672 宽无尾）。
  验证：13 种随机尺寸 + 20 张真实 JPEG，**与 cv2.resize 输出 0 字节差异**（resize 环节）。
- 全管线 vs 基线仅剩 stb vs libjpeg 解码噪声（0.6% 字节、±1 LSB、emb cos≥0.9996）——
  除非板端引入 libjpeg-turbo，否则为不可再压缩的地板。
- 第三次全量重编上线（二进制 md5 b76707d4）。待 golden + compare 复拍结果补录。
