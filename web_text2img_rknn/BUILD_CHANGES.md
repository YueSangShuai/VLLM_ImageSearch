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
