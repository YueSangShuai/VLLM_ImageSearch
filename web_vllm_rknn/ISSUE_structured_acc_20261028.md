# 问题记录：结构化属性输出「突然不准」全案（2026-10-28）

> 状态：**已定案并修复上线**。变更流水见 BUILD_CHANGES.md，本文是完整的问题档案。

## 症状
- 线上告警的结构化属性字段（`attributes`）退化为：全量抄写 prompt 里的颜色属性列表（27 个 `u*/l*` 一次吐完）、或输出列表外乱造词（`ulwhite`、`Lipad`、`pPink`、`bgcolor=ublack`）。
- 用户印象「之前很准」，怀疑新转换的 `models/qwen3vl-ft_vision_rk3588.rknn` 精度劣化。

## 排查实验（可复跑）

| # | 实验 | 命令/位置 | 结果 → 结论 |
|---|---|---|---|
| 1 | 文件完整性 | 板上/宿主 `md5sum` 全部模型 | 一致 → 排除文件损坏/被换 |
| 2 | 视觉塔 A/B | `scripts/bench/bench_struct_vision_ab_board.sh` | 固定 ft LLM 只换 vision（ft 塔 vs 基座塔），4 图输出**逐字相同** → 与视觉塔无关 |
| 3 | 描述轮核验 | 同上 demo 首轮 | 逐图正确（眼镜/裙装/肩包）→ 视觉链路健康 |
| 4 | 宿主 bf16 对照 | GPU + transformers greedy | 描述全对；同结构化 prompt 也全枚举 → 排除量化、排除权重 |
| 5 | 去列表实验 | 宿主，prompt 不带属性表 | 输出自由词（handbag/phone…）→ 列表必须保留（词表锚定） |
| 6 | penalty A/B | 6 张实际翻车 ROI，`bench_models --structured`（1.2/1.4/1.6） | 1.2→3/6 全枚举；1.4→枚举清零；1.6≈1.4 → 结构化用 1.4 |

## 根因（四层叠加）
1. **Prompt 复读陷阱**：结构化 prompt 尾部挂 57 属性列表，greedy 下任何模型（含 bf16）都会照抄。历史「准」的唯一来源是 host vLLM 链路带 `repetition_penalty=1.2`（`web_vllm/processor.py:300`，只罚生成段）。
2. **测试工具假象**：rknn-llm 官方 demo REPL 是 `getline` 逐行对话（main.cpp:425），多行 prompt 被拆成多轮、第 2 轮起**没有图像**（main.cpp:447 仅行内含 `<image>` 才注入）；且 demo 无 penalty。旧 bench 脚本的多行 prompt 结论全部作废。
3. **线上模型被换**：10/26 建 q35 平台时把板上 `run_web_rk.sh` 默认模型改成了 Qwen3.5-0.8B 基座（从未微调结构化）且未换回；之后服务又整体停过。
4. **W8A8 需要更高 penalty**：修复上线后新告警 41 条仍有 31 条全枚举——W8A8 在低质量小图上 1.2 压不住，需 1.4。

## 修复清单（全部已上线）
| 修复 | 位置 |
|---|---|
| 结构化专属 `VLM_STRUCTURED_REPETITION_PENALTY=1.4`（描述路径保持 1.2，防文本劣化） | `cpp/src/config.{h,cpp}` |
| matcher 复读守卫：解析后 ≥8 个颜色属性 = 判抄表，整条属性丢弃回退描述向量匹配 | `cpp/src/matcher.cpp extractStructuredAttributes` |
| `bench_models --structured`：服务同款路径离线测结构化，逐图合法/非法属性核对 | `cpp/src/bench_models.cpp` |
| 属性表抽成共享头（matcher/bench 单一来源） | `cpp/src/attributes.h` |
| 旧 bench 脚本单行 prompt 修复 + 「demo 结论作废」注释 | `scripts/bench/bench_struct_*.sh` |
| 板上旧部署整删、output_cpp 以 tar -h 重推、纯净 env 拉起 | `/home/Desktop/output_cpp` |

## 当前已知残留（非缺陷，按接受处理）
- W8A8 结构化偶发列表外 token（`ulwhite` 等）：matcher 解析按属性表过滤，不影响报警。
- penalty 1.4 下模型偶尔输出 `UWhite`/`side` 等变体：属性表查找表本来就大小写/去分隔符不敏感，能命中。
- 板卡时钟比宿主慢约 14 天：告警日期目录以板端时间为准（现落在 `alerts_2026-10-14.json`）。

## 经验教训（下次别再踩）
1. **测结构化一律用** `./run_bench_rk.sh --skip yolo,bge --structured --image <目录>`；demo 无 penalty + 逐行拆轮，结论无效。
2. 判断「模型变差」前先对齐**解码参数**（penalty/温度）——同一个模型换个采样参数观感天差地别。
3. 服务重启必须 `env -i`（start_svc_board.sh 已包），测试 export 的 `VLM_MODEL_NAME` 会顺着 `${VAR:-default}` 把服务带成别的模型。
4. `build_aarch64.sh` 会用宿主 `web_vllm_cpp/data` 覆盖 `output_cpp/data`，重编后先恢复（`/tmp/output_cpp_data_bak_*`）再推板。
5. 板上整包推送用 `tar -h`（解 symlink）；只更新二进制要先 `pkill -x web_vllm_rknn` 否则 Text file busy。
