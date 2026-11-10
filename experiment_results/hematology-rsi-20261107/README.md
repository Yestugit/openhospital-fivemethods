# 血液科 RSI 效率实验结果

本目录是已停止实验的结果快照：MemOS、Mem0、MemMachine、Memora 各完成 100 例（5×20），共 400 例；EMem 延期，没有有效的正式结果。无无记忆对照，无额外固定测试集。

- `efficiency-cases.*`：404 个逐病例增长点（含四个空库基线），RSI token、记忆正文 token 与存储大小。
- `efficiency-rounds.*`：20 个轮次，包含本轮与累计流程时间、记忆时间、持久化时间及诊疗评分。
- PNG 图表：逐轮与累计时间、表现及存储增长。
- `cases.json`：五方法计划共用的 100 例名单及顺序；`run-manifest.json`：原模型、硬件、配置及源码身份。
- `requests/`：实际请求成本账本，按大小拆为 JSONL；已知/未知 token、失败及重试按原记录保留。
- 各方法 `summary.json`、`report.md`、`rounds.json`：整体和逐轮结果。
- 各方法 `cases/001.json` 到 `100.json`：原始最终预测的完整内容，包括诊疗输出、评分及检索到的记忆；只拆文件，不截断。`case-index.json` 给出病例与轮次对应。
- `encounters/`：诊疗轨迹、最终结果、官方评分与反思；若 applied_memory 与对应病例完全相同，用 JSON Pointer `$ref` 引用该病例，避免重复保存。
- `operation-traces.jsonl`、`timing-events.jsonl`、`measurements/` 与 `case-usage/`：操作耗时、原始计时事件、存储/正文测量与已有逐例用量。

原本超过 GitHub 单文件限制的 results.json 已按上述形式拆分，语义内容保留在 summary、cases 和 operation-traces 中；没有上传重复的巨大 curve.json。原文件 SHA-256 见 methods.json。

**分析限制：**Memora 部分运行曾与已丢弃的 EMem 并行，存在故障、恢复及部分计时覆盖不完整的标记；不能宣称整个实验始终独占硬件或直接用它证明复杂度。不同轮次的病例不同，表现变化也受病例难度影响。

不包含模型权重、Python 环境、原生数据库、恢复检查点、旧 attempt 输出、API 密钥或压缩归档。数据用于查看结果和重做统计，不用于恢复已结束的实验。原始本机路径仅作为来源信息，不代表这些路径存在于克隆后的仓库。
