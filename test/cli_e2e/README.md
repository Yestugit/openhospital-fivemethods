# 真实 CLI 核查：2026-10-24

本页记录修复前 `50200dc` 的运行结果。合并后的修复已接通批次、轮次和日末处理，并允许带合法 outcome 的 streaming 输入。最新回归结果见各方法测试；本页保留的旧运行数据用于对照。

结论：四种方法的完整学习闭环尚不能通过标准 CLI 验收。`experience` 可以导入训练轨迹并冻结读取，但没有训练检索和原生 batch/epoch/day 命令；`streaming` 对这些插件的带历史输入直接拒绝。下面区分默认算法配置和专门检查读写链路的显式 smoke 配置。

本次从 `fix/ael-integration@50200dc` 运行真实 `.venv/bin/python -m mematlas run`，依序完成 MemRL → MemCon → MemQ → AEL。没有替换 provider、改算法或安装依赖。LLM 使用 SiliconFlow 的 `Qwen/Qwen2.5-72B-Instruct`；MemRL/MemQ 使用 `Qwen/Qwen3-Embedding-8B`、4096 维。MemCon 使用原生 CPU `all-MiniLM-L6-v2` 的已有本地缓存。它们是接入检查，未复现论文模型、环境或实验分数。

## 主要结果

每次 `experience` 使用两个有外部 success/reward 的训练用例、两个无 outcome 的测试用例，并显式共享同一 scope。以下运行均退出 0、框架操作失败数 0。

| 运行目录 | 配置 | 原生持久条目 | 待处理学习 | 两题各自检索命中 | provider 失败尝试 |
| --- | --- | ---: | --- | --- | ---: |
| `memrl-default` | 默认 HLE 算法，batch32 | 0 | pending batch 2 | 0 / 0 | 0 |
| `memrl-episode` | 显式 `hle_episode`，batch1 | 2 | 无 pending batch | 1 / 1 | 4 |
| `memcon-default` | 原生默认策略 | 4 | 不由这些计数判断 policy 学习完成 | 2 / 2 | 5 |
| `memq-default` | 默认 LCB 算法，batch50、epoch5 | 0 | pending batch 2、epoch 未结束 | 0 / 0 | 0 |
| `memq-batch-smoke` | 显式 `batch_size=1`，其余算法参数默认 | 2 | pending edges 2、epoch 未结束 | 1 / 1 | 6 |
| `ael-structured-default` | 默认 portfolio，带匹配 metadata | 2 episodic | controller 更新0、day0 | 0 / 0 | 0 |
| `ael-structured-read-smoke` | 显式只选 `full_detailed`，带匹配 metadata | 2 episodic | controller 更新0、day0 | 1 / 1 | 0 |

有命中的四项运行，两个测试回答均为 `COBALT-42`，exact match=1。默认 MemRL/MemQ 和默认 AEL 的回答为信息不可用。上述命中数是每题条目数，不是准确率；MemCon 的4条写入与原生每 episode 双写一致，没有去重。

## 已确认的问题与边界

1. **CLI 反馈学习链不通。** 四插件均采用 `history_ingestion=outcome`。`streaming` 有 history 时即抛 `CapabilityMismatchError: Streaming history requires labeled ingestion for this method; use the train/test protocol`，即使记录有合法 outcome。`experience` 训练只执行 `ingest_history`，无 retrieve；测试只读、不 learn。此次 MemRL/MemQ 原生 `q_visits` 均为0，不能把 `learn_count=2` 当作已检索记忆收到 Q 更新。MemCon 的策略动作更新及 AEL controller/usefulness 反馈也不能由本次导入验收。
2. **CLI 缺少原生学习边界。** 默认 MemRL/MemQ 的两条数据只被持久化，未满批不调用 native add。即使 MemQ 显式减小 batch，`end_epoch` 未发送，两个 Bellman edges 仍待处理。AEL 未收到 `end_day`，未执行夜间蒸馏/策略演化；输入没有 `_warmup`，也不构成论文 warmup 流程。没有将 commit/close 改作学习边界。
3. **AEL 默认策略的实际冷启动结果。** 原生 Thompson Sampling 在 seed42、当前未获反馈的 arms 上首先选择 `none`。测试快照恢复 NumPy RNG，因此第二题也选 `none`；portfolio fallback 只查 semantic/procedural，而库中只有 episodic，故仍为空。带匹配 metadata 的默认运行也复现此结果。保留了原生选择和快照行为，没有强制把默认策略换为 `full_detailed`。
4. **AEL 需要原生检索字段。** `ael-default`、`ael-read-smoke` 另用通用 `experience.json` 检查，两者均0命中：通用文本未提供可匹配的 ticker/task_type 等，原生 relevance 从0开始，query 文本不参与向量检索。`ael-experience.json` 在 outcome metadata 写入合成 `ticker/task_type`，并在测试 metadata.memory_query 明确查询它们；只有配合显式 read-smoke 才检索成功。这是输入接口要求，不是更换检索算法。
5. **退出0并不代表无错误日志。** MemRL 默认空库恢复两次记录 MemOS `Empty update request` ERROR：原生 loader 将空数组交给 Qdrant，再捕获并吞掉异常；本次空库恢复后仍为空。MemCon 记录 Chroma telemetry `capture() takes 1 positional argument but 3 were given` ERROR，未阻断数据读写。MemRL/MemCon/MemQ 的实际读写运行分别出现4/5/6次 `APIConnectionError`，均重试恢复；框架 ledger 只记录异常类型，未记录 cause，不能据此认定外网故障。同步桥接反复创建 loop 与复用异步客户端是待查线索，尚未证实根因。

标准 CLI 的 streaming 失败报告是 `memrl-streaming`、`memcon-streaming`、`memq-streaming`、`ael-streaming`，均退出1、在 start 阶段拒绝。MemRL 使用其合法 message 轨迹；其他三者使用 `actions.json`。`memrl-streaming-actions` 另被 MemRL 不支持的 `episode_start` 拒绝，属于事件兼容性探针，不用它证明 streaming 问题。`fixture-rejected` 为最初不合法的 outcome 字段探针，不计入插件验收。

## 可复现命令

从仓库根目录执行。测试输入 JSON 只记录凭据环境变量名；MemRL/MemOS 的原生 cube/config.json 会保存展开后的 provider 凭据。本次完整本地产物包含这些字段，目录已限制为仅当前用户访问，不能直接分享整个产物目录；`run.sh` 用 umask077 保护后续运行。先按各方法 README 安装 manifest 已声明的依赖；本轮使用现有共享 `.venv`，没有改变其已知的依赖冲突。

```bash
cd /home/lzy/project/mematlas
set -a
source /home/lzy/.config/mematlas/env
set +a
export MEMATLAS_CLI_E2E_ROOT="$PWD/.mematlas/runs/cli-e2e-$(date +%Y%m%d-%H%M%S)"

bash test/cli_e2e/run.sh memrl-default memrl.default.json experience.json
bash test/cli_e2e/run.sh memrl-episode memrl.episode.json experience.json
bash test/cli_e2e/run.sh memcon-default memcon.default.json actions.json
bash test/cli_e2e/run.sh memq-default memq.default.json experience.json
bash test/cli_e2e/run.sh memq-batch-smoke memq.batch-smoke.json experience.json
bash test/cli_e2e/run.sh ael-structured-default ael.default.json ael-experience.json
bash test/cli_e2e/run.sh ael-structured-read-smoke ael.read-smoke.json ael-experience.json
```

以下命令用于复现协议拒绝，每条预期退出1；请分别执行，以免外层 `set -e` 在第一条后停止。

```bash
bash test/cli_e2e/run.sh memrl-streaming memrl.episode.json experience.json streaming train
bash test/cli_e2e/run.sh memcon-streaming memcon.default.json actions.json streaming train
bash test/cli_e2e/run.sh memq-streaming memq.default.json actions.json streaming train
bash test/cli_e2e/run.sh ael-streaming ael.default.json actions.json streaming train
```

补充 AEL 的通用文本输入对照：

```bash
bash test/cli_e2e/run.sh ael-default ael.default.json experience.json
bash test/cli_e2e/run.sh ael-read-smoke ael.read-smoke.json experience.json
```

`run.sh` 内直接调用公开 CLI；每个运行有独立 cwd、artifact、report 和 `cli.log`，拒绝覆盖已有运行目录，超时360秒。`HF_HUB_OFFLINE=1` 只控制 MemCon 从已有缓存加载原生 embedding model；没有缓存时先按该方法 README 准备原模型。

`instrumentation/sitecustomize.py` 仅设置日志，并拒绝常规 Python open/import 对 `plan/` 词法路径的访问；它不是覆盖符号链接、原生 I/O 或已有文件描述符的文件系统沙箱。MemOS 重新设置 root logger 后会补回命名诊断 handler；不替换算法或 provider。日志以 `mematlas.methods.<id>.adapter` 标识接入层，`memrl.*`、`memq.*`、`memos.*`、`[MemCon]` 标识原生输出；`ael.native.calls` 是明确标记的接入层调用插桩，不冒充上游静默函数内部日志。

配置中的模型、base URL 和4096维是显式 transport 覆盖；MemRL 的 `llm_max_tokens=2048` 是此次模型输出限额。`hle_episode`、MemQ batch1、AEL 单 policy 均是可见的测试配置差异，未写入插件默认值，也不能代替完整论文学习验收。

## 本次证据与核查工具

当前完整产物在 `.mematlas/runs/cli-e2e-20261024/<运行目录>/`：`cli.log`、`results.json`、`report.md`、分方法 predictions/traces，以及 `artifacts/runs/<run_id>/operations.jsonl`、`usage.jsonl` 和原生数据库/state。旧快照仍在；最初 MemRL 产生的仓库 cwd 下原生日志/文件已保留到该根目录的 `native-global-backup/`，工作树里的这些生成物已恢复或移出。后续运行使用独立 cwd。

只读统计工具不导入插件、不调用模型；MemCon SQLite 使用 `mode=ro` 和 `query_only`，不会重新初始化 Chroma：

```bash
.venv/bin/python test/cli_e2e/summarize.py \
  "$MEMATLAS_CLI_E2E_ROOT/memrl-default" \
  "$MEMATLAS_CLI_E2E_ROOT/memrl-episode" \
  "$MEMATLAS_CLI_E2E_ROOT/memcon-default" \
  "$MEMATLAS_CLI_E2E_ROOT/memq-default" \
  "$MEMATLAS_CLI_E2E_ROOT/memq-batch-smoke" \
  "$MEMATLAS_CLI_E2E_ROOT/ael-default" \
  "$MEMATLAS_CLI_E2E_ROOT/ael-read-smoke" \
  "$MEMATLAS_CLI_E2E_ROOT/ael-structured-default" \
  "$MEMATLAS_CLI_E2E_ROOT/ael-structured-read-smoke" \
  "$MEMATLAS_CLI_E2E_ROOT/memrl-streaming" \
  "$MEMATLAS_CLI_E2E_ROOT/memcon-streaming" \
  "$MEMATLAS_CLI_E2E_ROOT/memq-streaming" \
  "$MEMATLAS_CLI_E2E_ROOT/ael-streaming"
```

本次 `summary.json` 包含上述13个报告：9个 experience 对照与4个 streaming 拒绝结果；不包括两次输入格式/事件类型探针。

报告可观察到 save/load 与冻结测试路径成功；MemQ/AEL 最终 state 中检索计数已回滚。本次 CLI 插桩没有独立采集冻结前后逻辑 state hash，不能以重复答案、最终 hash 或操作成功冒称完整相等证明。严格快照恢复另由原有各方法离线 suite 验证；本轮未重复运行那些 suite。AEL structured 样本仍通过通用 `add_episodic` 写入，没有覆盖需要 tool_evaluations/prediction/actual 的 rich `add_episode_experience`。没有验证跨进程 resume、真实日末 LLM evolution、完整 agent 或金融回测。

本轮只新增 CLI 核查资料并更新文档；没有修改 `core/`、`runtime/`、`providers/`、`methods/` 或 `main`。框架限制按五字段记录于根目录 `difference.md` 的【框架待优化项】，原生错误及测试配置差异另列，未将优化或上游 bug 修复并入实现。
