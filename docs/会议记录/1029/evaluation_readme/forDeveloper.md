# 测评运行逻辑与接口

本文只说明当前 QA 测评的三部分：总体调用链、`streaming` 流程和 `experience` 流程。重点是执行顺序，以及每一步交给下一层什么数据。

本文所说的“训练”是 Memory 读取经历并按方法能力更新记忆，不是训练 LLM 参数。默认 CLI 为每个案例生成一次回答，不包含通用多轮 Agent 或工具调用。

## 1. 总体调用链

### 1.1 从 CLI 到报告

```text
CLI: run
  ↓ 读取参数、配置和模型设置
_run_evaluation
  ├─ 创建回答用 LLM / 可选 LongMemEval judge
  ├─ load_benchmark(...) → BenchmarkAdapter
  └─ 读取一个或多个 MemAtlasConfig
  ↓
ExperimentRunner.evaluate_qa(...)
  ↓
qa_spec(...)
  ├─ 为每个配置创建 MethodRun
  ├─ 创建默认 memory_qa_answerer
  └─ 创建 StreamingMemoryProtocol 或 ExperienceTrainTestProtocol
  ↓
ExperimentSpec
  ↓
MatrixRunner.run(...)
  ├─ 按 test split 读取案例
  ├─ 对每个 MethodRun 单独执行完整运行
  ├─ 调用 Evaluator 计算质量和资源
  └─ 调用 ReportWriter 写结果
```

当前 CLI 的 `run` 入口在收到参数后进入 `_run_evaluation`。它根据 Benchmark 类型和数据路径创建适配器，读取 Memory 配置，并构造回答问题的 LLM。随后调用公开的 `ExperimentRunner.evaluate_qa(...)`。

`evaluate_qa` 不直接执行案例，而是委托 `qa_spec(...)` 组装 `ExperimentSpec`：

- `MethodRun`：一份 `MemAtlasConfig` 加上该次方法运行使用的 services；
- `answerer`：接收案例与 MemoryContribution，返回回答或 Prediction；
- `protocol`：决定案例历史如何进入 Memory，以及训练、回答和反馈的顺序；
- `benchmark`：提供案例和 Benchmark 原生评分。

之后 `MatrixRunner.run(...)` 负责执行。它先按指定 `split` 读取测试案例，再依次处理每个 `MethodRun`。每种方法的执行步骤如下：

```text
校验方法、测试输入和训练输入
  ↓
创建并初始化 Memory
  ↓
protocol.begin_run
  ↓
protocol.prime + Memory.flush
  ↓
逐个调用 protocol.run_case
  ↓
无案例失败时调用 protocol.end_run
  ↓
Memory.close
  ↓
Evaluator.evaluate_async
  ↓
生成 EvaluationResult
```

### 1.2 核心接口与数据交接

| 接口 / 对象 | 输入 | 输出或职责 |
| --- | --- | --- |
| `BenchmarkAdapter.cases(split)` | split 名称，如 `train`、`test` | 产出 `BenchmarkCase` |
| `BenchmarkCase` | `case_id`、`scope`、`history`、`query`、`reference`、可选 `outcome`、`metadata` | 表示一个训练或测试案例 |
| `EvaluationProtocol` | Memory、BenchmarkCase、BenchmarkAdapter | 校验输入并编排生命周期：`begin_run`、`prime`、`run_case`、`end_run` |
| `Memory.ingest_history(events, outcome=None)` | 历史事件及可选 EpisodeOutcome | 按方法声明的 ingestion 策略导入历史，可能进一步调用 observe、learn、flush |
| `Memory.retrieve(DecisionContext)` | scope、query、可选 history 和 metadata | 返回 `MemoryContribution` |
| `Answerer(case, contribution)` | 完整 BenchmarkCase 和 MemoryContribution | 返回普通回答，或返回与该 case 对应的 Prediction |
| `Memory.learn(outcome, events, retrieved)` | 结果反馈、经历事件、使用过的记忆来源 | 将反馈交给具体 Memory 方法；不负责答案评分 |
| `Evaluator.evaluate_async(...)` | Predictions、Benchmark、traces、usage | 返回质量 `metrics` 和资源 `resource_metrics` |
| `ReportWriter.write(results, output_dir)` | EvaluationResult 列表和目录 | 写报告、JSON 结果、逐题预测、轨迹和 manifest |

`DecisionContext` 是检索请求，不是写入指令。`MemoryContribution` 是 Memory 返回给回答阶段的内容，可能包括检索条目、整理后的上下文或直接答案。`Prediction` 保存一个案例的模型输出、评分 reference、检索来源和 metadata。

当前 QA 调用接口可概括为：

```python
async def ExperimentRunner.evaluate_qa(
    configs,
    benchmark,
    *,
    llm,
    model,
    report_dir=None,
    split="test",
    protocol="streaming",
    train_split="train",
    usage=None,
): -> list[EvaluationResult]

async def MatrixRunner.run(
    methods: Iterable[MethodRun],
    benchmark: BenchmarkAdapter,
    protocol: EvaluationProtocol,
    *,
    split: str = "test",
    report_dir: str | Path | None = None,
): -> list[EvaluationResult]
```

Protocol 的主要生命周期接口是 `begin_run(memory, benchmark)`、`prime(memory, benchmark)`、`run_case(memory, case) -> Prediction` 和 `end_run(memory)`，均为 async。`prime` 和 `end_run` 在基类中可以是 no-op，由不同 protocol 定义实际工作。

默认 `memory_qa_answerer` 只把 query 和检索内容放进提示词。Answerer 虽然收到完整
`BenchmarkCase`，但不能把其中的 `reference` 泄漏给回答模型；`reference` 只交给
Evaluator 评分。

## 2. `streaming` 流程

### 2.1 运行语义

`streaming` 没有单独的 train split。测试案例按顺序处理，每个案例先导入自己的 history，然后检索、回答；案例有 outcome 时，在回答之后再把结果反馈给 Memory。

```text
开始运行：epoch_start（若方法声明了该边界）
  ↓
对每个 test case：
  选择本案例尚未导入的 history 事件
  → Memory.observe / Memory.ingest_history
  → Memory.retrieve(DecisionContext)
  → Answerer(case, MemoryContribution)
  → 得到 Prediction
  → 若 case.outcome 存在：Memory.learn(...) → Memory.flush()
  ↓
所有案例成功时：epoch_end → Memory.flush()
```

标准 CLI 的 `qa_spec` 将 history policy 设为 `EVENT_ONCE`。重复事件按 `(scope.key(), event_id)` 去重写入，但当前案例的完整 history 仍会放入 retrieve 请求的 `DecisionContext.history`。

### 2.2 单案例调用顺序

`StreamingMemoryProtocol.run_case` 的逻辑可读作：

```python
async def run_case(memory, case):
    if reset_between_cases:
        await memory.reset(case.scope)

    events = select_history_events(case.history, history_policy)
    ingestion = memory.plugin.history_ingestion(memory.config.method)

    if ingestion == "outcome":
        for event in events:
            await memory.observe(event)
    else:
        await memory.ingest_history(events)

    prediction = await predict(memory, case)

    if case.outcome is not None:
        outcome = add_episode_id_and_case_metadata(case.outcome, case)
        await memory.learn(
            outcome,
            events=case.history,
            retrieved=outcome.metadata.get("retrieved", ()),
        )
        await memory.flush()

    return prediction
```

`_predict` 的数据交接为：

```python
contribution = await memory.retrieve(DecisionContext(
    scope=case.scope,
    query=case.query,
    history=case.history,
    top_k=case.metadata.get("top_k"),
    metadata=case.metadata,
))

answer = await answerer(case, contribution)
prediction = make_prediction(case, answer, contribution)
```

如果 Answerer 返回普通值，Protocol 会用它和当前案例的 `reference` 组装
`Prediction`；如果 Answerer 已经返回 `Prediction`，Protocol 只校验其
`case_id` 是否匹配。`reference` 只用于评分，不能放进回答模型的提示词。

### 2.3 历史导入和反馈边界

`history_ingestion` 来自方法插件的能力声明：

- 普通历史导入通过 `Memory.ingest_history(events)`；
- 声明 `outcome` 策略的方法，在 streaming 中先 `observe` 当前事件，让历史可参与当前 retrieve；
- 当前案例若有 outcome，只能在 retrieve 和 Answerer 完成后调用 `learn`，使结果不会在本题回答前泄漏给 Memory。

`Memory.ingest_history` 不是简单的批量 `observe`。它会根据方法策略决定是否需要
`outcome`，以及是否调用 `learn`、`flush`。因此 Protocol 只选择入口和调用顺序，
不把所有 Memory 方法强制成同一种内部实现。

## 3. `experience` 流程

### 3.1 运行语义

`experience` 将 Benchmark 分为 train 和 test 两个读取边界：

```text
校验 test 案例和 train 案例
  ↓
初始化 Memory
  ↓
读取 train split
  → 导入历史和方法需要的 outcome
  → 执行方法学习及 batch/epoch 边界
  → flush
  ↓
逐个读取 test split
  → 在 Memory.read_only() 中 retrieve
  → Answerer 生成回答
  → 保存 Prediction
  ↓
评分与报告
```

当前 CLI 的 `experience` 使用 `history_import` 模式：导入已有 train history，不会重新执行训练任务。`ExperienceTrainTestProtocol` 在 Python API 中支持可选 `training_agent`；传入时训练案例会由 Agent 实际运行。标准 CLI 的 `qa_spec` 没有传入 training_agent，因此 CLI 不会走这条分支。

### 3.2 Train 阶段接口

`validate_training_inputs(config, plugin, benchmark)` 从
`benchmark.cases(train_split)` 读取训练案例并检查方法是否支持案例中的事件、模态、
`observe`、`learn` 以及 `outcome`。如果提供 `training_agent`，训练案例改由 Agent
protocol 校验和执行。

`prime(memory, benchmark)` 的 history-import 路径大致如下：

```python
train_cases = tuple(benchmark.cases(train_split))
epochs = memory.plugin.learning_schedule(memory.config.method).get("epochs", 1)

for epoch in range(epochs):
    await memory.learning_boundary("epoch_start", current_epoch=epoch, total_epochs=epochs)

    for case in train_cases:
        if training_agent is not None:
            await training_agent.run_case(memory, case)
        else:
            outcome = case.outcome or outcome_from_metadata(
                scope=case.scope,
                success=case.metadata.get("success"),
                reward=case.metadata.get("reward"),
                feedback=case.metadata.get("feedback"),
            )
            await memory.ingest_history(case.history, outcome=outcome)

    await memory.learning_boundary("epoch_end")
    await memory.flush()
```

`Memory.ingest_history(events, outcome)` 会根据方法声明逐条写入历史，并在需要时
调用 `learn` 和 `flush`。`prime` 返回后 Runner 还会统一调用一次 `memory.flush()`。

### 3.3 Test 阶段接口与只读边界

测试案例由外层 `MatrixRunner` 按 `split` 读取。`ExperienceTrainTestProtocol.run_case` 对每个测试案例执行：

```python
async def run_case(memory, case):
    if reset_between_cases:
        raise ValueError("experience test cannot reset learned memory")

    async with memory.read_only():
        return await predict(memory, case)
```

`predict` 仍会构造包含 test `history` 的 DecisionContext，并执行 retrieve 和 Answerer；它不会调用 `observe`、`ingest_history`、`learn` 或 `flush`。测试案例的 outcome 不用于训练。

`read_only()` 的具体保护方式由 Memory 方法声明决定；如果方法不支持只读评测，
测试会直接失败。当前实现对每个 test case 单独进入只读上下文。

### 3.4 两种流程的核心区别

| 项目 | `streaming` | `experience` |
| --- | --- | --- |
| train split | 不需要 | 需要 |
| history 导入 | 每题回答前导入当前案例 history；重复事件可去重 | prime 阶段导入 train history |
| outcome 使用 | 当前案例回答后可 learn | train 导入时交给 Memory；test outcome 忽略 |
| test Memory 写入 | 可能写入每题 history | 不写入；retrieve 在只读上下文执行 |
| epoch 边界 | run 开始和结束各一次 | train 阶段每个 epoch 开始和结束 |
| 默认 CLI 是否运行训练 Agent | 否 | 否；CLI 导入 history |

实现时只需保持四个边界：`BenchmarkAdapter` 负责产生案例，`Protocol` 负责调用顺序，
`Memory` 负责统一的记忆能力入口，`Evaluator` 负责评分。具体 Memory 算法、存储和
LLM 提示词不应写进 Protocol。