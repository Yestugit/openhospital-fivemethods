

# RiseMem 实验评测快速上手

更新：接入说明与任务归属整理。——10.30

`ExperimentRunner` 用于比较记忆方法、评测已有 Agent，以及组织连续学习与冻结测试。用户提供**方法配置、任务数据、回答模型或 Agent**，框架负责执行、评分和结果记录。

使用前查看所选 Memory 方法与 Benchmark 的参数、可调用接口和使用条件。各方法可补充特有接口；更换方法或 Benchmark 时，按其要求调整配置、调用流程与 Agent 适配。

## 1. 安装与选择流程

在包含 `pyproject.toml` 的项目根目录执行：

```bash
python -m pip install -e ".[openai,benchmarks]"
python -m mematlas methods list
python -m mematlas methods describe baseline_bm25
```

| 评测需求 | 使用方式 |
|---|---|
| 每道题导入历史，再检索并回答 | `protocol="streaming_memory"` |
| 先用 train 经历更新记忆，再冻结测试 | `protocol="experience_train_test"` |
| 运行已有的多步或工具 Agent | `protocol="interactive_agent"` |
| 在多个训练检查点观察学习曲线 | `runner.evaluate_experience(...)` |

## 2. 可用接口

先创建运行器：

```python
from mematlas import ExperimentRunner

runner = ExperimentRunner()
```

| 接口 | 用途与返回 |
|---|---|
| `runner.check(experiment, answerer=..., agent=...)` | 检查方法配置、依赖和流程兼容性，返回检查报告 |
| `await runner.evaluate(experiment, answerer=..., agent=...)` | 执行配置中的实验，返回 `list[EvaluationResult]` |
| `await runner.evaluate_qa(configs, benchmark, llm=..., model=..., ...)` | 使用框架提供的问答流程比较方法 |
| `await runner.evaluate_experience(plan, memory, output_dir=..., train_batch=..., test_batch=...)` | 执行连续学习与检查点测试 |
| `await runner.resume_experience(source, plan, memory, ...)` | 从原实验目录的检查点恢复连续学习 |
| `runner.results(path)` | 读取结果文件或结果目录 |

普通评测可通过 `benchmark=` 传入已有数据适配器，通过 `services=` 注入方法所需服务，通过 `judge=` 提供 Benchmark 评分器。

## 3. 准备数据与实验配置

内置支持 LoCoMo、LongMemEval 和自定义 JSON/JSONL；数据文件由用户准备。

自定义数据示例，保存为 `cases.json`：

```json
[
  {
    "case_id": "case-001",
    "split": "test",
    "scope": {"user_id": "alice", "agent_id": "assistant"},
    "history": [
      {
        "event_id": "message-001",
        "role": "user",
        "text": "Helios 的服务端口是 8443。"
      }
    ],
    "query": "Helios 的服务端口是多少？只回答数字。",
    "reference": "8443"
  }
]
```

| 数据字段 | 用途 |
|---|---|
| `case_id`、`scope` | 案例身份与记忆归属 |
| `history` | 已有对话或任务经历 |
| `query` | 当前任务输入 |
| `reference` | 评分参考答案 |
| `outcome` | 已有经历的真实结果反馈，按方法要求提供 |
| `split` | 数据划分，例如 `train`、`test` |

实验配置由 `ExperimentConfig` 表达：

| 配置字段 | 用途 |
|---|---|
| `methods` | 待比较的 Memory 配置列表 |
| `benchmark` | 数据类型、路径和读取选项 |
| `protocol` | 评测流程 |
| `split`、`train_split` | 测试集与训练集名称 |
| `metrics` | 通用指标名称；省略时使用默认指标 |
| `output_dir` | 报告保存目录 |

## 4. 运行一次方法比较

下面比较无记忆与 BM25，使用同一个回答模型。先设置 `OPENAI_API_KEY`，再替换模型名称和服务地址。

```python
import asyncio

from mematlas import ExperimentConfig, ExperimentRunner
from mematlas.evaluation.qa import memory_qa_answerer
from mematlas.providers import OpenAICompatibleLLM


async def main():
    llm = OpenAICompatibleLLM(
        model="你的模型名称",
        base_url="你的 OpenAI 兼容服务地址",
        api_key_env="OPENAI_API_KEY",
    )
    answerer = memory_qa_answerer(llm, model=llm.model)

    experiment = ExperimentConfig.from_dict({
        "methods": [
            {"method": {"method_id": "baseline_none"}},
            {"method": {"method_id": "baseline_bm25"}},
        ],
        "benchmark": {"kind": "custom", "path": "cases.json"},
        "protocol": "streaming_memory",
        "metrics": ["common.exact_match", "common.token_f1"],
        "output_dir": "outputs/comparison",
    })

    runner = ExperimentRunner()
    report = runner.check(experiment, answerer=answerer)
    if not report["ready"]:
        raise RuntimeError(report)

    results = await runner.evaluate(experiment, answerer=answerer)
    for result in results:
        print(result.method_id, result.metrics, result.failures)


asyncio.run(main())
```

更换方法时按其清单调整 `methods`、服务与接入流程；更换 Benchmark 时同时核对数据格式、读取参数、任务执行、评分和可用协议。现有接口兼容时可以复用接入代码。

## 5. 接入自己的回答逻辑或 Agent

| 接入形式 | 用户提供的接口 | 框架负责 |
|---|---|---|
| 自定义回答逻辑 | `answerer(case, contribution) → str 或 Prediction` | 导入历史、检索、评分与统计 |
| 完整 Agent | `agent(case, memory) → AgentTrace` | 回合边界、轨迹核对、结果提交与评分 |
| 自定义数据集 | 实现 `BenchmarkAdapter.cases(split)` | 调用所选评测流程 |
| 自定义领域评分 | 实现 `BenchmarkAdapter.native_score(predictions)` | 汇总领域指标与通用指标 |

回调支持同步或异步形式。完整 Agent 在内部按需检索、使用工具并上报真实轨迹；典型接入由评测协议管理回合开始和结束。

```python
# experiment.protocol 设置为 "interactive_agent"
report = runner.check(experiment, agent=my_agent)
results = await runner.evaluate(experiment, agent=my_agent)
```

自定义回调可以访问完整案例，回答过程中应避免使用 `case.reference`。

## 6. 连续学习与冻结测试

先用 `ExperiencePlan` 指定训练案例、测试案例和检查点，再提供场景执行回调：

```python
from mematlas import Memory
from mematlas.evaluation import ExperiencePlan

plan = ExperiencePlan(
    train_ids=("train-1", "train-2"),
    primary_test_ids=("test-1",),
    advanced_test_ids=(),
    batch_count=2,
    checkpoints=(0, 1, 2),
)

result = await runner.evaluate_experience(
    plan,
    Memory.from_dict(memory_config),
    output_dir="outputs/experience",
    train_batch=train_batch,
    test_batch=test_batch,
)
```

上面的调用放在异步函数中；`memory_config` 和两个回调由用户提供：

| 回调 | 责任 |
|---|---|
| `train_batch(ids, experience, position) → BatchResult` | 执行指定训练任务、提交真实经历与反馈，返回实际处理状态 |
| `test_batch(ids, experience, position, split) → dict` | 执行测试并返回评分统计 |

回调通过 `experience.core_memory` 使用 Memory。运行器负责检查点、测试期间冻结和恢复；场景回调负责实际 Agent 与环境交互。

恢复时使用：

```python
result = await runner.resume_experience(
    "outputs/experience",
    plan,
    Memory.from_dict(memory_config),
    output_dir="outputs/experience-resumed",
    train_batch=train_batch,
    test_batch=test_batch,
)
```

## 7. 查看结果与使用约定

普通比较返回的每个 `EvaluationResult` 包含：

| 字段 | 内容 |
|---|---|
| `predictions` | 逐题输出 |
| `metrics` | 通用指标与 Benchmark 原生评分 |
| `resource_metrics` | 操作次数、时间、Token、存储及费用等 |
| `failures` | 失败阶段、案例与错误 |
| `run_metadata` | 方法配置和运行信息 |

普通比较输出 `results.json`、`report.md` 及逐题记录；连续学习输出检查点、`curve.json` 和运行结果。费用需要配置模型价格，未提供时保留未知值。

1. 比较方法时固定数据、顺序、Agent、模型与评分规则。
2. `experience_train_test` 默认导入已有训练经历；需要重新执行训练 Agent 时，通过 `agent=` 提供。
3. 训练与测试需要复用记忆时保持兼容的 `scope`；学习曲线的训练和测试案例 ID 应分离。
4. 检查 `failures` 后再解释汇总分数。需要标签、奖励或原生更新边界的方法，应满足其使用要求。
5. 公共接口表不表示每个方法都支持全部能力。运行前按所选方法和 Benchmark 的要求检查，按报错补齐缺失项；当前检查尚未覆盖全部条件。

完整使用清单待补齐，任务分别见[方法集成](../方法集成/项目任务与进展.md)和[测评](../测评/项目任务与进展.md)。
