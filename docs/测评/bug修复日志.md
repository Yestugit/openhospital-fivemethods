**审查结论：主要测评链路能够跑通，但当前仍有会影响分数、冻结验证和来源统计的问题，建议先修复这些问题，再开展正式的方法对比实验。**

本轮同时做了源码审查、独立交叉审查和实际运行；没有修改源码或提交 Git。

实际验证的范围：

| 范围 | 结果 |
|---|---|
| Mem0、MemoryOS 的已有历史、训练后测试、多检查点测评 | 6 条链路通过 |
| Mem0 子进程与 HTTP 执行下的三种实验安排 | 6 条链路通过 |
| 真实 Qwen＋Mem0＋Embedding | 训练、冻结、评分和结果读取通过；学习 2 条记录后，2 道测试题全部答对 |
| OpenHospital | 训练、多个检查点、最终 advanced 测试、评分故障及恢复通过 |
| 回归测试 | 最新评测专项 **50 项通过**；Mem0 **24 项通过**；医院原生评分 **9 项通过** |

MemRL 本机缺少原生依赖，本轮没有完成其真实训练链路。

**确认存在的问题如下，按影响排序：**

| 序号 | 问题 | 实测结果与影响 | 修改方向 |
|---|---|---|---|
| **1，优先** | **冻结失败产生的答案，恢复后仍被采用** | 让方法在冻结检索时修改记忆，首次运行正确报错；恢复后却直接使用此前缓存，最终得到 `exact_match=1.0`、`frozen_verified=true`。 | 冻结校验通过后才能确认缓存有效；失败检查点的未验证答案不得直接复用。 |
| **2，优先** | **更换 Agent 配置后，恢复仍混用旧答案** | 首次用正确答案配置运行，中断后换成同类 Agent 的错误答案配置；恢复只执行剩余题，混合新旧答案，得分 **0.5**。全程使用新配置应为 **0**。 | 恢复检查应包含 Agent 的实际配置；配置变化时拒绝复用旧结果。 |
| **3，优先** | **默认 QA 的拒答格式与 LoCoMo 评分冲突** | 默认提示要求回答 `The information is unavailable.`；LoCoMo 拒答题只接受另外两个固定短语。遵守默认提示的正确拒答得到 **0 分**。 | 对齐默认答题提示与 Benchmark 原生拒答要求。 |
| **4** | **缺少标准答案、也没有输出时，公共指标报满分** | 自定义案例不填 `reference`，Agent 返回空 `MemoryEpisode()`，最终仍是 `completed`，EM、F1 都为 **1.0**。原因是两个 `None` 被转为相同字符串。 | 无标准答案时不计算答案准确性指标；缺失输出应明确记录。 |
| **5** | **用户配置的指标没有完整生效** | 自定义 `task_accuracy` 在普通测评正常；在两种训练测评中，曲线有该值，最终 `result.metrics` 丢失。医院分支还会接受指标配置，却完全不执行它。 | 完整保留已计算指标；医院对支持的指标执行计算，对不支持的指标提前报错。 |
| **6** | **医院测试结果丢失实际使用的记忆来源** | 测试医生实际收到并使用了 **5 条记忆文本**，最终 `Prediction.applied_memory` 仍为空。临床评分正常，记忆来源分析缺数据。 | 在评测侧记录测试中实际使用的贡献，保持 Memory 冻结状态。 |
| **7** | **恢复后的操作统计遗漏恢复前记录** | 源运行完成过一次 `learn`，恢复后的最终结果却显示 `learn_count=0`，`traces` 中也没有此前的学习操作。底层日志仍保留。 | 恢复时载入历史操作记录并去重，使累计统计与完整实验一致。 |
| **8** | **LongMemEval 自定义裁判可能把否定标签判为正确** | `{"correct": "no"}`、`{"label": "false"}` 都被解析成 `True`，因为直接使用了 Python 的 `bool()`。 | 明确解析允许的标签；无法识别时报告评分错误。 |

关键代码位置：

1. **问题 1、2、5**：[案例执行与恢复缓存](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/evaluation/cases.py:177)。
2. **问题 3**：[默认 QA 提示](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/evaluation/qa.py:26)、[LoCoMo 拒答评分](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/benchmarks/locomo.py:71)。
3. **问题 4**：[公共答案指标](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/evaluation/metrics/common.py:69)。
4. **问题 5、6**：[医院结果生成](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/integrations/openhospital/rsi.py:44)、[冻结测试的贡献处理](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/integrations/openhospital/bridge.py:146)。
5. **问题 7**：[历史操作日志恢复](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/evaluation/experience.py:188)。
6. **问题 8**：[裁判标签解析](C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/benchmarks/longmemeval.py:106)。

另外，**LongMemEval 的指标口径需要明确**：当前工作区采用的标准 NDCG 公式与本地官方脚本不同；现有 `judge_task_accuracy` 也是按题目计算的准确率，与官方按题型计算的宏平均有区别。报告实验结果时，应明确这些名称和计算口径。

正常路径中，训练测试划分、固定测试集、多检查点执行、真实反馈校验，以及不统一触发 `evolve` 的安排均按预期工作。**当前最应优先处理的是前 3 项，它们会直接影响评测结论。**

1101
---
代码职责解耦重构

建议围绕 **“一个完整 QA、一个明确的方法扩展位置、一条清楚的实验执行链”**重构，具体修改如下。

| 项目 | 现在是什么 | 准备改成什么 | 为什么 |
|---|---|---|---|
| **1. 完整默认 QA** | 检索在 `cases.py`，提示与模型调用在 `qa.py`，通过回调连接 | `evaluation/qa.py` 提供完整 `DefaultQA`，集中完成**检索、提示组装、模型回答、轨迹与使用来源记录** | 打开一个文件就能看懂一道题如何完成，也便于方法复用 |
| **2. 方法专有接入** | 只能借助用户 `agent` 参数实现；文档约定的 `methods/<方法>/qa.py` 尚未真正接通 | 方法确有需要时提供 `methods/<方法>/qa.py`，在方法插件中声明，由框架选择；未声明的使用 `DefaultQA` | 新方法有固定接入位置，用户无需替框架补写方法适配 |
| **3. 单题执行约定** | 默认 QA 返回 `Prediction`，用户 Agent 返回 `MemoryEpisode`，执行路径不同 | 默认 QA、方法 QA、用户 Agent 统一采用 **`执行器(case, memory) -> MemoryEpisode`**；评测层统一转换为 `Prediction` | 共用单题执行、计时、来源处理和结果转换，减少重复分支 |
| **4. 实验执行收拢** | `_CaseExecution`、`_HistoryRunner`、Protocol 之间多次转交，部分逻辑重复 | 将 `cases.py` 整理为 `evaluation/execution.py`，负责一次方法实验的执行协调；清理重复 Runner 和回调转发 | 调用链直接，能够顺着函数看完整流程 |
| **5. 方法差异移出公共调度** | 公共 `_train_batch()` 中直接判断 `method_id == "memrl"` | MemRL 的专有学习数据处理放进 `methods/memrl/qa.py`；公共执行层调用统一的学习数据准备步骤 | 新增方法时，避免不断向公共调度加入方法分支 |
| **6. 评分职责归位** | `_configure_reader()` 同时创建答题模型和 LongMemEval 裁判 | 答题模型由 QA 模块使用；裁判配置归 Benchmark 评分路径，`Evaluator` 负责组织评分 | 答题与评分分别有明确归属 |

有三个边界需要一起明确：

1. **QA 负责完成单题。**它可以检索、组织模型输入、执行方法必需的操作，返回实际经历和使用来源。
2. **实验层负责安排学习与测试。**历史导入、`learn` 提交、冻结、检查点和恢复留在实验执行中。已有完整训练经历继续直接学习，无需额外生成一遍训练答案。
3. **方法 QA 只补必要差异。**能通过配置、公共字段或方法内部处理解决的，直接使用默认 QA。确需额外代码时，复用默认实现，只覆盖必要步骤，避免复制整套流程。

重构后的调用关系：

```text
ExperimentRunner.evaluate(config)
    │
    ├── 检查配置、加载 Benchmark、选择方法
    ├── 选择 DefaultQA／方法 QA／用户 Agent
    │
    ▼
evaluation/execution.py
    ├── 创建与关闭 Memory
    ├── 按实验安排学习历史／训练经历
    ├── 通过 experience.py 管理检查点、冻结和恢复
    │
    └── 执行一道题：await executor(case, memory)
            │
            ├── evaluation/qa.py：DefaultQA
            ├── methods/<方法>/qa.py：必要的专有实现
            └── 用户 Agent
            │
            ▼
        MemoryEpisode
            │
            ▼
    统一转换 Prediction → Evaluator 评分 → 保存结果
```

**以 MemRL 为例：**它可以复用默认的“检索后回答”，只在自己的 `qa.py` 中补充学习所需的数据整理和训练结束标记。公共执行代码不需要认识 MemRL。

实施时分三步提交：

1. 集中完整 QA，统一单题输入输出。
2. 接通方法 QA，迁出公共层的 MemRL 专有处理，清理重复调用层。
3. 同步现有调用方和测试，验证三种实验安排、医院、冻结与恢复。

**用户侧继续使用 `ExperimentConfig` 和 `ExperimentRunner.evaluate()`，这次重点整理内部职责和方法开发入口。**


重构后：
从 **`await ExperimentRunner().evaluate(config)`** 开始，当前调用链如下。路径均相对于 `src/mematlas/`。

**1. 进入测评，检查配置并创建一次方法实验**

```text
evaluation/experiment.py
ExperimentRunner.evaluate(config)
    │
    ├── _prepare(...)
    │   ├── _dataset(...) → 加载 Benchmark
    │   ├── benchmark.cases(...) → 取得训练／测试案例
    │   ├── validation.check_models(...) → 检查答题与评分模型
    │   ├── check_config(...) → 检查 Memory 配置和依赖
    │   └── validation.validate_cases(...) → 检查数据与方法能力
    │
    └── 对每个 Memory 方法调用：
        evaluation/execution.py
        _EvaluationRun.run(...)
            ├── Memory(...) → 创建本次实验的 Memory
            ├── _qa_type(plugin) → 选择 DefaultQA／方法 QA
            ├── _configure_scoring(...) → 配置 Benchmark 裁判
            └── 按 protocol 进入下面的实验安排
```

**2. 按实验安排，进入学习与测试**

```text
_EvaluationRun.run(...)
    │
    ├── streaming_memory：已有历史测评
    │   └── _run_history()
    │       ├── Memory._setup()
    │       └── 逐个案例：
    │           ├── history_episode(case) → 取得历史经历（有历史时）
    │           ├── _prepare_learning(...) → qa.prepare_episode(...)
    │           ├── Memory.learn(episode)
    │           ├── Memory.flush()
    │           └── execute_case(memory, case) → 执行单题
    │
    └── experience_train_test／experience_learning_curve
        └── _run_training(...)
            └── experience.py：_ExperienceRunner.run()
                ├── _resume_if_requested() → 按需恢复
                │
                ├── execution.py：_train_batch(...)
                │   ├── 取得已有训练经历，或执行用户 Agent
                │   ├── _prepare_learning(...) → qa.prepare_episode(...)
                │   └── Memory.learn(episode)
                │
                └── 到达测试检查点：_run_checkpoint(...)
                    └── async with Memory.read_only():
                        └── execution.py：_test_batch(...)
                            └── execute_case(memory, case)
```

训练后测试在训练完成后执行测试；多检查点实验会在指定训练位置重复执行测试。已有完整训练经历直接提交学习。

**3. 所有实验共用的单题执行链**

```text
evaluation/execution.py
execute_case(memory, case)                    # 开始单题计时
    └── _execute_episode(case, memory)
        │
        ├── 提供了 agent=：
        │   └── await agent(case, memory)
        │
        └── 使用框架 QA：
            └── evaluation/qa.py
                DefaultQA.__call__(case, memory)
                    ├── self.retrieve(case, memory)
                    │   └── runtime/memory.py：Memory.retrieve(...)
                    │       └── Memory._invoke("retrieve", ...)
                    │           └── self._method.retrieve(...)
                    │               └── 具体 Memory 方法的检索实现
                    │
                    ├── self.messages(case, contribution) → 组装提示
                    ├── provider.complete(...) → 模型回答
                    └── 返回 MemoryEpisode
                        包含：问题、回答、事件、实际使用的记忆贡献
        │
        ▼
    返回 MemoryEpisode
    │
    ▼
execute_case(...) → 统一转换为 Prediction，记录单题耗时
```

方法声明专有 QA 时，上面的 `DefaultQA` 由对应子类承担；例如 `MemRLQA` 复用默认答题流程，并通过 `prepare_episode()` 补充训练结束标记。

**4. 评分、保存并返回**

```text
_run_history()／_test_batch() 收集本轮 Predictions
    │
    ▼
evaluation/metrics/common.py
Evaluator.evaluate(...)
    ├── metric.compute(predictions) → 通用／自定义指标
    ├── benchmark.native_score_async(predictions) → Benchmark 原生评分
    └── 汇总来源、耗时、Token、存储等统计
    │
    ▼
_EvaluationRun → 组织 EvaluationResult
    │
    ▼
ExperimentRunner.evaluate(...)
    ├── execution.close() → 关闭 QA 自有模型客户端
    ├── ReportWriter.write(...) → 保存结果
    └── 返回 list[EvaluationResult]
```

**看代码时，先沿着 `experiment.py → execution.py → qa.py → runtime/memory.py → methods/<方法>/method.py` 这条主线阅读即可。**