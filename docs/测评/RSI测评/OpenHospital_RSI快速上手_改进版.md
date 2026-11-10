# OpenHospital RSI 测评快速上手（改进版）

更新：2026-11-01。**统一 Python 测评入口、医院服务管理和运行恢复已实现。**——11.01

使用 `ExperimentConfig` 配置实验、`OpenHospitalServices` 管理本地服务、`ExperimentRunner` 检查和执行；`ExperimentConfig.load(运行目录)` 读取原配置用于恢复。

本文面向首次使用 RiseMem 的用户，以 OpenHospital 为例，介绍记忆驱动的 RSI 测评原理，以及配置、运行和查看结果的完整流程。默认使用内置医生与医院环境，无需自行开发 Agent。

OpenHospital 提供病人、检查、诊疗与临床评分环境。RiseMem 让医生 Agent 在训练病例中积累记忆，再在固定测试病例上冻结评测，观察诊疗能力如何随经历积累变化。

```text
准备实验配置 → 预检病例计划 → 一次启动所需服务 → 执行实验 → 查看结果
```

## 1. 测什么，选择哪种实验

**测评目标：观察 Agent 的任务表现能否随记忆和经验积累而提高。**训练时，Agent 执行任务，Memory 吸收经历与反馈；到达检查点后，冻结记忆，在固定测试集上评测，形成学习曲线。Agent 模型参数保持不变，Memory 按自身机制更新。

| 组成 | 负责什么 | 本文使用的实现 |
|---|---|---|
| Memory | 存储、检索、学习与按需演化 | 用户选择的记忆方法 |
| Agent | 使用记忆并执行任务 | 内置医生 |
| 环境与评分 | 提供任务交互及评价 | OpenHospital 病人、检查与临床评分 |
| `ExperimentRunner` | 调度训练、检查点评测和结果保存 | 框架公共测评入口 |

下面选择医院病例的训练安排。

| 病例流 | 训练病例如何安排 | 观察内容 |
|---|---|---|
| `repeated` | 同一科室的一组病例，在多个阶段重复诊疗 | 重复经历后的表现变化 |
| `novel_same_department` | 同一科室，每阶段使用新病例 | 同领域新经历的积累效果 |
| `novel_departments` | 每阶段使用一个新科室的病例 | 跨科室积累与迁移表现 |

一次运行选择一种。`repeated`、`novel_same_department` 提供一个训练科室；`novel_departments` 提供与阶段数相同的不同科室。原 block 统一称为“阶段”。

| 数据划分 | 用途 |
|---|---|
| `train` | 执行诊疗，提供经历、评分和反思，允许学习 |
| `primary_test` | 固定测试集，在初始位置和各检查点重复冻结评测 |
| `advanced_test` | 在训练结束时额外冻结评测；数量设为 `0` 可关闭 |

训练与测试患者互不重叠。检查点横轴是**累计完成的训练诊疗次数**；重复诊疗同一患者也计一次。固定 `primary_test` 用于学习曲线；`advanced_test` 的难度由病例选择决定。测试反馈不提交记忆学习。

## 2. 安装与模型准备

以下命令在包含 `pyproject.toml` 的项目根目录执行，使用 PowerShell。医院源码、病例数据和科室名单已随框架提供。

```powershell
python -m pip install -e ".[yaml,openai,openhospital]"
python -m pip install -r src/mematlas/integrations/openhospital_runtime/requirements.txt
python -m pip install "qdrant-client>=1.12"
```

最后一行用于下文的 Mem0 示例。其他方法按自己的依赖要求安装。

在 `src/mematlas/integrations/openhospital_runtime/.env.local` 中填写：

```dotenv
MODEL_API_KEY=你的DashScope密钥
```

医院服务和医生启动时会读取此文件。当前内置医生默认使用 DashScope 的 `qwen3.5-flash`；医院病人模型也默认使用它。Memory 的模型单独配置于第 3 节。

## 3. 选择并配置 Memory

查看方法要求：

```powershell
python -m mematlas methods describe mem0
```

新建 `configs/mem0-oph.json`：

```json
{
  "method": {
    "method_id": "mem0",
    "params": {"infer": true, "hybrid_search": false}
  },
  "providers": {
    "llm": {
      "model": "qwen3.5-flash",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key_env": "MODEL_API_KEY"
    },
    "embedding": {
      "model": "text-embedding-v4",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key_env": "MODEL_API_KEY",
      "dimensions": 1024
    }
  },
  "scope": {"user_id": "oph-doctor", "agent_id": "doctor"},
  "runtime": {"artifact_dir": "outputs/oph-memory"}
}
```

此配置使用 LLM 提取与向量检索，关闭 Mem0 混合检索以减少首次安装依赖；正式比较时固定并记录方法变体。Embedding 维度须与模型一致。

`scope` 表示该医生共享的记忆归属，训练和测试使用相同归属。患者编号由医院层记录，跨病例可复用诊疗经验。

所选方法需要支持检索、经历学习、保存恢复和冻结测试，并接受医院生成的轨迹及反馈；缺少条件时预检报错。比较多个方法时分别使用独立输出目录。

## 4. 选择规模并预检病例计划

读取第 3 节的 Memory 配置，创建实验配置：

```python
from mematlas import MemAtlasConfig, ExperimentConfig, ExperimentRunner

experiment = ExperimentConfig(
    methods=(MemAtlasConfig.load("configs/mem0-oph.json"),),
    benchmark={
        "kind": "openhospital",
        "condition": "repeated",
        "departments": ["心内科"],
        "evaluation_departments": ["心内科"],
        "cases_per_block": 2,          # 每阶段诊疗次数
        "blocks": 2,                   # 阶段数
        "heldout_per_department": 1,   # 每科室固定测试病例数
        "advanced_per_department": 1,  # 最终额外测试，0 表示关闭
    },
    protocol="experience_learning_curve",
    seed=0,
    output_dir="outputs/oph-mem0",
)
```

医院参数放在 `benchmark` 中。默认使用内置医生配置，需要替换时补充 `benchmark["doctor_config"]`。模型参数保留在各自配置中；科室名称见[科室病例情况](openhospital各科室病例情况.md)。

默认训练前及每阶段结束后测试；需要自定义时，在 `ExperimentConfig(...)` 中填写 `checkpoints=(0, 4)`，数值表示累计训练诊疗次数。

按需提前检查配置与病例计划，不调用模型：

```python
runner = ExperimentRunner()
report = runner.check(experiment)
print(report)
```

报告展示是否可运行、缺失条件、不同患者数、累计诊疗次数、测试规模和检查点。`evaluate(...)` 执行时也会检查。上例的病例计划为：

```text
训练：2 个不同患者，2 个阶段，共 4 次诊疗
测试：固定病例 1 个，最终额外病例 1 个
检查点：0、2、4
```

### 4.1 RSI 测评参考方案——11.01

建议采用 **40 例 × 5 阶段**，保留五个学习阶段，适度减少训练量。

| 实验 | 训练安排 | 用途 |
|---|---|---|
| 同科新病例（主实验） | 血液科 200 名不同患者，每阶段 40 名 | 观察经验能否改善未见病例表现 |
| 重复病例（补充） | 血液科固定 40 人，重新诊疗 5 轮 | 观察错误修正与经验提炼 |
| 跨科新病例 | 血液→呼吸→消化→内分泌→神经，每科 40 人 | 观察跨科迁移与能力保持 |

1. **共同测试。**上述五科各选 10 例固定测试，在 `0、40、80、120、160、200` 次训练后冻结评测；最终另测每科 10 例独立病例。各方法、三种实验共用测试名单。
2. **对照与指标。**比较无记忆医生、普通记忆方法和目标 RSI 方法；报告临床分项、学习增益、跨科迁移与能力保持，同时记录 Token、耗时和存储。
3. **运行量。**每种实验、每个方法、每个种子共 `200 次训练＋300 次固定测试＋50 次最终测试＝550 次诊疗`。相比 `50×5`，训练量减少 20%，总诊疗量减少约 8.3%。

## 5. 启动服务并运行

### 5.1 启动服务并执行

使用医院服务对象管理本地服务，实验由公共 Runner 执行：

```python
import asyncio
from mematlas.integrations.openhospital import OpenHospitalServices

runner = ExperimentRunner()

async def main():
    async with OpenHospitalServices(experiment):
        return await runner.evaluate(experiment)

results = asyncio.run(main())
```

`OpenHospitalServices` 按配置启动所需服务、等待就绪，并在退出时关闭。端口和数据目录采用默认值；未启用最终额外测试时，不启动对应服务。

### 5.2 训练与测试流程

```text
实验配置 → ExperimentRunner
              ├── 训练：医生执行 → 环境反馈与医生反思 → Memory 学习
              └── 检查点：冻结 Memory → 固定病例测试 → 评分与学习曲线
```

Runner 自动接入内置医生与医院场景，用户无需编写 Agent 或执行回调。

内置医生负责问诊、检查、诊断、治疗与反思；需要时替换[医生配置](../../integrations/openhospital_runtime/hospital_agent_example/config.yaml)。病例选择由实验计划决定。

### 5.3 反馈与学习

默认训练流程：

```text
诊疗完成 → 环境评估 → 医生反思 → 整理经历 → learn
```

环境返回：

| 内容 | 信息 |
|---|---|
| 病例与状态 | 患者编号、评估状态 |
| 临床评分 | 诊断准确率、检查精确率；治疗总分、安全性、有效性匹配度、个性化 |
| 评价明细 | 提交的诊断、检查和治疗方案，以及判定、检查覆盖率和治疗评分说明 |
| 训练参考答案 | 参考诊断、必要检查、匹配结果、参考治疗方案及 `ground_truth`；测试报告隐藏这些信息 |

反思由医生 Agent 生成。框架默认将诊断准确率、检查精确率、治疗总分取平均作为 `reward`，达到 `0.5` 判定 `success`；失败或不完整评分不提交学习。

默认即时交付；研究批量或延迟反馈时，在 `benchmark` 中设置：

| 设置 | 行为 |
|---|---|
| `feedback_timing="batch"` | 按阶段交付；需要时以 `feedback_every=N` 指定诊疗间隔 |
| `feedback_timing="delayed", feedback_delay=N` | 延迟 N 次训练诊疗后交付 |

评分与反思在每例结束后生成，评分映射采用上述默认值。

分别记录完成诊疗数、已提交学习数、待反馈数。测试反馈不提交学习，检查点评测不提前交付未到期反馈。

## 6. 查看结果与恢复

直接查看返回结果，或读取已保存的实验：

```python
result = results[0]
print(result.metrics)
print(result.learning_curve)
print(result.failures)

saved = ExperimentRunner.results("outputs/oph-mem0")[0]
```

| 结果 | 查看内容 |
|---|---|
| 临床表现 | 诊断、检查、治疗与综合临床分（`metrics`） |
| 学习变化 | 固定测试集曲线、相对初始位置的增益（`learning_curve`、`run_metadata["learning_curve_summary"]`） |
| 运行完整性 | 评分覆盖率、失败数、完成／学习／待反馈数（`evaluation_coverage`、`failures`、`run_metadata["progress"]`） |
| 补充分析 | 实际使用的记忆、资源统计和其他曲线指标（`predictions`、`resource_metrics`）；最终指标不包含金额费用 |

缺失评分不补为 `0`。

主要文件：

```text
outputs/oph-mem0/
├── results.json / report.md          统一结果与报告
├── experiment.json                 可重新加载的实验配置
├── frozen_cohort.json               病例、训练顺序、检查点与数据指纹
├── execution.json                   医生、服务与恢复条件
├── encounters/                      训练病例经历和反馈
└── runs/001-mem0/
    ├── curve.json                   训练与测试进度明细
    ├── snapshots/                   Memory 与调度检查点
    └── curves/                      临床／学习曲线 SVG、CSV
```

从原运行读取配置，指定新的输出目录，再通过同一个 Runner 恢复：

```python
from dataclasses import replace

experiment = replace(
    ExperimentConfig.load("outputs/oph-mem0"),
    output_dir="outputs/oph-mem0-resumed",
)

async def resume():
    async with OpenHospitalServices(experiment):
        return await runner.evaluate(
            experiment,
            resume_from="outputs/oph-mem0",
        )

results = asyncio.run(resume())
```

保留原运行记录、方法数据库和医院世界数据。

## 7. 内部如何完成 RSI 测评

```text
实验层：病例计划、多回合调度、反馈交付、检查点评测、结果汇总
                              ↓
场景层：适配 Agent 与环境，执行任务，映射交互、结果和反馈
        ├── OpenHospital
        ├── 问答
        └── 其他场景
                              ↓
通用层：记忆操作、经历关联、学习校验、回合记录、冻结保护与恢复
                              ↓
Memory 方法：原生算法、存储、更新与演化
```

三层适用于整个框架，OpenHospital 是场景层的一种实现。

实验层决定何时交付经历、何时测试；通用层执行学习校验与冻结保护；医院场景负责诊疗和临床数据映射。

代码职责：

| 代码 | 调整 |
|---|---|
| 医院 `bridge.py` | 临床查询、记忆使用关联和经历映射 |
| 医院 `execution.py` | 保留 SDK 执行、评分请求及重试，复用公共记录与冻结能力 |
| 公共 `experience.py`、`experience_memory.py`、`feedback.py` | 回合状态、反馈交付、冻结校验与恢复 |
| 公共和医院 `accounting.py` | 公共部分统一统计，医院部分采集医生与服务用量 |
| 医院 `rsi.py` | 保留医院实验接入和临床曲线汇总 |
| `methods/<方法>/` | 承接方法专用字段及必要步骤 |

`evaluation/scenes.py` 按配置选择场景适配器。新增场景实现相同接入约定并登记，Runner 保持统一流程。

优先复用已有组件，保持一个测评执行入口。演化遵循方法自身机制：能够自动判断时内部触发，需要外部触发时调用 `evolve`；`evolve` 不承担冻结评测的准备职责。

阅读实现时，按 [application.py](../../integrations/openhospital/application.py) → [rsi.py](../../integrations/openhospital/rsi.py) → [execution.py](../../integrations/openhospital/execution.py) → [bridge.py](../../integrations/openhospital/bridge.py) 即可串起主流程；[metrics.py](../../integrations/openhospital/metrics.py) 定义医院汇总指标。

**验证：**已用真实 Qwen、Mem0 和医院服务完成 1 例训练、训练前后各 1 次固定测试，以及保存后的恢复；评分覆盖率为 100%。三种病例流、反馈交付和中断恢复由回归测试覆盖。
