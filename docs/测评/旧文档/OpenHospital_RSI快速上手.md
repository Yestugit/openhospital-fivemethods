# OpenHospital RSI 测评快速上手

更新：2026-11-01。本文描述当前实现，供使用和后续改进参考。

OpenHospital 提供病人、检查、诊疗与临床评分环境。RiseMem 让医生 Agent 在训练病例中积累记忆，再在固定测试病例上冻结评测，观察诊疗能力如何随经历积累变化。

**当前状态：病例计划预检可运行；正式启动脚本仍有配置残留阻塞，见第 5 节。**本文中的正式运行命令对应现有入口，尚需修复该阻塞后验证。

```text
选择病例流 → 配置 Memory → 预检计划 → 启动医院服务
→ 医生训练与记忆学习 → 检查点冻结测试 → 临床评分与学习曲线
```

## 1. 测什么，选择哪种实验

测评期间保持医生实现、模型和测试病例固定，改变训练经历与 Memory 方法，记录固定测试集上的表现。医生模型参数保持不变，记忆方法按自身机制更新。

| `--condition` | 训练病例如何安排 | 观察内容 |
|---|---|---|
| `repeated` | 同一科室的一组病例，分多个 block 重复诊疗 | 重复经历后的表现变化 |
| `novel_same_department` | 同一科室，每个 block 使用新病例 | 同领域新经历的积累效果 |
| `novel_departments` | 每个 block 使用一个新科室的病例 | 跨科室积累与迁移表现 |

一次运行选择一种。`repeated`、`novel_same_department` 提供一个训练科室；`novel_departments` 提供与 block 数量相同的不同科室。

| 数据划分 | 用途 |
|---|---|
| `train` | 执行诊疗，提供经历、评分和反思，允许学习 |
| `primary_test` | 固定测试集，在初始位置和各检查点重复冻结评测 |
| `advanced_test` | 在训练结束时额外冻结评测；数量设为 `0` 可关闭 |

训练与测试患者互不重叠。检查点横轴是**累计完成的训练诊疗次数**；重复诊疗同一患者也计一次。测试反馈不提交记忆学习。

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

所选方法需要支持检索、经历学习、保存恢复和冻结测试。方法还需接受医院生成的轨迹及反馈；缺少条件时预检查会报错。当前启动一次运行一个方法，比较多个方法时分别使用独立输出目录。

## 4. 选择规模并预检病例计划

先用小规模重复病例实验：2 个不同训练病例，每个诊疗 2 次，共 4 次训练；固定测试 1 例，最终额外测试 1 例。

```powershell
python -m mematlas.integrations.openhospital `
  --condition repeated `
  --departments 心内科 --evaluation-departments 心内科 `
  --cases-per-block 2 --blocks 2 `
  --heldout-per-department 1 --advanced-per-department 1 `
  --eval-every 2 --seed 0 `
  --dry-run --output-dir outputs/oph-plan
```

预期返回 `status="plan_only"`、`train_exposure_count=4`、`checkpoints=[0,2,4]`，并生成 `frozen_cohort.json`。这一步验证病例与计划，不调用模型。

后续按需要调整以下参数：

| 参数 | 用户选择与含义 |
|---|---|
| `--condition repeated/novel_same_department/novel_departments` | 第 1 节的实验安排 |
| `--departments 科室1 [科室2 ...]` | 训练科室，名称见 [科室病例情况](openhospital各科室病例情况.md) |
| `--evaluation-departments 科室1 [科室2 ...]` | 固定测试科室；省略时默认使用 5 个科室 |
| `--cases-per-block N --blocks B` | 每个 block 的诊疗次数与 block 数，总训练次数为 `N × B` |
| `--heldout-per-department N` | 每个测试科室的固定 primary 病例数 |
| `--advanced-per-department N` | 每个测试科室的最终 advanced 病例数，允许 `0` |
| `--eval-every N` | 每完成 N 次训练测试一次；初始位置和最终位置也测试 |
| `--seed N` | 病例抽样与顺序的随机种子 |

省略 `--eval-every` 时，默认在初始位置及每个原始 block 结束后测试。相同种子与测试科室可保持测试患者一致。

## 5. 启动服务并运行

### 5.1 启动三个医院服务

分别在三个终端、同一项目根目录执行，保持运行：

```powershell
python src/mematlas/integrations/openhospital_runtime/run_openhospital_local.py --dataset train
```

```powershell
python src/mematlas/integrations/openhospital_runtime/run_openhospital_local.py --dataset primary_test
```

```powershell
python src/mematlas/integrations/openhospital_runtime/run_openhospital_local.py --dataset advanced_test
```

默认端口依次为 `8001`、`8002`、`8003`，健康检查地址为对应服务的 `/health`。服务保留病例世界与用量日志，默认目录为 `.mematlas/hospital-world/`。

### 5.2 执行小规模实验

在第四个终端执行，使用新的输出目录：

```powershell
python -m mematlas.integrations.openhospital `
  --method-config configs/mem0-oph.json `
  --condition repeated `
  --departments 心内科 --evaluation-departments 心内科 `
  --cases-per-block 2 --blocks 2 `
  --heldout-per-department 1 --advanced-per-department 1 `
  --eval-every 2 --seed 0 `
  --output-dir outputs/oph-mem0
```

内置医生负责问诊、检查、诊断、治疗与反思，无需用户编写 Agent 或批次回调。需要调整医生时，通过 `--doctor-config 路径` 提供 YAML；默认医生配置见 [config.yaml](../../integrations/openhospital_runtime/hospital_agent_example/config.yaml)。医生配置中的病例选择由实验计划覆盖。

**当前启动阻塞：**`application.py` 仍访问已删除的 `runtime.state_mode`，通过 Memory 检查后会抛出 `AttributeError`。不要在配置中补这个字段，当前配置类不接受它。此处需要后续修正并验证正式 CLI 全流程。

### 5.3 按需调整反馈时机

| 参数 | 行为 |
|---|---|
| `--feedback-timing immediate` | 默认：每例完成后提交真实反馈并学习 |
| `--feedback-timing batch --feedback-every N` | 每 N 次训练集中交付反馈；省略间隔时按原始 block 交付 |
| `--feedback-timing delayed --feedback-delay N` | 延迟 N 次训练后交付反馈 |

默认先生成临床评分和反思；增加 `--feedback-generation on_delivery` 可延迟到反馈交付时生成。临床三项核心评分默认取平均值作为 reward，达到 `0.5` 判 success；失败或不完整评分不会作为有效反馈提交学习。

延迟反馈可能在实验结束时仍有待交付项，因此“完成诊疗数”与“已提交学习数”分别记录。冻结测试不会提前交付未到期反馈，也不会统一触发 `evolve`。

## 6. 查看结果与恢复

运行结果通过公共测评入口读取：

```python
from mematlas import ExperimentRunner

result = ExperimentRunner.results("outputs/oph-mem0")[0]
print(result.failures)                           # 先检查运行失败
print(result.metrics)                            # 最后一次 primary 测试评分
print(result.run_metadata["progress"])            # 完成、学习、待反馈数量
print(result.run_metadata["learning_curve_summary"])
for point in result.learning_curve:
    print(point["checkpoint"], point["split"], point.get("diagnosis_accuracy"))
```

| 结果 | 查看内容 |
|---|---|
| `metrics` | 诊断准确率、检查精确率／召回率、治疗评分、综合临床分及分科室结果 |
| `learning_curve` | 各检查点的 primary 结果，以及最终 advanced 结果 |
| `run_metadata["learning_curve_summary"]` | 首末增益、归一化曲线面积、每 100 次训练增益、最大回落 |
| `predictions` | 最后一次 primary 测试的诊疗输出、临床报告和实际使用的记忆贡献 |
| `resource_metrics` | 耗时、Token／请求、内存及存储；最终指标不包含金额费用 |

同时检查 `evaluation_coverage`、失败数和评分缺失数；缺失评分不会自动补为 0。

主要文件：

```text
outputs/oph-mem0/
├── results.json / report.md          统一结果与报告
├── frozen_cohort.json               病例、训练顺序、检查点与数据指纹
├── execution.json                   医生、服务与恢复条件
├── encounters/                      训练病例经历和反馈
└── runs/001-mem0/
    ├── curve.json                   训练与测试进度明细
    ├── snapshots/                   Memory 与调度检查点
    └── curves/                      临床／学习曲线 SVG、CSV
```

**恢复方式：**沿用原运行参数，改用新的 `--output-dir`，增加 `--resume-from outputs/oph-mem0`。保留原输出、方法数据库和医院世界目录，医生、数据、方法及服务条件应保持一致。

**当前恢复限制：**统一运行器把恢复快照放在 `runs/001-<方法>/`，医院启动脚本仍有旧的根目录检查与缓存路径约定。公共调度和医院执行的恢复已做回归验证；正式 CLI 的目录衔接还需核对。

## 7. 内部如何完成 RSI 测评

```text
医院启动脚本：准备病例计划、医生、医院服务和 Memory
        ↓
ExperimentRunner.evaluate(...)
        ↓
_HospitalEvaluation → _HospitalRunner（复用公共 _ExperienceRunner）
        │
        ├── 训练：SDK 执行医生诊疗 → 评分／反思 → 按反馈时机 learn
        └── 检查点：冻结 Memory → 固定病例诊疗 → 临床评分 → 校验状态
        ↓
EvaluationResult → 统一报告与医院学习曲线
```

1. **医院层映射真实经历。**`OpenHospitalMemoryIntegration` 把诊疗轨迹、最终输出、反馈和已用记忆整理为 `MemoryEpisode`；同一患者的不同训练轮次使用不同经历编号。
2. **公共层组织实验。**训练进度决定检查点；冻结测试禁止提交学习，检查记忆状态，支持通过快照恢复。方法内部的学习与演化遵循其自身机制。
3. **评分留在医院环境。**医生通过 SDK 与病人及检查服务交互，医院依据病例参考信息生成临床评分；参考诊断不会直接作为医生的记忆查询输入。

阅读实现时，按 [application.py](../../integrations/openhospital/application.py) → [rsi.py](../../integrations/openhospital/rsi.py) → [execution.py](../../integrations/openhospital/execution.py) → [bridge.py](../../integrations/openhospital/bridge.py) 即可串起主流程；[metrics.py](../../integrations/openhospital/metrics.py) 定义医院汇总指标。
