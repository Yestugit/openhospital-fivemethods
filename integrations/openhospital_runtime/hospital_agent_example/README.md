# Hospital Agent Baseline Example

这是一个虚拟诊疗比赛的医生 Agent 示例项目，展示如何基于 `hospital-agent-sdk` 开发、训练、测试和部署自己的医生 Agent。

## 目录说明

```text
baseline_example/
├── agent/
│   ├── __init__.py
│   ├── agent.py          # 需要修改：医生 Agent 主逻辑，包含 train/test 和诊疗策略
│   ├── prompt.py         # 需要修改：Prompt 模板和输出格式约束
│   └── memory.py         # 需要修改：训练反思、病例经验、检索记忆等
├── data/
│   ├── memory_data/
│   │   └── memory.md     # 可自定义：baseline 默认的本地文件记忆
│   └── ref_data/
│       ├── departments.json
│       ├── diseases_catalog.json
│       └── examinations_catalog.json
├── config.yaml           # 可配置：训练患者数量、输出目录、memory 路径等
├── train.py              # 可选修改：本地训练入口，默认可直接使用
├── test.py               # 可选修改：本地测试和批量评估示例，默认可直接使用
├── requirements.txt      # 必须维护：新增第三方库需要写在这里
├── Dockerfile            # 一般不需要修改：部署入口，默认启动 python3 -m agent.agent
└── README.md
```

参赛时需要重点修改 `agent/agent.py`、`agent/prompt.py`、`agent/memory.py`：`agent.py` 负责诊疗流程和 action 调用策略，`prompt.py` 负责模型输入和输出格式约束，`memory.py` 负责训练反思、病例经验、检索记忆等记忆逻辑。`config.yaml` 可以按需配置训练患者数量、随机种子、输出目录、记忆路径等参数。

注意：

- `data/memory_data/` 可以根据你选用的记忆存储形式自定义。baseline 默认使用本地 `memory.md`，你也可以改成 JSON、SQLite、向量数据库或其他形式。
- 记忆的存储形式不限定；如果使用数据库或对象存储等外部服务，需要使用可直接访问的外部存储服务。
- 部署环境不能依赖 `docker-compose` 拉起额外服务，因此不要把数据库、向量库等作为同一个提交中的 compose 依赖。
- `data/ref_data/` 是标准科室、疾病和检查名称，不要修改；提交结果中的诊断和检查名称必须使用这里提供的标准名称。
- `Dockerfile` 一般不需要修改，以免影响平台部署和评测，默认已经包含部署入口和启动命令。
- `train.py` 是本地训练入口，可以按需修改；`test.py` 展示了启动 Agent 服务、调用 `/test`、批量评估结果的完整本地测试流程。

## 如何开发医生 Agent

主要修改 `agent/agent.py` 中的 `MyDoctorAgent`。参赛者需要实现或调整 `train` 和 `test` 流程，并通过 `self.actions` 调用比赛能力：

- `ask_patient`：询问患者。
- `order_examination`：申请检查。
- `prescribe_treatment`：提交诊断和治疗方案。
- `evaluation`：训练阶段获取单个病例的评测结果。
- `batch_evaluation`：批量评估 `final_results.jsonl` 中的病例结果。

如果要调整模型输入，可以修改 `agent/prompt.py`；如果要保存训练经验、病例总结或向量检索结果，可以修改或替换 `agent/memory.py`。

## 配置说明

配置分为环境变量和 `config.yaml` 两部分。本地研究只需医院服务地址和医生模型密钥；本地训练、测试、输出目录和记忆策略可通过 `config.yaml` 配置。

### 环境变量

- `SERVICE_BASE_URL`：本地医院服务地址，默认 `http://127.0.0.1:8001`。
- `MODEL_API_KEY`：医生 Agent 和患者模拟调用模型的密钥；本地运行需要自行提供。

本地运行会自动读取医院目录下的 `.env.local`；该文件已被 Git 忽略。需要临时切换到测试服务时，可以在当前终端设置 `SERVICE_BASE_URL=http://127.0.0.1:8002` 覆盖默认训练地址。

在医院目录使用 `python run_openhospital_local.py --dataset train` 独立启动本地医院服务。该启动方式只监听 `127.0.0.1:8001`，不需要比赛账号或 PostgreSQL；病例和检查数据来自仓库内的 `hospital_service`。正式实验请固定患者模型、医生模型和数据划分。

### `config.yaml`

#### `output_dir`

- `output_dir`：训练和测试运行产物的输出目录。默认是 `outputs`，SDK 会在其中生成 `train/`、`test/`、事件日志和最终结果。

#### `train`

- `train.selection`：本地训练患者选取方式，支持 `random`、`forward`、`reverse`。`random` 按随机种子抽样，`forward` 按服务返回顺序从前往后取，`reverse` 按服务返回顺序从后往前取。
- `train.patient_count`：本地训练使用的患者数量；为空时使用服务返回的全部患者。
- `train.random_seed`：`train.selection: random` 时使用的随机种子。
- `train.patient_ids`：指定训练患者 ID 列表；非空时优先使用该列表，忽略 `selection`、`patient_count` 和 `random_seed`。

#### `test`

- `test.selection`：本地 `/test` 调用的患者选取方式，支持 `random`、`forward`、`reverse`。含义与 `train.selection` 相同。
- `test.patient_count`：本地 `/test` 调用使用的患者数量；建议保持较小，便于快速验证。
- `test.random_seed`：`test.selection: random` 时使用的随机种子。
- `test.patient_ids`：指定本地测试患者 ID 列表；非空时优先使用该列表，忽略 `selection`、`patient_count` 和 `random_seed`。

#### 自定义配置

`config.yaml` 也可以添加自定义配置，医生 Agent 代码可以通过 `self.config` 读取。例如：

- `memory`：baseline 示例使用 Markdown 文件作为记忆存储，并在 `agent/memory.py` 中读取 `memory.md_path`、`memory.max_notes`、`memory.max_note_chars`；如果你选择数据库、向量库、对象存储或其他记忆方案，可以根据自己的 `memory.py` 逻辑改造这一段配置。
- `log_llm_prompts`：是否记录每次大模型调用的 prompt 和响应。默认关闭；设为 `true` 后会在每次运行目录下生成 `llm_prompts/`，用于调试模型输入输出。

## 医生可用 Actions

所有 action 都通过 `self.actions.<action_name>(...)` 调用，通常在 `async def train(...)` 或 `async def test(...)` 中使用，需要 `await`。

### `ask_patient`

向患者提问，返回患者回复文本。适合收集主诉、现病史、既往史、用药史、过敏史、家族史等信息。

输入：

```python
answer = await self.actions.ask_patient(
    patient_id=patient_id,
    input_data={
        "question": "请描述这次最主要的不适、开始时间和伴随症状。",
        "chat_history": [
            {"from": "doctor", "text": "之前的问题"},
            {"from": "patient", "text": "之前的回答"},
        ],
    },
)
```

参数说明：

- `patient_id: str`：当前病例 ID，由 SDK 在 `train(patient_id)` / `test(patient_id)` 中传入。
- `input_data: dict`：发给患者服务的 JSON 对象。baseline 使用 `question` 和 `chat_history`，也可以加入自己的上下文字段。

输出：

```python
"患者回答文本"
```

返回值是 `str`。SDK 会自动记录问诊轮数，最终 `prescribe_treatment` 的返回结果中会包含 `conversation_rounds`。

### `order_examination`

申请检查，返回检查结果。检查名称应尽量使用 `data/ref_data/examinations_catalog.json` 中的标准名称。

输入：

```python
exam_response = await self.actions.order_examination(
    patient_id=patient_id,
    items=["血常规", "尿常规"],
    reason="鉴别感染、贫血和泌尿系统异常。",
)
```

参数说明：

- `patient_id: str`：当前病例 ID。
- `items: Iterable[str]`：要申请的检查名称列表。
- `reason: str`：申请检查的原因，可为空字符串。

输出：

```json
{
  "results": {
    "血常规": {
      "result": {
        "血红蛋白": "118 g/L［参考值：140-180 g/L］",
        "血白细胞计数": "13.2 x10^9/L［参考值：4.5-11.0］"
      },
      "status": "abnormal",
      "abnormal_indicators": []
    },
    "尿常规": {
      "result": {
        "尿蛋白": "阴性",
        "尿红细胞": "0-3 个细胞/HPF"
      },
      "status": "normal",
      "abnormal_indicators": []
    },
    "不存在的检查": {
      "result": "无效检查",
      "status": "invalid",
      "abnormal_indicators": []
    }
  }
}
```

字段说明：

- `results`：检查结果字典，key 是请求的检查名称。
- `result`：检查结果。常见形式是“指标名称 -> 指标值”的字典；如果该检查无效，则返回 `"无效检查"`；如果该检查只适用于特殊场景，也可能返回一段补充描述文本。
- `status`：检查状态。`normal` 表示正常，`abnormal` 表示异常，`invalid` 表示无效。
- `abnormal_indicators`：异常指标名称列表。

### `prescribe_treatment`

提交最终诊断和治疗方案，并结束当前病例。测试阶段每个病例必须最终调用这个 action，返回值会写入 `final_results.jsonl` 并作为提交结果。

输入：

```python
final_result = await self.actions.prescribe_treatment(
    patient_id=patient_id,
    diagnosis=["肺炎"],
    treatment_plan="建议抗感染治疗，结合病情补液、退热、止咳，并监测生命体征和复查相关指标。",
    reasoning="结合问诊和检查结果，诊断与治疗方案匹配。",
)
```

参数说明：

- `patient_id: str`：当前病例 ID。
- `diagnosis: Any`：最终诊断，建议使用标准疾病名称列表。
- `treatment_plan: str`：治疗方案文本。
- `reasoning: str`：诊疗依据，可为空字符串。

输出：

```json
{
  "patient_id": "patient_xxx",
  "team_id": "your-team-id",
  "diagnosis": ["肺炎"],
  "treatment_plan": "治疗方案文本",
  "reasoning": "诊疗依据文本",
  "ordered_examinations": ["血常规", "胸部CT"],
  "conversation_rounds": 3,
  "finished": true
}
```

字段说明：

- `diagnosis`、`treatment_plan`、`reasoning`：你提交的最终诊疗结果。
- `ordered_examinations`：当前病例中已成功申请的检查，SDK 会去重并保持顺序。
- `conversation_rounds`：当前病例调用 `ask_patient` 的次数。
- `finished`：是否完成病例；调用该 action 后固定为 `true`。

### `evaluation`

训练阶段用于评估单个已完成病例，通常在 `prescribe_treatment` 之后调用，用于反思和改进策略。测试阶段不要依赖该 action。

输入：

```python
report = await self.actions.evaluation(
    patient_id=patient_id,
    final_result=final_result,
)
```

参数说明：

- `patient_id: str`：当前病例 ID。
- `final_result: dict | None`：`prescribe_treatment` 的返回结果。如果不传，SDK 会尝试读取当前病例最近一次最终结果。

输出：

```json
{
  "patientId": "patient_xxx",
  "status": "evaluated",
  "diagnosisAccuracy": 1.0,
  "examinationPrecision": 0.8,
  "treatmentOverallScore": 0.75,
  "treatmentSafety": 1.0,
  "treatmentEffectivenessAlignment": 0.8,
  "treatmentPersonalization": 0.7,
  "diagnosisDetail": {
    "submitted": ["肺炎"],
    "expected": ["肺炎"],
    "matched": ["肺炎"],
    "accuracy": 1.0,
    "status": "correct"
  },
  "examinationDetail": {
    "ordered": ["血常规", "胸部CT"],
    "expected": ["血常规", "胸部CT"],
    "matched": ["血常规", "胸部CT"],
    "precision": 1.0,
    "coverage": 1.0
  },
  "treatmentDetail": {
    "submitted": "你提交的治疗方案",
    "reference": "参考治疗方案",
    "overallScore": 0.75,
    "safety": 1.0,
    "effectivenessAlignment": 0.8,
    "personalization": 0.7,
    "reasoning": "评测模型给出的解释"
  },
  "ground_truth": {}
}
```

评测报告字段可能随比赛服务调整而变化，训练代码应优先读取自己需要的字段，并对缺失字段做好兜底。

### `batch_evaluation`

批量评估一个 `final_results.jsonl` 文件，或包含该文件的输出目录。适合本地 `/test` 运行结束后，对整批测试病例做统一评估。
传入目录时，SDK 会自动读取该目录下的 `final_results.jsonl`；传入文件路径时，SDK 会直接读取该 JSONL 文件。

输入：

```python
report = await self.actions.batch_evaluation("outputs/test/test_xxx")
```

也可以传入具体文件：

```python
report = await self.actions.batch_evaluation("outputs/test/test_xxx/final_results.jsonl")
```

输出：

```json
{
  "diagnosis_accuracy": 0.8,
  "examination_precision": 0.7,
  "treatment_overall_score": 0.75,
  "treatment_safety": 0.9,
  "treatment_effectiveness_alignment": 0.7,
  "treatment_personalization": 0.65,
  "counts": {
    "final_results": 2,
    "evaluated_patients": 2
  },
  "treatment_details": [
    {
      "patient_id": "Patient_00001",
      "overall_score": 0.75,
      "safety": 0.9,
      "effectiveness_alignment": 0.7,
      "personalization": 0.65,
      "reasoning": "评测模型给出的解释"
    }
  ],
  "submitted_at": "2026-05-22T10:00:00+08:00"
}
```

其中 `diagnosis_accuracy`、`examination_precision`、`treatment_overall_score` 是三个主要评估分项；`treatment_safety`、`treatment_effectiveness_alignment`、`treatment_personalization` 分别表示治疗方案在安全性、有效性、个性化上的表现；`counts` 统计提交和评估的病例数量；`treatment_details` 是每个病例的治疗方案评估明细。评估报告会同时写入同目录下的 `final_results_eval_report.json`。

## Quick Start

安装依赖：

```bash
cd baseline_example
pip install -r requirements.txt
```

配置本地训练环境变量：

```bash
export SERVICE_BASE_URL=http://127.0.0.1:8001
export MODEL_API_KEY=<your-model-api-key>
```

运行本地训练：

```bash
python train.py
```

训练患者可以在 `config.yaml` 的 `train` 中配置。

## 启动测试服务

本地启动 Agent 测试服务：

```bash
python -m agent.agent
```

服务默认监听 `0.0.0.0:7860`，测试接口为 `POST /test`。本地调用示例：

```bash
curl -X POST http://127.0.0.1:7860/test
```

本地测试直接访问 `SERVICE_BASE_URL` 指向的医院服务，医生 Agent 使用 `MODEL_API_KEY` 调用模型。服务以 `train` 数据集启动时，这里测试的仍是训练病例；需要独立测试集时，请以对应 `--dataset` 启动另一个服务并切换地址，避免训练阶段看到测试参考答案。

也可以直接运行完整测试示例：

```bash
python test.py
```

该脚本会自动启动 Agent 服务，调用 `POST /test` 生成 `final_results.jsonl`，再调用 `batch_evaluation` 进行批量评估并输出测试和评估结果。

## 部署

部署到 ModelScope Studio 创空间时，提交最新代码后，在空间页面点击上线，等待空间构建状态变为“运行中”即可。

部署后需要保证服务监听 `7860` 端口，并暴露 `POST /test` 接口。默认通过 `Dockerfile` 启动 `python3 -m agent.agent`，会监听 `0.0.0.0:7860`；一般不需要修改部署入口。提交前建议先在本地运行 `python train.py` 和 `python -m agent.agent`，确认训练流程和测试服务都能正常工作。

本地研究无需配置 ModelScope 比赛令牌；使用远程比赛服务时仍须遵守官方认证流程。
