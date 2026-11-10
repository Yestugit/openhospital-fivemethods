# OpenHospital RSI 测评交接任务书

更新日期：2026-10-26。

**接手任务：核对、修复并跑通 RiseMem 的 OpenHospital RSI 测评流程，确认现有实现能完成多轮诊疗、记忆学习、冻结测试、评分、结果保存与断点恢复。**先确认当前版本的实际可运行情况，再修复阻塞问题，交付能够复现运行的测评流程。当前阶段以流程完整、行为正确、结果可读取为完成目标。

**任务范围和验收以第 1 节为准。**后文的科室组合、250 次诊疗规模、多方法多种子比较及效果分析保留为参考方案，由接手人核实后再确定后续安排。本次交接不要求完成方法排名、多种子统计或取得正向 RSI 增益。

下文命令均从 Python 包的父目录执行；在完整工作区中为 `C:\Users\12084\Desktop\研究生\code\MemAtlas\src`。单独获取团队仓库时，将仓库目录命名为 `mematlas`，并在其父目录操作。医院源码、完整病例数据和名单已经随框架提供。

## 1. 接手后具体做什么

1. **核实科室记录，确认采用哪些科室与病例。**先阅读 `mematlas/integrations/openhospital_runtime/hospital_agent_example/data/ref_data/departments.json` 的完整 19 科室名单、`mematlas/integrations/openhospital_runtime/department_patient_ids.json` 的分科室与分划分患者记录，以及 `1008任务执行/02_OpenHospital实验方案.md` 第 2 节的历史候选安排。原始科室患者分组还保存在医院数据目录的 `patients_by_department.json`。结合病例数量、诊疗与评分可用性，确认本轮联调科室、单科室重复／新病例安排和多科室安排，并与负责人对齐。第 3.1 节列出完整病例数量，后文命令中的科室仅用于演示参数写法。

2. **确认当前流程能否实际跑通，并定位阻塞位置。**从接收到的实际代码和环境出发，检查三个医院服务、医生、模型配置、Memory 接入和测评入口。历史小样本运行用于参考；需要在本次交接版本上重新执行一次小规模真实模型运行，确认诊疗、评分、反思、学习与测试分别能执行到哪一步。记录失败位置、错误信息、复现命令及已有输出，形成明确的修复清单。

3. **修复并完成一条可复现的测评闭环。**优先解决医院接入、输入输出、执行调度、反馈和结果保存中的阻塞问题，完成“病例选择 → 医生诊疗 → 官方评分 → Memory 学习 → 固定病例测试 → 保存结果”。先使用无记忆和 BM25 配置联调，再确认负责人提供的实际参评 Memory 能完成相同流程。涉及方法算法或公共框架的问题保留复现并协调处理，医院测评侧的修复做好 Git 提交。

4. **确认 RSI 测评的关键行为正确。**用少量真实病例检查三种任务流：同一患者至少独立诊疗两轮；同科室的不同患者依次诊疗；至少两个科室连续诊疗。核对记忆在轮次与科室间延续、每次就诊独立记录、训练与测试病例隔离、测试不学习、评分反馈对应当次经历，以及完成诊疗数与实际学习数准确。先验收即时反馈，再检查现有批次／延迟配置的行为与预期一致。

5. **确认恢复和结果输出可用。**验收一次中断续跑，检查已完成病例和记忆状态可以恢复；诊疗、评分与学习进度可以对应。检查原始病例结果、临床评分、测试窗口和曲线文件能正常保存和读取，同时确认模型调用与 Token、耗时、内存和存储等计算成本记录的实际覆盖。分数用于检查测评是否正常工作，记录缺失项及原因。

6. **交付可直接继续使用的版本。**提供修复提交、实际采用的科室与小样本名单、无密钥配置、依赖与启动命令、真实运行输出，以及简明的已完成／剩余问题记录。最终由接手人确认“当前能完成哪些测评、还有哪些限制、扩大病例规模还缺什么”。在上述闭环验收完成后，再与负责人确定正式规模和方法效果对照的安排。

## 2. 目前已有的代码与证据

测评实现已合入原项目目录。2026-10-26 核对时，历史外层目录分支为 `refactor/1011-framework-episodes`，当时记录的代码基线为 `eb41e98`；其中包含此前截至 `4c89439` 的 OpenHospital 修复，之后又将公共回合、反馈和调度职责统一到框架层。**医院材料现已收拢至团队仓库的独立目录，当前运行版本以团队提交及实际工作区为准；历史外层记录用于追溯此前的联调过程。**

| 已有能力 | 接手时需要检查的行为 |
|---|---|
| 三种任务流与固定测试面板 | 相同 seed 下名单、顺序、诊疗次数与冻结计划一致 |
| 多轮记忆与独立就诊身份 | 同一患者再次诊疗时生成新的就诊记录，前轮经验继续可用 |
| 医院输入与记忆归因 | 轨迹有顺序；反馈关联当次就诊及实际放入医生输入的记忆 |
| 即时、区块、延迟反馈 | 反馈边界与评估间隔独立；未到期反馈进入持久队列 |
| 测试冻结 | 测试前后记忆与反馈状态指纹一致，测试评分用于报告 |
| 单例官方评分与统一汇总 | 训练和测试评分口径一致，评分失败保留失败状态 |
| 逐病例恢复与医院世界持久化 | 保留已完成诊疗、评分、待反馈状态和历史用量 |
| 结果、曲线与用量产物 | 可定位到具体病例、测试窗口及实际记忆应用 |

此前完成过真实模型小样本运行和故障恢复。主目录保留的示例为 `.tmp/1011-original-smoke/live-resumed/`，完整运行再次恢复的示例为 `.tmp/1011-original-smoke/complete-resumed/`。这些记录属于当时版本的联调证据；接手后需要针对冻结的交接版本重新做小样本验收。正式的方法比较、三种条件的 RSI 效果及多种子结论尚待本任务完成。

主要代码位置如下，按职责阅读即可：

| 位置 | 用途 |
|---|---|
| `-m mematlas.integrations.openhospital` | 测评启动入口 |
| `mematlas/integrations/openhospital_runtime/run_openhospital_local.py` | 本地医院服务启动入口 |
| `mematlas/integrations/openhospital/` | 医院病例计划、医生接入、执行、评分汇总及曲线 |
| `mematlas/evaluation/`、`mematlas/runtime/` | 通用持续学习、反馈、测试冻结和恢复 |
| `mematlas/integrations/openhospital_runtime/` | 医生示例、医院 SDK、医院服务及原始病例数据 |

## 3. 科室与实验协议

### 3.1 完整科室记录与待确认事项

**OpenHospital 共包含 19 个科室。当前交接保留完整可选范围，联调科室、后续单科室与多科室组合均由接手人核实后与负责人确认。**按负责人当前要求，选科室时先考虑心内科以外的科室。

相关文件按用途读取：

1. `mematlas/integrations/openhospital_runtime/hospital_agent_example/data/ref_data/departments.json`：完整科室名称。
2. `mematlas/integrations/openhospital_runtime/department_patient_ids.json`：分科室、分 train／primary_test／advanced_test 的患者 ID；下表由该文件统计。
3. `mematlas/integrations/openhospital_runtime/hospital_service/hospital_service/service/data/patients_data/patients_by_department.json`：原始科室患者分组；同目录及各 split 的 `patient_departments.json` 保存患者对应的科室标签。
4. `1008任务执行/02_OpenHospital实验方案.md` 第 2 节，以及 `1008任务执行/evidence/streams/`：历史候选与当时生成的病例计划。

截至 2026-10-26，按当前科室映射筛出的病例数量如下：

| 科室 | train | primary_test | advanced_test |
|---|---:|---:|---:|
| 儿科 | 205 | 28 | 17 |
| 免疫科 | 205 | 28 | 25 |
| 内分泌科 | 451 | 47 | 28 |
| 口腔科 | 221 | 24 | 9 |
| 呼吸内科 | 712 | 65 | 46 |
| 妇产科 | 432 | 58 | 24 |
| 心内科 | 861 | 76 | 89 |
| 感染科 | 687 | 79 | 32 |
| 普外科 | 214 | 27 | 9 |
| 泌尿外科 | 439 | 44 | 25 |
| 消化内科 | 438 | 50 | 20 |
| 皮肤科 | 627 | 112 | 71 |
| 眼科 | 764 | 71 | 67 |
| 神经内科 | 435 | 50 | 37 |
| 耳鼻咽喉科 | 781 | 63 | 43 |
| 肾内科 | 233 | 12 | 9 |
| 肿瘤科 | 217 | 23 | 10 |
| 血液科 | 448 | 37 | 27 |
| 骨科 | 640 | 106 | 52 |

这些数量采用“患者科室标签只有一个科室”的筛选规则，保留同科室合并症。原始数据为 train 10,000、primary_test 1,000、advanced_test 1,000 例；当前映射筛选后分别为 9,010、1,000、640 例，跨科室病例已排除。确认所用病例时，还需核对其诊断标签与诊疗、评分可用性。

历史方案使用**心内科、呼吸内科、消化内科、内分泌科、神经内科**作为五科候选，单科室部分使用心内科。目前能定位的文件未记录后来排除心内科后的最终组合，当前采用的科室仍待确认。

接手人先确认一个能完成真实诊疗与评分的科室，用少量病例跑通闭环，再确认至少两个科室的连续诊疗。后续扩大到单科室 250 名新患者或五科室各 50 人时，再按各 split 实际数量选择科室、调整规模。表中部分科室不足 250 名训练患者，口腔科、普外科与肾内科的 advanced_test 各只有 9 例，需要相应调整后续方案；小规模联调可据实际病例数安排。

当前代码省略科室参数时仍会采用含心内科的旧默认值，**所有命令都应显式填写训练科室 `--departments` 和测试科室 `--evaluation-departments`**。

### 3.2 三种任务流与共同测试面板

| 条件参数 | 训练安排 | 独立训练患者数 | 完成诊疗次数 | 主要观察 |
|---|---|---:|---:|---|
| `repeated` | 选定科室固定 50 人，每轮重新问诊诊疗，共 5 轮 | 50 | 250 | 直接复用、反馈修正与病例熟悉收益 |
| `novel_same_department` | 同一选定科室 250 人，每人一次，分 5 个区块 | 250 | 250 | 同科室未见病例上的经验迁移 |
| `novel_departments` | 五科室各 50 人，每人一次，按科室区块进入 | 250 | 250 | 跨科室迁移、干扰和退化 |

50×5 来自会议举例，下述规模作为后续方案参考。具体科室、病例数和窗口由接手人确认。按五科室方案运行时，三种条件共同使用五科室测试面板：每科 primary 10 人、advanced 10 人，共各 50 人。primary 在完成诊疗数 0、50、100、150、200、250 时运行；advanced 在最后运行。

同一 seed 下，重复组的 50 人包含于同科室新病例组的 250 人中；三个条件的 primary、advanced 患者及顺序完全一致。训练、primary、advanced 来自三个独立原始划分。测试面板用于展示表现，方法调参和候选选择使用训练内部数据。

**每个方法、每个条件、每个种子默认会执行 250 次训练诊疗与 350 次测试诊疗，合计 600 次。**每次诊疗又包含多次模型调用，规划实验规模时需估计总调用量与耗时。先完成小样本，再安排完整矩阵。五科室进入顺序也是实验条件；多个种子默认改变抽样与病例顺序，科室区块顺序需单独控制或在结果中明确记录。

### 3.3 学习、反馈与评估

主实验建议先固定 `--feedback-timing immediate --feedback-generation after_case --eval-every 50`。医院奖励默认是诊断准确性、检查精确率、治疗总体得分的等权均值；`--reward-weights 1 1 1` 控制权重，`--success-threshold 0.5` 控制成功标签。参评方法共同使用预先确定的映射，变体另列。

反馈时机补充实验可以采用：

| 设置 | 实际含义 |
|---|---|
| `--feedback-timing immediate` | 每例结束后送达反馈并触发方法学习 |
| `--feedback-timing batch --feedback-every 50` | 每完成 50 次诊疗送达积累的反馈 |
| `--feedback-timing delayed --feedback-delay N` | 每例反馈在额外完成 N 次诊疗后送达 |
| `--feedback-generation after_case` | 病例结束就评分、反思，之后按调度送给 Memory |
| `--feedback-generation on_delivery` | 到反馈送达时才评分、反思和学习 |

两种生成方式控制不同的实验条件，报告必须注明。延迟实验结束时，尚未到期的反馈保留 pending；比较时对齐完成诊疗次数，并同时报告实际学习数量。对需要多例积累、成功轨迹或效用更新的方法，评估窗口应包含其产生可用经验、再次检索和更新的完整机会。

### 3.4 同时评价效果与计算成本

会议依据：`0921会议记录.txt` 第 51、63 行，老师明确要求比较效率、检索效率、内存占用和效果；`1005会议记录.txt` 第 313、319 行，老师再次强调现有工作对成本考虑不足，以及“成本非常重要”。据此，本任务将计算成本落实为获得和维持 RSI 改善所消耗的计算资源，不要求按 API 单价换算金额。

| 评价维度 | 本任务需要报告的内容 |
|---|---|
| 模型计算开销 | LLM 输入／输出 Token、调用次数，Embedding 调用次数及可获得的输入量；按实际模型分别记录 |
| 时间与效率 | 总运行时间、单病例耗时、记忆写入／学习／维护耗时、检索平均延迟及随经验积累的变化 |
| 资源占用 | 进程峰值内存、记忆与索引存储大小；涉及本地模型训练或演化时补充实际 GPU 时长与峰值显存 |
| 准备与持续运行 | 预训练、预演化、初始记忆及索引构建单列；持续诊疗中的检索、经验生成、维护和更新分别统计 |
| 效果与成本的关系 | 同诊疗次数下同时比较效果和资源开销；固定模型时补充效果随累计 Token 或运行时间的变化，观察相近计算资源下的收益 |

统计时分开记录医生执行、Memory 操作、医院模拟及评分开销，训练与检查点评估也分别汇总，从而看清方法自身开销及整个测评的计算量。耗时比较固定硬件、模型和并发设置；不同模型的 Token 分开报告，Token 数只作为模型使用量指标。通过远程 API 无法获取的 GPU 时长和 FLOPs 保留未测状态。

当前代码已有模型用量、记忆操作与检索耗时、进程峰值内存及存储统计。外部模型服务与数据库进程的内存、GPU 资源不在本地进程统计范围内，需结合实际部署补充。接手人检查必需指标的覆盖情况，缺少记录时交给框架负责人补齐；结果中写明测量范围。既有输出中的金额字段不纳入本任务的测评指标与验收要求。

## 4. 主流程：固定医生，更换 Memory

```text
对每个方法、每种任务流、每个 seed：
    载入固定的病例顺序、共同测试面板和医生配置
    创建本次实验的独立 Memory、输出目录和医院用量身份
    在初始 Memory 上冻结运行 primary 测试，记录 checkpoint=0

    对训练顺序中的每次就诊：
        创建新的就诊身份，即使患者曾经出现过
        医生问诊、检查、决策；按方法检索并应用已有记忆
        保存有序诊疗轨迹与本次实际应用的记忆来源
        按反馈规则生成或排队评分、反思
        送达到期反馈，让方法按自己的学习机制更新 Memory
        保存进度、记忆和待反馈状态

        到达评估窗口时：
            冻结当前 Memory 与反馈状态
            对固定 primary 病例重新诊疗、评分
            核对冻结指纹，保存病例结果与汇总表现
            恢复训练阶段，继续沿用当前 Memory

    最后冻结运行 advanced 测试
    保存学习曲线、记忆使用记录、原始评分和计算资源统计
```

反复出现的训练患者可以带来具体病例记忆收益；未见患者测试反映迁移。分析时同时观察两条曲线，并结合无记忆对照判断模型与环境波动。

## 5. 从接收文件到首次运行

### 5.1 准备环境和模型

已有可用环境时直接使用。新机器建议先用 Python 3.11 建环境，完成无记忆与 BM25 联调，再按正式方法安装其数据库及其他依赖。

```powershell
Set-Location 'C:\Users\12084\Desktop\研究生\code\MemAtlas\src'
py -3.11 -m venv .venv
$Python = '.\.venv\Scripts\python.exe'
& $Python -m pip install -r mematlas/integrations/openhospital_runtime/requirements.txt
& $Python -m pip install pytest pytest-asyncio openai
```

独立医院目录的依赖清单包含 FastAPI、Uvicorn、SDK 与服务所需的依赖。SDK 和医生源码从独立医院目录加载；本地入口使用文件持久化，运行它无需配置 PostgreSQL。正式方法的额外依赖按方法要求安装。

在本机私有文件 `mematlas/integrations/openhospital_runtime/.env.local` 配置 `MODEL_API_KEY`。本地服务和医生会自动读取。医院模型地址与模型可分别通过 `HOSPITAL_MODEL_BASE_URL`、`HOSPITAL_PATIENT_MODEL` 配置。当前医生 SDK 默认使用 DashScope 兼容接口与 `qwen3.5-flash`；医院模型环境变量不会自动更换医生模型。Memory 方法需要的 LLM、Embedding 由其方法配置提供。

开始实验前记录医生、患者、检查、评分及 Memory 各自实际使用的模型与配置。密钥由接手人在本机设置，交接材料提供环境变量名称与无密钥配置。

### 5.2 启动三个医院服务

在包父目录打开三个终端，分别执行一条命令：

```powershell
.\.venv\Scripts\python.exe mematlas/integrations/openhospital_runtime/run_openhospital_local.py --dataset train --port 8001 --world-dir .mematlas/oph-world-rsi-v1
.\.venv\Scripts\python.exe mematlas/integrations/openhospital_runtime/run_openhospital_local.py --dataset primary_test --port 8002 --world-dir .mematlas/oph-world-rsi-v1
.\.venv\Scripts\python.exe mematlas/integrations/openhospital_runtime/run_openhospital_local.py --dataset advanced_test --port 8003 --world-dir .mematlas/oph-world-rsi-v1
```

健康检查分别访问 `http://127.0.0.1:8001/health`、`http://127.0.0.1:8002/health`、`http://127.0.0.1:8003/health`，核对 dataset 为 train、primary_test、advanced_test，且均有 world_id 与 contract。

同一组对照沿用这套固定医院世界；服务重启时保持原 world-dir。修改数据、医院模型或服务代码后新建世界目录并建立新的实验组。病例抽样 seed 控制名单和顺序，患者回答与模型评分仍存在随机性。

### 5.3 准备两个联调方法配置

在测评终端执行以下命令，生成不含密钥的配置：

```powershell
$Python = '.\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force .mematlas/oph-configs | Out-Null
@{method=@{method_id='baseline_none'; params=@{}}} |
  ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 .mematlas/oph-configs/none.json
@{method=@{method_id='baseline_bm25'; params=@{}}} |
  ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 .mematlas/oph-configs/bm25.json
```

正式参评方法使用负责人提供并验收过的配置。AWM 临床配置需要 `profile="medicine"`，ExpeL 需要 `benchmark="medicine"`；实验名称记录其临床适配身份。方法配置保留 `runtime.state_mode=fresh`，测评恢复由 `--resume-from` 控制。

### 5.4 先生成三种计划，再运行真实小样本

本节及第 5.5 节命令中的呼吸内科、消化内科、内分泌科、神经内科、血液科用于演示参数写法，均需替换为确认后的科室。小规模联调可以只选一个科室，再扩展到两个科室；五科室组合在后续需要时确定。

以下命令仅检查数据并保存计划，**不调用模型**：

```powershell
$Panel = @('呼吸内科', '消化内科', '内分泌科', '神经内科', '血液科')
& $Python -X utf8 -m mematlas.integrations.openhospital `
  --prepare-comparison --departments $Panel --evaluation-departments $Panel `
  --cases-per-block 50 --blocks 5 --heldout-per-department 10 `
  --advanced-per-department 10 --seed 0 --eval-every 50 `
  --output-dir .mematlas/oph-rsi-plans/seed0
```

检查三个条件目录中的 `frozen_cohort.json`：独立训练人数依次为 50、250、250，诊疗次数均为 250，检查点均为 0、50、100、150、200、250，两个测试集各 50 人且三个条件完全一致。2026-10-26 已对命令中的示例组合执行上述等价检查并通过，验证文件位于 `.tmp/20261026-rsi-handoff-check/plans/`；这项检查说明该组合的数据数量与计划生成可用。

下面开始**调用真实模型**：同一个呼吸内科训练患者重新诊疗两次，每个测试划分取一人。

```powershell
& $Python -X utf8 -m mematlas.integrations.openhospital `
  --condition repeated --departments 呼吸内科 --evaluation-departments 呼吸内科 `
  --cases-per-block 1 --blocks 2 --heldout-per-department 1 `
  --advanced-per-department 1 --seed 0 --eval-every 1 `
  --feedback-timing immediate --feedback-generation after_case `
  --method-config .mematlas/oph-configs/bm25.json `
  --output-dir .mematlas/oph-rsi-smoke/bm25-repeated
```

随后用独立目录完成无记忆对照、两个不同患者的同科室流程，以及两个科室各一人的多科室流程。对应修改为 `--condition novel_same_department`，或 `--condition novel_departments --departments 呼吸内科 消化内科 --blocks 2`；测试面板也显式指定。

额外验收一次 `batch --feedback-every 2 --feedback-generation on_delivery`：在第一次诊疗后的窗口，应看到一例待反馈；第二次诊疗后，两例反馈送达。再验收一次中断恢复。恢复时保留原运行参数，添加 `--resume-from <原运行目录>`，同时把 `--output-dir` 改为新的空目录。沿用原医院世界、方法配置、代码和奖励规则；框架会核对恢复条件。

### 5.5 完整主实验启动方式

小样本验收后，在同一测评终端运行以下示例。它会依次启动 BM25 的三个条件，实际调用模型。

```powershell
$Panel = @('呼吸内科', '消化内科', '内分泌科', '神经内科', '血液科')
foreach ($Condition in @('repeated', 'novel_same_department', 'novel_departments')) {
  $TrainingDepartments = @($Panel[0])
  if ($Condition -eq 'novel_departments') { $TrainingDepartments = $Panel }
  & $Python -X utf8 -m mematlas.integrations.openhospital `
    --condition $Condition --departments $TrainingDepartments `
    --evaluation-departments $Panel --cases-per-block 50 --blocks 5 `
    --heldout-per-department 10 --advanced-per-department 10 --seed 0 `
    --feedback-timing immediate --feedback-generation after_case --eval-every 50 `
    --reward-weights 1 1 1 --success-threshold 0.5 `
    --method-config .mematlas/oph-configs/bm25.json `
    --output-dir ".mematlas/oph-rsi/baseline_bm25/$Condition/seed0"
  if ($LASTEXITCODE -ne 0) { throw "Run failed: $Condition. Inspect the saved output before continuing." }
}
```

无记忆对照更换为 `none.json`，输出路径的方法名改为 `baseline_none`。其他方法更换为其正式配置与独立目录。更换 seed 时同步更改参数及目录名。计划生成与真实运行需使用相同参数；运行器会重新生成并保存本次实际名单，可与预先计划核对。

目前 CLI 只接受上述三种 condition；旧文档中的 `--condition legacy` 已移除。`--checkpoints` 的帮助文字保留了旧称呼，当前任务流的评估位置以**完成诊疗次数**为准。

若修改了测评代码，优先运行现有相关回归测试：

```powershell
& $Python -m pytest -q mematlas/test/test_openhospital_layout.py `
  --basetemp=.pytest-tmp/oph-handoff --tb=short
```

## 6. 验收、结果读取和交付

1. **协议验收。**三种任务流病例数量正确，测试面板相同；重复组每次重新诊疗，跨轮记忆延续；新病例组没有重复训练患者。新运行的 Memory 相互独立，恢复运行保留原状态。

2. **学习与隔离验收。**小样本中能够查看训练经历、学习记录及第二次诊疗的记忆输入。所有测试窗口的 `frozen_verified` 为 true；测试前后持久状态一致，测试轨迹与评价没有进入后续训练。延迟实验的队列状态与实际学习数量相符。

3. **评分与恢复验收。**完整运行的 `status` 为 completed，计划内窗口均有结果。`evaluation_coverage`、`metric_counts`、失败及缺失病例数能对应到原始记录。评分失败保持失败状态；中断后恢复能够复用已完成诊疗，并保留历史用量。

4. **结果分析验收。**同时报告重复训练表现和固定未见病例表现，按科室查看迁移与退化。主要指标为 `diagnosis_accuracy`、`examination_precision`、`treatment_overall_score`，报告初始／最终得分、增益、完整曲线的 AUC 和最大退化。`official_examination_precision` 使用不同有效分母，单独标明口径。结合第 3.4 节报告模型用量、耗时、内存与存储，分析效果与计算成本的关系。记忆实际应用次数说明复用频率，收益判断结合对照表现。

5. **最终交付。**提交冻结实验配置、全部原始输出、效果与计算成本汇总表及曲线，以及一份简短分析，回答“哪些方法在重复病例上改善、哪些改善能迁移到新病例、跨科室是否退化、反馈时机会怎样影响结果、改善需要多少额外计算资源”。多种子比较以相同测试患者的配对结果为基础，统计时保留患者和种子层面的相关性。修复测评代码时单独提交 Git，并记录使用版本。

| 产物 | 读取用途 |
|---|---|
| `frozen_cohort.json`、`execution.json` | 实际病例、顺序、反馈配置与医生／医院／框架运行条件 |
| `result.json` | 完成状态、实际进度、学习曲线摘要、资源与记忆复用汇总 |
| `curve.json` | 训练区块和测试窗口的详细指标、冻结验证与反馈进度 |
| `curves/` | CSV 与 SVG 学习曲线，现有图主要展示分项得分 |
| `test_cases/`、`checkpoints/` | 测试病例评分与窗口汇总 |
| `encounters/`、`official_sdk/`、`memory_traces.jsonl` | 诊疗、反思、实际应用的记忆与学习追踪 |
| `snapshots/`、`test_execution/` | 恢复所需状态及已完成测试诊疗 |
| `usage.jsonl`、`usage_records.json`、`accounting.json` | 模型原始用量、运行阶段及历史调用记录；结合轨迹和资源指标汇总计算成本 |

计算资源记录缺失时标明未测范围，保留已知 Token、调用次数、耗时、内存和存储数据。准备阶段、持续运行阶段与检查点评估的开销分别报告。

## 7. 要发给接手人的文件

**建议交付一个保持项目目录结构的代码包，附本任务书。代码包应包含当前实际源码、三个医院组件和病例数据。**具体清单如下；表内路径均相对于项目根目录。

| 优先级 | 文件或目录 | 交接说明 |
|---|---|---|
| 必须 | `RSI_OpenHospital_测评交接.md` | 首先阅读，按此确定范围、执行与验收 |
| 必须 | `pyproject.toml`、完整 `mematlas/` | 医院适配依赖公共框架、运行时及方法注册，需一并提供 |
| 必须 | `-m mematlas.integrations.openhospital`、`mematlas/integrations/openhospital_runtime/run_openhospital_local.py`、`mematlas/integrations/openhospital_runtime/build_openhospital_department_ids.py` | 测评、医院服务与名单生成入口 |
| 必须 | `mematlas/integrations/openhospital_runtime/department_patient_ids.json` | 当前科室病例映射 |
| 必须 | 第 3.1 节所列 `departments.json`、原始科室分组文件与 `1008任务执行/02_OpenHospital实验方案.md` | 完整科室记录与历史候选方案，供接手人确认；原始文件随医院组件一并提供 |
| 必须 | `mematlas/integrations/openhospital_runtime/hospital-agent-sdk/` | 包含本项目改动的 SDK 实际源码 |
| 必须 | `mematlas/integrations/openhospital_runtime/hospital_agent_example/` | 医生源码、Prompt、`config.yaml` 与依赖文件 |
| 必须 | `mematlas/integrations/openhospital_runtime/hospital_service/` | 服务源码、依赖、检查目录和完整病例数据；保留原有相对路径 |
| 必须 | `mematlas/test/test_openhospital_layout.py`；完整工作区中的相关公共回归测试 | 医院随包运行检查与公共测评行为验证 |
| 必须 | 正式参评方法配置、医生配置、实验参数清单 | 明确方法及变体、模型、硬件与并发、科室、seed、奖励、反馈、窗口、计算资源额度；配置去除密钥 |
| 随包参考 | `RSI_OpenHospital测评任务说明.md`、`OpenHospital_系统对接说明.md` | 提供已有协议和医院背景；命令及版本以本交接文档为准 |
| 可选 | `.tmp/1011-original-smoke/live-resumed/` 与 `complete-resumed/` | 旧版真实联调与恢复样例，便于理解产物；保留引用所需文件 |
| 可选 | `.tmp/20261026-rsi-handoff-check/plans/` | 本次不含心内科的三种计划核验结果，供核对格式 |

发送时还需注意以下几点：

1. **版本随包冻结。**记录团队仓库提交、当前工作区变更，以及 `sources.json` 中的组件来源版本。交接前由相应代码负责人确认哪些未提交修改进入版本；采用源码包时直接提供确认后的实际文件，并保留文件清单或校验值。仅导出当前 HEAD 会遗漏未提交文件和新方法目录。

2. **医院组件作为独立项目随仓库提供。**SDK、医生示例、医院服务和完整数据位于 `mematlas/integrations/openhospital_runtime/`，使用自己的启动脚本，作为普通文件纳入团队 Git；来源提交与迁入前的本地改动清单见该目录的 `sources.json`。获取团队仓库即可取得这些材料，原始 `openhospital.zip` 留作历史归档。

3. **原始数据保持完整目录。**医院服务需要患者资料、参考诊断、科室标签与检查目录；仅有 `department_patient_ids.json` 无法执行诊疗。参考诊断留在医院评分端，医生与 Memory 输入使用可观察信息。

4. **运行环境在对方机器重建。**个人 `.venv/`、缓存、`__MACOSX/`、无关方法原仓库和旧实验大目录可从交接包排除。若某个正式方法确实需要额外模型、初始记忆或准备产物，在其配置清单中单独列出并提供。服务和医生配置发送无密钥版本，`.env.local` 由接手人按变量说明配置。

5. **新实验与续跑分开交付。**本次迁移改变了源码位置和指纹，新实验使用新的医院世界目录；需要续跑历史实验时，另发对应 world-dir 和完整运行目录，并保留当时的运行版本及配置。旧的 `MemAtlas-rsi-fix` 路径无需作为交接来源，本任务以原 `MemAtlas` 目录冻结出的版本为准。

交接验收的首个成果应为：**对方能在自己的环境生成三种计划，并完成一次两轮真实诊疗、冻结测试和中断恢复。**随后按冻结协议执行完整测评。
