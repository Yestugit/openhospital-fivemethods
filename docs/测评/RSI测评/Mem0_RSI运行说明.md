# Mem0 × OpenHospital RSI

> 历史说明：本文对应 2026-11-02 的 Mem0 additive 重实现及当时框架。2026-11-05 起的
> 官方 native 适配另见 [五方法适配验收](../../方法集成/20261105_五方法native适配验收.md)。
> 旧检查点不自动迁移；本文的病例安排不作为新实验方案。

本分支从已核实远端 main 的 b266e2a 起步，原 MemoryOS 工作区和实验保持原位。新实验使用自己的 world、记忆库和统计；不得恢复旧 MemoryOS 实验来运行 Mem0。

## 配置变体

使用本项目集成的 Mem0 additive 实现，standard、infer=true、write_policy=episode、hybrid_search=true、rerank=false、top_k=5。每个完整病例提交后执行 LLM 提取；通常每例一次，跨日期事件分段仍按方法的时间锚定规则处理。保留语义、BM25 和实体混合排序规则，英文 spaCy 模型在中文输入中可能没有实体或关键词贡献。此配置不等同于 Mem0 默认逐轮写入，也不等同于论文的四操作更新版本。

医生/患者/记忆 LLM 使用 qwen3.5-flash，记忆提取关闭 thinking。Embedding 使用现有 Qwen3-Embedding-0.6B、1024维、本地GPU优先，模型指纹固定。完整记忆正文与临床证据不按字符裁剪；查询沿用公共医院适配的原预算。既有记忆展示不再作为新临床事实递归写入，实际应用来源仍由 applied_memory 保留。

## 环境

本次环境在新工作树的 .venv 中安装新增依赖；通过只读 .pth 复用原环境的已安装基础包，原环境不安装或升级。小写 mematlas 包入口通过环境内目录联接指向当前工作树，使子进程使用本分支代码。环境和联接均不提交。

全新环境可安装 configs/openhospital/mem0-requirements.txt，并按硬件安装匹配的 PyTorch；CUDA环境参考旧实验的 torch==2.8.0、cu126。Qwen权重无需重复下载。密钥只通过环境文件读取，不写入配置、日志或Git。--env-file 可引用原环境文件，--model-dir 可引用原模型目录，--reference 可引用原AWM结果。

以下命令在新工作树根目录执行；只读路径可替换为部署位置：

```powershell
$taskModel = 'D:/master_study/MemAtlas/models/Qwen3-Embedding-0.6B'
$taskEnv = 'D:/master_study/MemAtlas/integrations/openhospital_runtime/.env.local'
$taskReference = 'D:/master_study/MemAtlas/.mematlas/oph-rsi/20261028-173625-929/awm'

# 离线计划、依赖与实际病例文件哈希检查，不调用模型
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py check --model-dir $taskModel --env-file $taskEnv --reference $taskReference

# 独立预跑：20次训练，第10例后自动保存并重启，随后2次冻结测试
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py run --pilot --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/pilot-seed0 --model-dir $taskModel --env-file $taskEnv --reference $taskReference

# 读取状态、报告；report不调用模型
.\.venv\Scripts\python.exe scripts/openhospital-mem0-rsi.py status --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/pilot-seed0
.\.venv\Scripts\python.exe scripts/openhospital-mem0-rsi.py report --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/pilot-seed0

# 手动停止、恢复、监控
.\.venv\Scripts\python.exe scripts/openhospital-mem0-rsi.py stop --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/pilot-seed0
.\.venv\Scripts\python.exe scripts/openhospital-mem0-rsi.py resume --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/pilot-seed0 --env-file $taskEnv
.\.venv\Scripts\python.exe scripts/openhospital-mem0-rsi.py monitor --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/pilot-seed0 --port 8768
```

恢复必须沿用原模型和配置，模型指纹及诊疗代码变化会阻止续跑。只终止本启动器拥有的服务，不接管占用端口；医院使用8001/8002/8003，本地embedding使用8766。

## 正式实验与英文实时监控

seed=0，重复组250次、同科新病例250次、跨科训练250次+primary300次+advanced50次，共1100次。科室依次为血液、呼吸、消化、内分泌、神经。正式三组并发，病例、完整顺序和本工作树实际数据哈希必须与旧AWM参考一致。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py run --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/formal-seed0 --model-dir $taskModel --env-file $taskEnv --reference $taskReference --monitor --port 8767
```

`--monitor` 在服务准备阶段就启动英文仪表盘，并自动打开默认浏览器，地址为 `http://127.0.0.1:8767`。页面每3秒刷新，终端每10秒输出状态；当前PowerShell应保持打开。若浏览器未自动打开，手动访问此地址即可。完整实验尚未启动。

仪表盘同时展示三组各自的训练/冻结测试进度、总完成比例、当前病例、科室、轮次、检查点、病例耗时、最近5例平均耗时和进程/心跳状态；展示诊断准确率、检查precision/recall、治疗及其安全性/有效性/个性化评分、评分覆盖率，以及所有已完成冻结检查点的指标。调用与已知输入/输出token按医生、记忆提取、患者模拟、检查模拟、评分和embedding分列，另列缓存、未知用量、失败请求、JSON修复/兜底、学习次数和待反馈数。

医生/记忆请求完成后用量会随心跳更新；医院模拟与评分的用量在每例结束同步。心跳超过30秒会显示警示，但这不等于病例失败；错误原因与worker日志路径也会显示。未报告的指标不当作0，known tokens不包含未知请求用量；API未报告缓存字段时不推断命中率。训练评分是累计已完成病例的均值；Latest test checkpoint是当前检查点已评分病例的均值，已完成检查点另表保留。

关闭网页不会停止实验。启动器退出后，其自动监控服务会关闭；可以在另一个PowerShell中执行下面的独立监控命令，查看运行或已完成的结果。独立监控可先于实验启动，等待suite.json就绪。

```powershell
# 独立浏览器监控（与自动监控并用时换用8768）
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py monitor --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/formal-seed0 --port 8768

# 持续终端进度 / 请求按病例边界停止
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py watch --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/formal-seed0
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py stop --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/formal-seed0

# 启动器退出后，从原正式目录恢复；配置、模型和代码必须保持一致
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py resume --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/formal-seed0 --env-file $taskEnv --monitor --port 8767
```

2026-11-03已按要求精简旧MemoryOS数据：`D:/master_study/MemAtlas/.mematlas/oph-rsi/memoryos/retained-results` 保留部分实验的逐例最终结果、评分、配置、曲线与去重用量，约10 MB；快照、原始轨迹、数据库和重复日志已删除，旧MemoryOS实验不可恢复。原模型、密钥文件、AWM参考及Mem0预跑保持原位。

## 容错、缓存和审计

原响应和修复响应使用同一字段与嵌套类型验证，检查名称必须来自当前类别且未执行。JSON问题在当前决策内修复，最多追加3次JSON模式请求。exam_item仍失败时，按目录顺序采用当前类别第一个未做检查；类别无效采用第一个还有未做检查的类别；候选耗尽则进入已有信息下的诊断。兜底只生成检查请求，不伪造检查结果。其他决策或提取仍无法恢复时保存可恢复边界，不无限重跑。

每次医生响应、修复、校验错误、用量及兜底事件保存于official SDK日志；提取原始响应和每次修复保存于条件目录memory-responses.jsonl。结果记录json_repair_count/json_fallback_count，兜底病例仍计入临床评分，不静默剔除。

仅缓存正文完全相同且模型指纹相同的embedding（最多2048条），实际推理与命中/节省token单列。复用确定的NLP解析和稀疏编码。稳定记忆前缀与提取系统提示启用独立消息边界缓存，正文及顺序不变，不缓存医生答案。API是否命中依实际返回统计，缺失字段按未知处理。

报告按训练/评测阶段及doctor、memory、patient、examination、judge分别计数；按record_id合并恢复前后记录，未知用量不伪记为零。费用仅有可靠单价时换算，短预跑不代表正式效果或确定总费用。

本次已完成的20＋2预跑结果见 [预跑报告](Mem0_RSI预跑报告.md)；正式启动命令已就绪，尚未执行。

2026-11-03 API故障恢复补充：医生直连/代理与记忆模型的临时超时、连接故障、408/429和5xx最多尝试8次，重试保持同一请求内容，间隔2/4/8/16/30/30/30秒。医生和记忆模型单次超时120秒；鉴权等永久错误不重复尝试。每次失败保留用量未知记录，不将未知token计为零。成功请求不增加等待。

重试耗尽返回可恢复暂停；启动器等待30/60/120秒，最多自动恢复3次（同一已完成病例位置），有进度后重新计数。长期不可用则停留paused，保留恢复数据。恢复从最近完整病例断点开始：已完成病例和学习不重做，未完成病例重新诊疗，原失败轨迹与费用保留在旧attempt中。病例内API重试不重新执行问诊、检查或学习。

此次运行在第一组59、第二组35、第三组35次训练及50次冻结测试后保存，第二组从第36例Patient_08256恢复。源代码严格恢复校验仍启用；API重试补丁采用一次性白名单迁移工具`scripts/openhospital-retry-migration.py`，先离线验证再显式`--apply`。迁移保存原execution和快照清单及前后哈希收据，只更新运行代码签名，逐个核对记忆文件哈希不变；医院世界、病例、模型和临床策略不变。后续正常resume无需再次迁移。

2026-11-04 检查指标名称修复：第三组第153例Patient_04557的肾上腺功能检查因模型在指标名中插入空格而反复校验失败，医院返回500。现在仅在目录映射唯一时兼容空白及全/半角括号；不合并不同采样时间、体位或标本，重复指标、缺失值、非法状态仍严格拒绝。完整解析与修复阶段的部分指标解析共用该规则，不填造检查结果。

医院HTTP客户端也采用最多8次临时故障重试及2/4/8/16/30/30/30秒退避，耗尽进入现有可恢复暂停/自动恢复路径；401等永久错误立即报告。本次仅恢复第三组，前两组已完成。迁移使用`--patch exam-names`，备份三份医院SQLite数据库、execution和快照清单，白名单只接受检查解析器及SDK HTTP客户端变化；医院数据表和记忆文件逐个哈希核对不变。医院world_id随解析器契约版本更新并记录前后对应，既有检查、请求和用量记录全部保留。恢复起点为152次训练、200次冻结测试；第153例重新诊疗，已完成病例及学习不重复，失败日志和用量保留。

## Primary / Advanced 测试与并行

原始OpenHospital数据按既有名单划分为train 10,000人、primary_test 1,000人、advanced_test 1,000人。本实验只从科室标签恰好等于目标科室的患者中抽样；按排序后的ID以`Random(f"0:{科室}:{split}")`洗牌选取，与旧AWM冻结名单一致。患者资料、检查和参考诊断来自相应数据划分，评分端使用参考答案，医生只能通过问诊和检查获取患者信息。

第三组五科为血液科、呼吸内科、消化内科、内分泌科、神经内科。每科训练50人，共250人，按这五科顺序学习。Primary从每科独立抽10人组成固定50人面板，在训练进度0/50/100/150/200/250各测试一次，共300次；不是六批不同患者。Advanced从另一份独立划分每科抽10人，共50人，仅最终测试一次。三份名单互斥。本次Primary抽中50人均为单参考诊断，Advanced的50人中43人为单参考诊断、7人为双参考诊断；Advanced是数据划分名称，不表示每一例必然多病或一定更难。共250训练＋300 Primary＋50 Advanced＝600次。

测试不学习、不更新记忆；Primary用于固定患者面板上的学习曲线，Advanced用于最终独立面板评估。统计中的Frozen held-out tests总计350，即300＋50。

2026-11-04加入冻结病例并行，当前正式运行设置为5路：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/openhospital-mem0-rsi.py resume --run-dir D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/formal-seed0 --env-file 'D:/master_study/MemAtlas/integrations/openhospital_runtime/.env.local' --test-concurrency 5 --monitor --port 8767
```

`--test-concurrency`取1至16，写入suite.json，后续resume默认沿用；首次未设置时为1。训练始终串行；冻结期最多5个病例同时诊疗/评分，同一病例内步骤依旧顺序执行；每例独立医生、日志目录和记忆来源记录，共享同一只读记忆，检索经过公共Memory锁串行执行。患者、检查和医生API等待可以重叠，不改变检索规则或记忆正文。当前仅允许Mem0原生只读模式启用多路。

结果按原名单顺序汇总，每例成功后独立写入诊疗及评分文件。临时故障或停止请求会停止派发新病例，等待在途病例落盘后退出；恢复跳过已完成病例，评分失败仅重试评分。医院用量导入加锁，避免并发重复计数。批次结束及每例关键步骤验证记忆指纹，冻结污染不会被接受为成功结果。

网页在冻结阶段显示Parallel tests、Active test patients和Current test panel；检查点结果包含`test_concurrency`、`test_batch_wall_time_s`，单例`response_time_s`仍表示该例实际延迟。并行主要减少整批墙钟时间，不承诺5倍加速；实际收益受检索串行、共享embedding及API限流影响。并行不会主动减少token或诊疗步骤，也不保证随机LLM响应逐字相同。本次在训练189例、冻结200例处保存并通过`--patch parallel-tests`白名单迁移，只更换执行/启动/显示代码契约，原病例、记忆及医院世界契约不变。

## 实验数据位置（2026-11-04）

代码工作树在 C 盘，正式实验数据固定保存在 `D:/master_study/MemAtlas/.mematlas/oph-rsi/mem0/formal-seed0`；所有运行命令显式传入该绝对 `--run-dir`，避免相对路径跟随代码工作树。代码路径与数据路径可以不同。

本次因 C 盘空间不足，在第三组第190次训练和200次冻结测试的完整断点停止。迁移逐文件核对 SHA256，保留 C 盘原件备份。原 C 盘数据入口通过目录联接指向 D 盘，以保持历史断点、配置和响应日志中的绝对路径有效；该入口之后新增的数据实际写入 D 盘。embedding 服务日志和 provider 检查日志也通过目录联接写入 `D:/master_study/MemAtlas/.mematlas/openhospital-mem0-runtime`。此次迁移不修改临床代码、病例、记忆模型或断点签名。

恢复仍使用 `resume`，不要对该目录重新执行 `run`。第191例已有完整诊疗和评分落盘，恢复会复用其记录，并从第190例的完整记忆状态继续学习和保存；前两组保持已完成。

2026-11-04 Windows路径修复：第三组训练200例后，5路Primary测试的SDK日志路径因完整患者哈希、SDK时间戳和`llm_prompts`目录叠加超过Windows限制，触发WinError 206。并行日志目录改用检查点、split首字母和固定名单序号，完整患者ID及哈希仍保留在执行/评分记录中。训练、测试及重评分的SDK文件入口统一采用Windows扩展绝对路径，保留完整提示词和响应，不关闭日志。Windows真实日志读写及冻结并行/恢复等28项相关回归通过；断点使用`--patch sdk-paths`白名单迁移，只允许`execution.py`变化，备份清单并逐个核对记忆哈希。恢复从200例训练断点继续Primary测试，已完成训练不重做。
