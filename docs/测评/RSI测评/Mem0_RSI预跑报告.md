# Mem0 × OpenHospital 预跑报告

状态：completed；训练 20/20，冻结测试 2/2。
冻结验证：True；第10例后恢复：True。

前5例平均耗时：58.99 秒；后5例：22.81 秒。
训练 JSON 修复：0；训练兜底：0；测试修复：0；测试兜底：0。

| 阶段 / 来源 / 类型 | 调用 | 输入 token | 输出 token | 失败 | 已报告缓存命中 token |
|---|---:|---:|---:|---:|---:|
| evaluation/doctor/llm | 14 | 20086 | 2112 | 1 | 0 |
| evaluation/examination/llm | 4 | 9398 | 2339 | 0 | 0 |
| evaluation/judge/llm | 2 | 6131 | 109 | 0 | 0 |
| evaluation/memory/embedding | 65 | 6935 | 0 | 0 | 0 |
| evaluation/patient/llm | 2 | 1238 | 77 | 0 | 0 |
| training/doctor/llm | 109 | 117069 | 19422 | 0 | 0 |
| training/examination/llm | 10 | 21990 | 4590 | 0 | 0 |
| training/judge/llm | 20 | 60339 | 1160 | 0 | 0 |
| training/memory/embedding | 1430 | 90111 | 0 | 0 | 0 |
| training/memory/llm | 20 | 337712 | 11707 | 0 | 154622 |
| training/patient/llm | 16 | 12206 | 589 | 0 | 0 |

冻结测试：诊断准确率 0.0，检查精确率 0.1，治疗总体得分 0.48。仅2例，不作方法效果结论。
Embedding服务累计：{'exact_cache_hits': 798, 'inferred_texts': 700, 'inferred_tokens': 91125, 'saved_tokens': 5959}（包含连通性检查和2次合成缓存校验；病例用量以表格为准）。
准备阶段用量单列于JSON报告preparation_usage；此前bootstrap检查另有1条LLM响应、已知16 token，原文保存在provider-bootstrap目录。
共记录1次请求失败，1次请求用量未知；失败请求仍保留在统计中，表中已知输入输出不能代表完整账单。
按预跑均值纯算术外推：750次训练串行约7.1小时，350次测试约9.6小时，合计约16.7小时；三组并发时跨科组自身约12.0小时。模型限流、病例复杂度和共享GPU会改变耗时；两个测试病例且其中一次重试，不足以作可靠工期预测。
医生含诊疗与训练反思；patient/examination 为医院模拟，judge 为评分。各阶段分别统计，恢复后的同一 record_id 只计一次。
费用未换算：未配置已核实单价。本次短预跑只用于运行与成本观察，不能证明正式实验效果或确定总费用。正式1100次实验未启动。

## 交付与验收记录（2026-11-02）

分支：`codex/openhospital-mem0-rsi`。起点为获取并核实的远端 main：`b266e2a2b11c9aeb3c105b8cdad3314e12032d3e`，创建时提交和文件树一致。原 `codex/openhospital-memoryos-rsi` 分支、未提交修改和数据保留。未推送新分支，未运行正式实验。

配置：standard / additive / infer=true / write_policy=episode / hybrid_search=true / rerank=false / top_k=5；完整病例提交后提取记忆。医生、患者和记忆模型为 qwen3.5-flash；记忆提取关闭 thinking。Embedding 为本地 Qwen3-Embedding-0.6B、1024维，指纹 `3e1181ea664dbf75585ae3c0a0d11df3a01ad3206a5f8c04b06b32476da9469a`。

- 最终相关离线回归205项通过（67.26秒）；医院检查生成器另13项通过。
- 正式病例计划和旧 AWM 的病例、完整顺序及本工作树数据哈希一致：250＋250＋600＝1100；计划核对不调用模型。
- 预跑取重复组首20次训练，第10例保存并停止；attempt-002从保存点恢复，再完成10次训练和2次冻结测试。
- 20个不同病例各提取一次；20次学习成功，最终111条记忆。第10例的102个已见事件全部保留，最终152个事件ID无重复，待写缓冲为空。
- 冻结测试前后记忆指纹一致。预跑没有自然触发JSON修复/兜底；错误语法、字段类型、非法检查名称、连续3次修复失败、目录耗尽、完整日志均通过离线回归验证。
- 精确embedding缓存开关下相同文本向量逐值一致；模型指纹参与缓存键。稀疏空查询路径及原生混合检索回归通过。
- 记忆提取输入337,712 token，其中154,622 token报告缓存命中（45.8%），8,138 token报告缓存创建。精确embedding缓存节省5,959个推理token；未测无缓存全套诊疗，因此没有确定的总耗时/费用节省比例。
- 医生缓存字段未报告，按未知处理。已知token不含那次失败请求的未知用量；模型价格未核实，不换算费用。

原始结果位于工作树 `.mematlas/oph-rsi/mem0/pilot-seed0/`（不入Git）：

| 文件/目录 | 内容 |
|---|---|
| `suite.json`、`environment.json` | 运行配置、完成状态、依赖版本、模型指纹 |
| `formal-plan-audit.json` | 三组正式病例、顺序和数据哈希核对 |
| `resume-event-audit.json` | 第10例到第20例事件、学习与提取去重 |
| `cache-equivalence.json` | 实际本地缓存向量等价验证 |
| `pilot-report.json`、`pilot-report.md` | 完整用量、运行时间与审计汇总 |
| `repeated/memory-responses.jsonl` | 每次提取的提示、响应和用量 |
| `repeated/attempt-001/`、`repeated/attempt-002/` | 恢复前后病例、SDK日志、用量、快照、评分 |

启动、停止、恢复、监控和正式命令见 [运行说明](Mem0_RSI运行说明.md)。重新预跑须使用新目录，避免覆盖本次结果。正式实验须使用独立的 `formal-seed0` 目录。
