# 按需求扩展 Agent 接入

先运行 [快速入门](Agent开发者快速入门.md)。本篇按“你要改什么”查阅，不要求每项都实现。所有位置均指仓库根目录的 [quickstart_agent.py](../quickstart_agent.py)；业务代码也可放在自己的启动模块中再导入。

| 你需要什么 | 去哪里改 | 需要动 Memory 算法吗 |
|---|---|---|
| 更换已有记忆方法、参数或服务 | A · `MEMORY_CONFIG` | 不需要 |
| 同场景更换 Agent | D · `AGENT_EXECUTE` 的绑定或 SDK 转换函数 | 不需要 |
| 更换数据文件／字段 | C、`load_tasks()`、`to_text_task()` | 不需要 |
| 接入新场景、多步检索或工具过程 | 自己的 `execute(task, memory)`，然后绑定到 D | 不需要 |
| 使用真实评分或延迟反馈 | `main()` 的 E 处，任务结束后 | 不需要 |
| 调整在线学习、批次、epoch 或冻结 | 调用 `run_task()` 的位置和外层执行循环 | 不需要 |
| 集成未注册的记忆算法 | 独立方法插件及其注册代码 | 需要实现该方法的公共适配 |

## 1. 我只想换 Memory 方法或参数

**改 A，沿用 C 和 D。**填写方法 ID、该方法参数、需要的 `providers`，配置示例见快速入门 A 节。依赖、配置和能力可用 `--describe-method 方法ID` 查询。

创建新 Memory 使用 `Memory.from_dict(config)`，不要直接替换活动实例的内部方法。不同算法的原生状态不会自动迁移。

前提是当前接入提供了新方法所需的真实轨迹、评价和学习阶段，并能接收其贡献形式。缺少这些能力时，继续看第 4—6 节；**不要为了换算法，在执行器中按方法 ID 重写检索或更新算法。**

特殊客户端不一定需要另写插件：通过 `MethodServices(llm=..., embedder=...)` 注入即可，检查与创建时都传同一个 `services=`。Agent 自己使用的模型仍在原 Agent 配置中。

## 2. 我想在同一场景换 Agent

**改 D 的 Agent 绑定；先不改任务格式和记忆接线。**原 Agent 已配置好模型、提示和工具，且异步接收完整消息、返回字符串时：

```python
# existing_agent 是自己已经创建好的对象，answer 是其兼容的实际方法。
AGENT_EXECUTE = partial(execute_text, agent_call=existing_agent.answer)
```

SDK 的调用格式不同，就在自己的项目中写转换函数，再填入 `agent_call=`。快速入门 D 节给出示例；实际调用方法和回答字段按真实 SDK 填写。同步函数、流式响应和非字符串结果也需转换。

`partial` 只预先绑定原 Agent，外层执行仍是 `execute(task, memory) → AgentTrace`。完整消息包含加入的记忆，不能只取最后一条用户消息。新 Agent 的工具钩子不同，则另外调整第 4 节的过程记录；无需更换数据读取和业务评分规则。

## 3. 我只想换数据文件或字段

**不改 Agent 和 Memory，只改 C 或读取／转换函数。**

| 数据情况 | 具体改法 |
|---|---|
| 已是支持的 JSONL 格式 | 用 `--tasks 文件路径`，或在 C 设置 `TASKS = load_tasks(path)` |
| 仍是 JSONL，只是字段名不同 | 在 `to_text_task()` 中改字段映射 |
| CSV、数据库、业务队列等 | 写自己的读取函数，逐条产生执行器接受的任务，交给外层循环 |
| 新场景需要原生输入 | 保留原始记录，交给第 4 节的执行器，不经过文本转换 |

例如，你准备的 JSONL 使用 `prompt`，不是 `question`：

```json
{"prompt": "Helios 的端口是多少？", "system_prompt": "你是运维助手。"}
```

在 `to_text_task()` 的简写分支，将 `query = row.get("question")` 改为 `query = row.get("prompt")`，其他消息组织逻辑可复用。已有对话历史时也要映射进 `messages`，不能只保留当前问题。

读取函数只负责取数据、转换格式，不调用 Memory 或 Agent。普通迭代器可以逐条消费；多轮运行需重新读取或提供可复用的数据源。`task` 是执行器输入，**不是把整条数据自动写入长期记忆的接口**。

## 4. 我想接入新场景，或让 Agent 中途使用记忆

**在业务项目中实现 `async execute(task, memory) -> AgentTrace`，再将 D 改为 `AGENT_EXECUTE = execute`。**可以参考现有 `execute_text()` 修改，不能直接将返回字符串的旧 Agent 函数当作统一执行器。

代码放在独立模块时，可从 `quickstart_agent` 导入 `AgentTrace`、`AgentStep`、`EventKind` 等已加载的公共类型；运行入口同样从该脚本导入 `run_task`。

首先约定自己的输入。例如医院可以是 `{"patient_id": "P001", "mode": "train"}`，代码修复可以是仓库位置和问题描述；C 或读取函数提供这些原始记录。执行器负责解释它们，不要求转换成 `query/messages`。

然后在真实业务流程中接入需要的公共能力：

| 你要做的事 | 放在哪里 | 使用的现有接口 |
|---|---|---|
| 根据当前信息查记忆 | 实际决策之前，可多次调用 | `await memory.recall(query, scope=memory.scope)`；复杂查询用 `await memory.retrieve(DecisionContext(...))` |
| 把记忆交给 Agent | 构造本次实际输入时 | `application = memory.apply_contribution(messages, contribution)` |
| 关联实际提供的记忆来源 | 贡献实际进入 Agent 输入后 | `memory.confirm_application(contribution)` |
| 记录消息、动作、工具结果 | 真实步骤产生时 | `await memory.record_step(AgentStep(...))` |
| 执行并记录一次模型／工具调用 | 允许包装原调用的位置 | 可选 `await memory.call_model(agent_call, messages, contribution=contribution)` 或 `await memory.call_tool(tool, name, arguments)`，同次调用不再手动重复执行或记录 |

贡献可能包含文本、直接答案、工具、模型钩子或附件，执行器必须真实处理。默认文本接入只支持文本和直接答案；不能忽略其他字段，也不能假定框架能看到 Agent 内部未上报的工具过程。

真实执行完成后，按下面的返回结构组织结果；`actual_result` 和 `task_description` 由你的执行代码产生：

```python
return AgentTrace(
    case_id=None, scope=memory.scope, steps=memory.steps,
    final_output=actual_result, metadata={"task": task_description},
)
```

`final_output` 可以是结构化结果。若已有完整外部轨迹，可把真实步骤完整放进 `steps`；如果已经逐步记录，返回步骤必须与 `memory.steps` 一致。来源通过实际应用接口关联，不编造。

**任务边界不用重写：**外层 `run_task(memory, scope, task, execute=execute)` 负责开始、核对轨迹、结束和异常清理。执行器不重复开始／结束任务或提交学习。任务 ID 默认生成，也可通过外层的 `task_id=` 指定；一次重新执行使用一个新的 ID。一个 Memory 当前不交错执行多个活动任务。

**已有 OpenHospital 用户：**优先复用 [医院接入](../integrations/openhospital/) 和 [运行说明](../docs/测评/快速说明.md)，不要把 `doctor.train(patient_id)` 直接绑定到文本入口。现有医院流程自行提交完整经历，不能再套一层 `run_task` 重复提交；要采用统一入口，需先对齐任务边界和提交责任。`mode="test"` 也不代替实际冻结。

## 5. 我需要记忆利用真实评分，或评分晚到

**改 `main()` 的 E 处，替换默认反馈提交。**已有评价器可以直接复用；否则在业务项目实现 `async evaluate(task, trace) -> EpisodeOutcome`，从真实结果取得成功标签、奖励、文字评价和来源。

```python
outcome = await evaluate(task, trace)
if outcome.scope != trace.scope:
    raise ValueError("评价与轨迹的记忆归属不一致")
await memory.submit_feedback(
    trace.case_id, scope=trace.scope,
    success=outcome.success, reward=outcome.reward,
    feedback=outcome.feedback, metadata=outcome.metadata,
)
```

如果 `execute` 已返回真实的 `success`、`reward`、`feedback`，现有主循环会传递它们，无需再次评价。不要求标签的方法可省略；要求标签的方法不能用空值或虚构评价绕过。

**评分晚到：**暂不提交，先 `await memory.checkpoint(path)`，保留 `trace.case_id`、scope、原配置和实际运行目录。稍后在原运行提交；重启后用 `Memory.resume(config, run_dir)` 打开，需要恢复保存状态再 `await memory.load_state(path)`。返回 `AgentTrace` 表示取得轨迹，不等于已经学习。

## 6. 我需要在线学习、批次更新或冻结测试

**改调用参数或外层循环，不改算法实现。**

| 需要什么 | 具体改法 |
|---|---|
| 先保存轨迹，之后反馈学习 | 保持 `run_task(..., learning=False)`，按第 5 节提交 |
| 步骤产生时就让方法观察 | 按方法要求使用 `run_task(..., learning=True)`；任务后仍按策略提交经历 |
| 批次／epoch 更新 | 在外层训练循环的真实边界调用下方接口，方法插件负责内部更新 |
| 临时冻结学习 | 在任务边界使用 `async with memory.read_only():`，其中不提交训练反馈 |
| 主动清理记忆 | `await memory.reset(scope)`；与临时冻结不同 |

自定义训练循环时，放在已经打开的 Memory 中，按这些位置通知边界：

```python
await memory.learning_boundary("epoch_start", current_epoch=0, total_epochs=1)
# 每批真实任务执行并提交后，每批调用一次：
await memory.learning_boundary("batch_end")
# 本轮全部批次结束后：
await memory.learning_boundary("epoch_end")
```

`learning=False` 不是全局冻结，保存检查点或 `flush` 也不能代替学习边界。轮数、批次划分、重放已有经历还是重新执行 Agent，由业务计划明确。默认 quickstart 只有串行文本循环，没有通用训练计划器；复杂安排可参考 [现有运行器](../evaluation/experience.py) 和 [训练测试协议](../evaluation/protocols/memory.py)。

## 7. 我需要集成仓库没有的记忆算法

只有这里才实现 [BaseMemoryMethod、MethodPlugin](../core/plugin.py)：把原算法接到公共事件、查询和经历上，声明参数、依赖、能力与学习边界。可以在独立插件模块实现，不要求修改原 Agent。

使用 `PluginRegistry.register(plugin)` 注册，再将同一个 `registry=` 传给 `Memory.check` 和 `Memory.from_dict`。Agent、场景和数据读取不注册为 Memory 插件。

接入新组合时可通过 `Memory.check(config, requirements=ProtocolRequirements(...))` 检查事件、模态、评价字段、冻结或恢复能力，定义见 [ProtocolRequirements](../core/contracts.py)。实际贡献形式和学习时机仍要核对，不兼容时明确报错。

本文的 `run_task`、文本适配器和读取函数均来自 quickstart 示例；公共 Memory 方法来自框架。未新增包级任务 API，也无需另外复制任务管理代码。
