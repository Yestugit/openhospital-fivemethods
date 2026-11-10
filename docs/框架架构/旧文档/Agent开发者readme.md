

# RiseMem：为你的 Agent 接入记忆

更新：接入说明与任务归属整理。——10.30

RiseMem 提供多种记忆方法，支持检索、交互记录、反馈学习和状态恢复。你可以沿用已有 Agent 的模型、提示词和工具流程。

## 1. 安装与选择方法

在包含 `pyproject.toml` 的项目根目录执行：

```bash
python -m pip install -e .
python -m mematlas methods list
python -m mematlas methods describe baseline_bm25
```

建议先用 `baseline_bm25` 验证接入。使用其他方法前，按其参数、可调用接口与使用条件清单准备配置、依赖和必要反馈。共性接口之外，各方法可提供特有接口。

## 2. 配置与创建

```python
from mematlas import Memory, MemoryScope

config = {
    "method": {
        "method_id": "baseline_bm25",
        "params": {"top_k": 3},
    },
    "runtime": {
        "artifact_dir": ".mematlas/my-agent",
    },
}
scope = MemoryScope(user_id="alice", agent_id="assistant")
```

| 配置 | 用途 |
|---|---|
| `method.method_id` | 选择记忆方法 |
| `method.params` | 方法特有参数 |
| `providers` | 记忆方法所需的 LLM、Embedding 服务 |
| `runtime.artifact_dir` | 运行产物与状态保存位置 |
| `MemoryScope` | 记忆归属，例如用户、Agent |

Agent 自己的模型继续在原 Agent 中配置。特殊模型客户端可通过 `MethodServices(llm=..., embedder=...)` 注入。

## 3. Agent 开发者可用接口

表中标注 `await` 的接口需要在异步函数中调用。

下表介绍框架公共能力，具体方法支持的接口和必需调用步骤以方法说明为准。完整使用清单待补齐，任务见[方法集成](../方法集成/项目任务与进展.md)。

**方法查询与初始化**

| 接口 | 用途 |
|---|---|
| `MemAtlas().methods()` | 查看已注册方法 |
| `MemAtlas().describe(method_id)` | 查看参数、能力、依赖与限制 |
| `Memory.check(config)` | 检查配置与依赖，返回包含 `ready` 的报告 |
| `Memory.from_dict(config)` | 创建 Memory，推荐通过 `async with` 使用 |

**交互与记忆使用**

| 接口 | 用途与返回 |
|---|---|
| `await memory.start_episode(scope=..., episode_id=..., learning=...)` | 开始一个任务回合 |
| `await memory.record_step(step)` | 记录真实消息、动作或工具结果 |
| `await memory.recall(query, scope=..., top_k=...)` | 按文本检索，返回 `MemoryContribution` |
| `await memory.retrieve(context)` | 使用 `DecisionContext` 提供更完整的查询条件 |
| `memory.apply_contribution(messages, contribution)` | 组织记忆增强输入，返回 `MemoryApplication` |
| `memory.confirm_application(contribution)` | 确认该记忆贡献已实际提供给 Agent |
| `await memory.call_model(model, messages, contribution=...)` | 应用记忆并调用模型，自动记录相关步骤与来源 |
| `await memory.call_tool(tool, name, arguments)` | 调用工具，自动记录动作与结果 |
| `await memory.finish_task(output=..., learn=False)` | 保存本回合轨迹，返回 `AgentTrace` |
| `await memory.submit_feedback(episode_id, scope=..., success=..., reward=..., feedback=...)` | 提交真实反馈并进入学习路径，返回 `EpisodeResult` |
| `memory.discard_episode()` | 放弃当前活动回合，清理局部记录 |

**历史导入、状态与扩展**

| 接口 | 用途 |
|---|---|
| `await memory.ingest_history(events, outcome=...)` | 导入已有历史，按方法要求提供评价 |
| `memory.record_episode(episode)` / `await memory.submit_episode(episode)` | 接入已有完整经历；分别用于记录、记录并提交学习 |
| `await memory.checkpoint(path)` | 保存方法状态与公共回合历史 |
| `Memory.resume(config, run_dir)` | 重新打开原运行 |
| `await memory.load_state(path_or_state)` | 加载状态或完整检查点 |
| `async with memory.read_only():` | 冻结学习，使用已有记忆执行任务 |
| `await memory.reset(scope)` | 清理指定范围的记忆 |
| `memory.stats()` | 查看运行状态、操作统计与资源用量 |
| `await memory.command(name, payload)` | 调用方法公开的特有命令 |
| `await memory.learning_boundary(boundary, ...)` | 按方法要求通知批次、epoch 等学习边界 |

## 4. 最小接入示例

下面使用 BM25。`agent_call` 是你已有的异步执行入口，接收消息列表，返回回答文本。

```python
from uuid import uuid4
from mematlas import AgentStep, EventKind, Memory

async def run(agent_call):
    report = Memory.check(config)
    if not report["ready"]:
        raise RuntimeError(report)

    async with Memory.from_dict(config) as memory:
        question = "Helios 的服务端口是多少？"
        task_id = uuid4().hex

        await memory.start_episode(
            scope=scope, episode_id=task_id, learning=False,
        )
        try:
            await memory.record_step(AgentStep(
                kind=EventKind.MESSAGE, role="user", text=question,
            ))
            contribution = await memory.recall(question, scope=scope)

            # 自动应用记忆、调用原执行入口并记录回答。
            answer = await memory.call_model(
                agent_call,
                [{"role": "user", "content": question}],
                contribution=contribution,
            )
            trace = await memory.finish_task(
                output=answer, metadata={"task": question}, learn=False,
            )
        except BaseException:
            memory.discard_episode()
            raise

        # BM25 无需成功标签；此处提交已记录的经历。
        await memory.submit_feedback(trace.case_id, scope=trace.scope)

        run_dir = memory.config.runtime.artifact_dir
        await memory.checkpoint(f"{run_dir}/checkpoint.json")
        print(trace.final_output)
```

多步 Agent 可以在自己的执行循环中多次检索，使用 `record_step()` 上报真实过程。工具调用可使用 `call_tool()` 包装；已自动记录的步骤无需再手动记录。

## 5. 关键返回数据

| 类型 | 读取什么 |
|---|---|
| `MemoryContribution` | `items` 为检索条目；还可能包含上下文、工具、模型钩子、附件或直接答案 |
| `MemoryApplication` | `messages` 为增强后的消息；其余贡献需按 Agent 能力接入 |
| `AgentTrace` | `case_id`、`scope`、`steps`、`final_output` |
| `EpisodeResult` | `execution`、`feedback`、`learning` 分别表示执行、反馈和学习状态 |

## 6. 使用约定

1. **跨回合复用：**保持同一个 Memory 和一致的记忆归属；每次任务使用独立 ID，一个 Memory 串行处理任务。
2. **重启恢复：**默认创建新运行。继续旧记忆时使用 `Memory.resume(config, run_dir)`，需要恢复检查点时再调用 `load_state()`。
3. **记录与学习：**`learning=False` 配合 `finish_task(learn=False)` 先保存轨迹，后续通过 `submit_feedback()` 提交；在线更新按方法说明开启。
4. **冻结测试：**使用 `read_only()`，其中不提交训练反馈。放弃回合仅清理局部记录，已发生的长期记忆更新仍会保留。
5. **方法切换：**按所选方法的使用清单调整配置、接口调用与 Agent 适配。框架不要求只改配置即可切换所有方法。
6. **使用条件：**运行前检查所需参数、输入、反馈与能力，按报错补齐缺失项；当前检查尚未覆盖全部条件。
