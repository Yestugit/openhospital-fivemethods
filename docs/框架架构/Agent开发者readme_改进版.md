# RiseMem：Agent 开发者 README（改进设计）

更新：2026-10-31。——10.31

本次补充：公共 Memory 接口统一为 `index`、`add`、`search` 和 `episode_end`；方法内部处理学习更新和自动演化。——10.31

本文记录已确认的目标用法，示例用于说明设计。[原版 README](Agent开发者readme.md) 保留重构前的用法供参考。

RiseMem 为已有 Agent 提供记忆。Agent 负责提示组装、模型调用、工具执行和业务评价；Memory 负责接收信息、检索、学习、演化和保存记忆。公共层保留原始事件、组织完整经历并检查必要条件，各方法保留自己的算法、存储和更新时机。

**当前可执行范围：**已接入 Memory 接口 v3：mem0、a-mem、memoryos、memrl、reasoningbank、expel、awm、memcon、ael。

> 原本：Memory 包装模型、工具调用及回合执行。改进后：Agent 自行执行，Memory 接收实际事件与经历。记忆方法内部用于提取和维护记忆的模型调用继续保留。

## Quickstart：记录并检索计划——11.01

使用 Mem0 记录一条计划，再检索相关记忆。以下示例使用公共 Memory 接口 v3。

在包含 `pyproject.toml` 的项目根目录安装：

```bash
python -m pip install -e ".[openai]" "qdrant-client>=1.12"
```

在运行脚本的同一个 PowerShell 窗口设置：

```powershell
$env:DASHSCOPE_API_KEY = "你的 Qwen API 密钥"
$env:OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

将以下代码保存为 `quickstart_agent.py`，执行 `python quickstart_agent.py`：

```python
import asyncio
from mematlas import Memory, MemoryEvent, MemoryUnit

config = {
    "method": {"method_id": "mem0", "params": {"hybrid_search": False}},
    "scope": {"user_id": "alice"},
    "providers": {
        "llm": {"model": "qwen-plus", "base_url": None, "api_key_env": "DASHSCOPE_API_KEY"},
        "embedding": {"model": "text-embedding-v4", "base_url": None,
                      "dimensions": 1024, "api_key_env": "DASHSCOPE_API_KEY"},
    },
}

async def main():
    async with Memory(config) as memory:
        event = MemoryEvent(
            "message", "我计划在12月15日去北京参加学术会议。", role="user"
        )
        receipt = await memory.add(MemoryUnit(events=[event]))
        await memory.episode_end(receipt.episode_ref)
        result = await memory.search("我计划什么时候去北京？")
        print([item["content"] for item in result.items])

asyncio.run(main())
```

输出包含“12 月 15 日去北京参加学术会议”的记忆。已有 Agent 可将检索结果加入自己的提示。

一个任务可以多次调用 `add()`。普通流程无需反复填写 `episode_ref`，运行时会自动关联当前任务。反馈晚到时，使用 `receipt.write_ref` 提交新的 `MemoryUnit(type="feedback", related_refs=(receipt.write_ref,), feedback=实际反馈)`。

`base_url=None` 从环境变量读取服务地址；`hybrid_search=False` 省去混合检索的额外依赖，仍使用真实模型提取记忆和向量检索。

## 1. 安装与配置

在包含 `pyproject.toml` 的项目根目录安装并查看方法：

```bash
python -m pip install -e .
python -m mematlas methods list
python -m mematlas methods describe mem0
```

目标配置示例：

```python
config = {
    "method": {"method_id": "mem0", "params": {"hybrid_search": False}},
    "scope": {"user_id": "alice", "agent_id": "assistant"},
    "runtime": {"artifact_dir": ".mematlas/my-agent"},
    "providers": {
        "llm": {"model": "qwen-plus", "base_url": None,
                 "api_key_env": "DASHSCOPE_API_KEY"},
        "embedding": {"model": "text-embedding-v4", "base_url": None,
                      "dimensions": 1024,
                      "api_key_env": "DASHSCOPE_API_KEY"},
    },
}
```

`method` 保存算法参数，`scope` 保存默认记忆归属，`runtime` 保存运行配置；方法需要模型时配置 `providers`。Agent 的模型继续在原 Agent 中配置。

> 原本：调用方构造 `MemoryScope`，在事件、查询和经历中传递。改进后：默认归属放进 config，普通调用自动使用；任务确需切换归属时覆盖，持久化时记录实际归属。

每个方法通过 `__init__.py` 作为包入口，导出配置类和必要的公共适配对象，并简述必要服务、数据与反馈、调用流程，可通过 `help(方法包)` 查看。`config.py` 提供配置类，`example.py` 展示最小接入流程；`method.yaml` 用于程序声明与校验。方法差异通过公共数据结构的 `extensions` 和方法配置表达，Agent 统一使用公共 Memory 接口。入口示例见 [Mem0](../../methods/mem0/__init__.py)、[MemRL](../../methods/memrl/__init__.py)、[MemoryOS](../../methods/memoryos/__init__.py)；逐方法要求见[方法书写约束](../方法集成/1031方法书写约束.md)。——10.31

使用前按模板补齐条件，缺失项由检查说明；切换方法时保留公共接口及原有调用位置，按要求补充配置与数据。逐方法清单的落实见[方法集成任务](../方法集成/项目任务与进展.md)。

## 2. 数据结构

按用途保留一种表达，结构数量随实际需要确定。配置外的主要交互数据如下：

| 结构 | 字段与用途 | 原本 → 改进后 |
|---|---|---|
| `MemoryEvent` | `event_id`、可选 `episode_id`、`timestamp`、`kind`、`role`、`content`、`extensions`；记录消息、动作、工具结果或观察 | `AgentStep` 与 `MemoryEvent` → 统一事件，普通流程无需填写学习单元 ID |
| `MemoryUnit` | `events`、`task`、`output`、`feedback`、`applied_memory`、`extensions`，以及按需提供的 `episode_ref`、`related_refs`；表示一次可独立提交的记忆单元 | Agent 通过 `add()` 提交，一个任务可包含多个单元 |
| `WriteReceipt` | `write_ref`、`episode_ref`；标识已提交单元及其所属任务 | 用于关联后续调用和延迟反馈 |
| `MemoryEpisode` | 运行时聚合的任务、事件、输出、评价、实际使用的记忆贡献及扩展数据 | 保留为运行时内部聚合和方法学习数据 |
| `MemoryContribution` | `items`、`context`、`extensions`；返回记忆内容及来源 | 检索输出再包装为 `MemoryApplication` → Agent 直接使用检索输出 |

`MemoryUnit` 表示一次可独立提交的记忆单元，可以包含事件、任务、输出、反馈、实际使用的记忆贡献和方法扩展数据。一个任务可以包含多个 `MemoryUnit`，运行时将相关单元关联到同一个任务。`MemoryEpisode` 保留为运行时内部聚合和方法学习数据。批次、epoch 等更新时机遵循方法原生规则。——10.31

查询直接传参数。内容与记忆条目采用固定嵌套格式，无需额外构造 `DecisionContext`、`ContentBlock`、`MemoryItem`：

1. **内容：**文本直接用字符串；图文按共用内容格式提供，方法说明列出要求。
2. **条目：**`items` 中保存 `memory_id`、`content`、可选 `score` 和来源信息。`context` 按需保存方法整理后的内容及来源关联，普通结果避免重复保存相同文本。
3. **扩展：**`extensions` 按方法组织专有数据，如 AEL 的工具评价和策略记录、M2A 的直接答案。用户提供真实业务数据，方法产生的关联信息由框架保留并传回学习流程。
4. **实际应用：**`applied_memory` 为实际使用的 `MemoryContribution` 列表，多次检索分别记录；保留其中的内容、来源和方法关联信息。原本只传 `items`，改进后回传完整贡献，避免丢失策略数据。——10.30

## 3. 数据如何填写

以下为目标格式。用户提供事件、经历信息与真实反馈，公共层按需补齐已记录的轨迹；检索输出由框架返回。——10.30

| 数据 | 必填内容与默认行为 |
|---|---|
| `config` | 按第 1 节示例填写，所选方法必需的参数与服务按其使用清单补齐 |
| `MemoryEvent` | 提供 `kind`、`content`；消息填写 `role`，事件 ID 和时间默认生成，通过 `MemoryUnit.events` 提交，普通流程无需填写任务引用 |
| `MemoryUnit` | 按当前提交内容填写 `events`、`task`、`output` 或 `feedback`，`applied_memory` 默认空列表；`episode_ref` 通常省略，由运行时关联当前任务 |
| `WriteReceipt` | 读取 `write_ref` 和 `episode_ref`，用于任务结束标记或后续反馈关联 |
| `MemoryEpisode` | 由运行时按相关单元聚合，供方法内部学习使用 |
| `MemoryContribution` | 读取 `items`、可选的整理内容 `context` 和 `extensions`；实际应用后按完整贡献记录到经历中 |
| `extensions` | 默认空字典，方法专有数据按方法名组织；字段和必需条件见该方法清单 |

消息与工具结果示例：

```python
message = MemoryEvent(kind="message", role="user", content="查询 Helios 端口")
tool_result = MemoryEvent(
    kind="tool_result", role="tool", content="Helios 的端口为 8443",
    extensions={"tool": {"name": "read_config", "arguments": {"service": "Helios"}}},
)
```

文本内容直接用字符串；图文内容拟采用 `[{"type": "text", "text": "问题"}, {"type": "image", "uri": "image.png"}]`，由所选方法声明是否支持。

检索返回的条目格式示例，用户无需自行构造：

```python
# contribution.items 中的一条记忆
{"memory_id": "m1", "content": "Helios 的端口为 8443",
 "score": 0.9, "source_event_ids": ["e1"]}
```

记忆单元示例见第 5 节。普通流程无需填写 `episode_ref`，运行时会自动关联当前任务；需要跨调用或提交延迟反馈时，使用 `WriteReceipt.write_ref`，必要时再显式提供 `episode_ref`。方法要求结果标签时，在 `MemoryUnit.feedback` 中补齐真实的 `success`、`reward` 或 `text`；反馈晚到时提交关联原写入的新单元。专有数据例如 `extensions={"ael": {"tool_evaluations": actual_evaluations}}`；检索产生的策略信息随贡献保留，无需用户编造。

## 4. 可调用接口

下表为公共 Memory 接口；标注 `await` 的调用在异步函数中使用。`index()` 查看方法提供的记忆层级或访问视图；`add(unit)` 提交一个记忆单元，一个任务可以多次提交；`search(query)` 按当前需求检索记忆；`episode_end()` 标记一次 Agent 任务结束。方法内部负责即时写入、缓冲、去重、反馈学习、批次更新和自动演化。——10.31

`detail`、`update`、`delete`、`flush`、`save`、`load`、`reset`、`read_only` 和 `stats` 属于公共运行时能力，是否支持由方法能力声明决定。`evolve` 不作为 Agent 公共接口；方法能够依据累计次数、阈值或内部状态自动演化时，由方法内部触发。

方法内部处理可以不同。例如，Mem0 可在收到记忆单元后即时写入，MemRL 可先缓冲相关单元，在取得完整轨迹和真实反馈后更新；Agent 继续按相同用途和位置调用公共接口。——10.31

| 目的 | 接口 | 原本 → 改进后 |
|---|---|---|
| 方法查询 | `MemAtlas().methods()` / `MemAtlas().describe(method_id)` | 查看已注册方法、参数、能力与条件 |
| 检查 | `Memory.check(config)` | 保留入口，检查方法所需配置与条件 |
| 创建与关闭 | `async with Memory(config) as memory` | 多种创建包装 → 一个构造入口 |
| 记忆视图 | `await memory.index()` | 查看方法提供的记忆层级或访问视图 |
| 提交记忆单元 | `await memory.add(unit, episode_ref=None, scope=None)` | 提交事件、任务、输出、反馈或完整记忆单元，返回 `WriteReceipt` |
| 检索 | `await memory.search(query, level_ref=None, top_k=None, scope=None, extensions=None)` | 按当前需求执行方法原生检索，返回 `MemoryContribution` |
| 任务结束 | `await memory.episode_end(episode_ref=None, status="completed", scope=None)` | 标记任务边界，方法按自身条件完成学习或更新 |
| 查看详情 | `await memory.detail(...)` | 方法声明支持时查看记忆详情 |
| 更新记忆 | `await memory.update(...)` | 方法声明支持时更新记忆 |
| 删除记忆 | `await memory.delete(...)` | 方法声明支持时删除记忆 |
| 提交缓冲处理 | `await memory.flush()` | 方法声明支持时处理缓冲，无需在每次 `add()` 后调用 |
| 保存／恢复 | `await memory.save(path)` / `await memory.load(path)` | 状态与检查点包装 → 一套保存／恢复入口 |
| 冻结更新 | `async with memory.read_only():` | 保留记忆保护，Agent 在外部执行 |
| 清理 | `await memory.reset()` | 仅方法声明支持时可用 |
| 统计 | `memory.stats()` | 查看操作与已采集的资源用量 |

**易错点：**——10.31

1. `flush()` 仅在方法需要处理缓冲时使用；`add()` 后无需一律调用。即时写入或自动处理缓冲的方法可直接继续使用，具体要求见方法清单。
2. 方法能够根据累计次数、阈值或内部状态自动判断演化时，由内部触发；演化逻辑放入 `add()`、`episode_end()` 或内部更新流程。
3. `save()` 用于保存当前记忆与必要运行状态，`load()` 用于恢复；学习与更新时机由方法内部决定。
4. 方法切换时，Agent 继续使用同一组公共接口和原有调用位置。额外信息通过配置、`MemoryUnit.extensions` 或 `search(..., extensions=...)` 补充。
5. `episode_end()` 只标记任务边界，不推断成功、奖励或学习结果；真实评价通过 `MemoryUnit.feedback` 提交。

**记忆归属默认沿用 `config.scope`；需要切换且方法支持时，可通过调用参数 `scope` 覆盖。**`scope` 确定记忆归属，`episode_ref` 关联任务；延迟反馈沿用原单元的实际归属。

1. **记忆单元与任务衔接。**`add()` 默认由运行时自动关联当前任务。需要跨调用或提交延迟反馈时，使用 `WriteReceipt.write_ref` 或显式提供 `episode_ref`。
2. **反馈按需提供。**无需评价的方法可以直接处理经历；需要评价的方法取得真实反馈后提交。已处理事件避免重复写入，完整轨迹继续保留。
3. **提交与更新分工。**`add()` 提交记忆单元，单元可以包含事件、任务、输出、反馈、实际使用的 `MemoryContribution` 和方法所需的 `extensions`。方法根据自身规则决定何时完成学习或更新。
4. **方法差异数据。**检索参数传入 `search(..., extensions=...)`，提交数据放入 `MemoryUnit.extensions`，检索产生的方法关联信息保存在 `MemoryContribution.extensions`。稳定参数放入 `method.params`，方法切换时继续使用同一组公共接口。

Agent 使用 `search()` 获取记忆，使用 `add()` 提交事件、输出、反馈或完整记忆单元，使用 `episode_end()` 标记任务结束。方法内部负责即时写入、学习更新、批次处理和自动演化。——10.31

## 5. 接入示例

以下是 Mem0 文本问答用法，复用第 1 节的 config。`agent_call(messages)` 为已有异步 Agent 入口，返回文本；本例将全部检索条目提供给 Agent。

```python
from mematlas import Memory, MemoryEvent, MemoryUnit


async def run(agent_call, question):
    report = Memory.check(config)
    if not report["ready"]:
        raise RuntimeError(report)

    async with Memory(config) as memory:
        events = [MemoryEvent(kind="message", role="user", content=question)]
        user_event = events[0]
        contribution = await memory.search(question, top_k=3)
        memory_text = "\n".join(item["content"] for item in contribution.items)

        # 原 Agent 自行执行；保留自己的提示、历史和工具流程。
        messages = [
            {"role": "system", "content": f"相关记忆：\n{memory_text}"},
            {"role": "user", "content": question},
        ]
        answer = await agent_call(messages)
        assistant_event = MemoryEvent(kind="message", role="assistant", content=answer)
        events.append(assistant_event)

        receipt = await memory.add(MemoryUnit(
            task=question,
            events=events,
            output=answer,
            applied_memory=[contribution],
        ))
        # Mem0 无需结果标签；其他方法按要求在 feedback 中补齐真实反馈。
        await memory.episode_end(receipt.episode_ref)
        await memory.save("memory-state.json")
        return answer
```

首次运行可以没有历史记忆；继续旧记忆时在检索前 `await memory.load("memory-state.json")`。连续任务复用同一个 Memory，在循环外创建和关闭。

单条资料也可以作为一个记忆单元直接提交；以下适用于支持此粒度且无需结果标签的方法，在已创建的 `memory` 中调用：

```python
receipt = await memory.add(MemoryUnit(
    events=[MemoryEvent(kind="observation", content="Helios 的端口为 8443")],
))
await memory.episode_end(receipt.episode_ref)
```

普通流程无需填写 `episode_ref`，运行时会自动关联当前任务。反馈晚到时，使用先前返回的 `receipt.write_ref` 关联原单元：

```python
await memory.add(MemoryUnit(
    type="feedback",
    related_refs=(receipt.write_ref,),
    feedback={
        "success": True,
        "reward": 1.0,
        "text": "外部评价已返回",
    },
))
```

`evolve` 不作为 Agent 公共接口。方法能够依据累计次数、阈值或内部状态判断演化时，由方法内部触发；演化逻辑放入 `add()`、`episode_end()` 或方法内部更新流程。

> 原本：通过 Memory 开始回合、转换步骤、包装 Agent 调用、结束回合并提交反馈。改进后：Agent 自行执行，通过 `search()` 获取记忆，通过 `add()` 提交记忆单元，通过 `episode_end()` 标记任务结束；方法在内部实现公共约定，并按自身规则处理学习与演化。——10.31

## 6. 使用约定

1. **保留真实过程。**事件 ID 保持稳定。普通流程通过 `MemoryUnit` 提交实际事件，运行时自动关联当前任务；跨调用关联或延迟反馈使用 `WriteReceipt.write_ref`，必要时再显式提供 `episode_ref`。
2. **保留实际来源。**`applied_memory` 保存实际使用的完整贡献；裁剪 `items` 或 `context` 时同步保留准确来源及方法关联信息，候选结果不全部视为已使用。
3. **遵循方法更新规则。**运行时关联同一任务的记忆单元，方法内部按原生规则完成即时写入、去重、缓冲、学习和批次／epoch 更新。`episode_end()` 标记任务边界。
4. **使用真实反馈。**需要标签的方法在 `MemoryUnit.feedback` 中补齐真实的 `success`、`reward` 或 `text`；反馈晚到时，通过 `related_refs` 关联原单元后提交，不虚构标签。
5. **冻结测试。**通过 `read_only()` 检索并执行 Agent，冻结期间禁止 `add()` 和其他记忆更新。方法能够自动判断演化时，由方法内部处理；Agent 不调用演化入口。具体方法需支持冻结与所需恢复能力。

评测侧继续保留 `BenchmarkCase`、`Prediction`、`EvaluationResult` 及必要的 `EpisodeResult`、`BatchResult`，分别用于任务、评分、汇总与进度记录。`EpisodeOutcome` 的评价字段合并进经历，`EpisodeResult` 的处理状态保留。共享接口改动需同步迁移 Benchmark 和 OPH，保留案例隔离、反馈、冻结与恢复行为。——10.30



## 7.改进需求日志【用这个参考修改思想】

### 10.31
原句：【原本昨天我是想每个方法的实现可以有差异，但是用途实际上要保持一致，不能说每个memory方法一个用途，或者说这种差异要是用户无感的，用户反正通用方法都是这么写，你如何使用倒是没关系，但是不能说我这个方法obs是这样用的，不需要learn，那个方法obs只是存一下，需要learn，那用户还要自适应的修改，那不好用的。我的意思是，给用户要是一个明确的使用方法【 用户得到一套明确的使用约定，各方法负责在内部实现这套约定。  】，至于传入之后不同memory方法如何使用可以有差异（我也不知道这个合理吗？） 通用的方法的用途最好保持一致，不能依据每个方法依据不同memory都不同，这样用户没法用。然后通用的函数最好在变更memory方法的时候，最好通用的方法的函数不需要变（我指的是用户接入的时候，例如在这个位置observe、那个位置learn，那么切换方法的时候这个位置不需要变，），变的就是不同memory方法特有的通用方法的数据结构或者接入新的方法或者添加新的方法位置或者方法啥的，就是基本上原始接入后，只需要微调一下或者不用微调就能够适应新的方法，要达到这种效果
2、不同方法最好不是必要的话不要给出特有的方法，确实需要特有的方法的时候，再给出，这个特有的方法最好数量要保持少，这样才方便用户使用。另外，特有方法指的是我们原始暴露给用户agent使用memory系统的接口，不足以支撑这个memory方法完成的时候，不得不新建的时候，现有通用接口确实不可以包住的时候才用。而不是给他作为补充新的功能！】

：给 Agent 开发者一套稳定、明确的记忆使用方式。切换 Memory 方法时，用户尽量沿用原来的调用位置；方法自己适配算法差异，用户只在必要时补充少量数据或调用。
1. **公共接口的用途固定。**一个公共接口，给用户的要是一个明确的使用方法，用户得到一套明确的使用约定，各方法负责在内部实现这套约定。Agent 在需要写入记忆的交互事件发生时调用 observe(event)，需要记忆时调用 retrieve(query)，一个学习单元完成时调用 learn(episode)。evolve按需书写，不一定都有，有独立演化机制的方法，按其规定的周期或条件调用 evolve，演化一段时间内积累的记忆、经验或策略。  这些接口对不同方法含义相同，用户不应因为换了方法，就要判断“这个方法只用 observe，那个方法还得用 learn”，再重写流程。
2. **方法内部处理可以不同。**例如 Mem0 可以在收到助手回复后立即写入；MemRL 可以先记录事件，等 learn(episode) 收到完整轨迹和真实反馈后再更新。两种方法的内部机制可以不同，Agent 仍按同一个位置调用 observe 和 learn。方法负责处理即时写入、缓冲、去重或批次更新。
3. **切换方法时，尽量保留 Agent 原有流程。**方法可能需要不同的数据字段、反馈或事件类型，用户按清单补充即可；确实有必要时，可以增加少量调用或做小幅调整。目标是原有接入经过少量适配或无需适配，就能使用新方法。
4. **evolve 的公开条件。**只有方法确实要求 Agent 或外部流程主动触发演化时，才声明支持并提供公共 `evolve()`。无需外部触发的自动演化方法，仅保留内部演化逻辑，不公开 `evolve()`，也不额外提供可选手动触发入口。冻结评测本身不触发演化。——10.31
5. **特有接口要少且必要。**只有现有公共接口和数据结构无法承载方法的核心操作时，才增加特有接口，并说明它为什么必需。特有接口用于完成方法本身的核心记忆机制，不作为额外功能的扩展入口。  按照这个修改吧，修改文档的时候，不要给我加我没有说过的内容
