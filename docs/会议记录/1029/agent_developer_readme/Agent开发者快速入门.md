# Agent 开发者快速入门

本文先列用户需要准备的数据、已有 Agent 和 Memory 配置，再说明按需适配的内容、拟开放接口及已经可以调用的框架接口，最后给出使用流程和完整示例。my_agent 是演示函数，不是用户必须实现的接口；待实现能力单独标明。

依据源码 `72d3f62`，更新于 2026-10-28。本篇用于在已有 Agent 中接入记忆工具箱，不展开 Benchmark 评分和 OpenHospital。想先验证环境，可直接运行第 4 节。

## 1. 用户需要修改和接入的内容

### 1.1 用户准备数据、已有 Agent 和 Memory 配置

你需要提供已有的 Agent 和本次业务数据，再选择 Memory 方法。**框架不要求用户实现名为 my_agent 的公开接口，也不要求重写原 Agent。** my_agent 只是第 4 节演示函数的名称。

| 你提供什么 | 具体内容 | 是否必需 |
|---|---|---|
| 已有 Agent | 已配置好的模型、提示、工具与执行流程；接入时提供实际执行入口 | 必需 |
| 本次任务数据 | 当前问题、原提示和必要对话历史；有其他模态时提供实际内容 | 必需 |
| Memory 配置 | 方法 ID、参数、保存目录；按方法要求准备服务和依赖 | 必需 |
| 记忆归属 scope | 用户与 Agent 身份；同一归属后续保持一致 | 必需 |
| 真实执行过程和评价 | 工具动作、结果、成功标签、奖励等 | 所选方法需要时提供 |

在第 4 节脚本的 main() 中修改 config 和 scope，例如：

```python
config = {
    "method": {"method_id": "baseline_bm25", "params": {"top_k": 3}},
    "runtime": {"artifact_dir": ".mematlas/my-agent"},
}
scope = MemoryScope(user_id="alice", agent_id="assistant")
```

本次任务到达时准备 question 和完整 messages，再传入执行流程。第 4 节的 questions 循环是两项演示任务，实际业务应使用真实请求；仅在文件顶部定义变量不会改变 main() 内的配置或输入。

换 Memory 时修改方法 ID 和其必要配置，再创建新实例。Agent 自己的模型仍在原 SDK 中配置；记忆侧服务配置见第 5.6 节。方法的事件、模态、反馈与输出能力需兼容，不承诺只改 ID 就支持所有组合。

### 1.2 当前示例怎样接入已有 Agent：兼容时直接传入

**先找原 Agent 的实际执行入口，只有输入输出不兼容时才写转换代码。** 当前第 4 节的临时 run_task 接收 execute 参数，要求它是可被 await 的调用入口：输入消息列表，返回最终文本。

| 原 Agent 的情况 | 当前示例中怎么处理 | 是否需要新增 my_agent |
|---|---|---|
| 已有异步函数，接收 messages 并返回 str | 将该函数直接传给 execute | 不需要 |
| Agent 对象的某个实际方法符合上述要求 | 将已绑定到该对象的方法传给 execute；方法名以真实 SDK 为准 | 不需要 |
| 对象可异步调用，输入输出符合要求 | 将该可调用对象直接传给 execute | 不需要 |
| 参数、消息格式或返回结构不兼容 | 按真实 SDK 做转换，将兼容的执行入口交给 execute | 需要适配时再写函数，名称可自行决定 |
| 只有同步执行入口 | 本文临时 run_task 会 await execute，不能直接传同步函数；需按实际执行环境适配 | 按需，不强制使用特定名称 |

适配代码只负责调用原 Agent、转换输入并提取真实回答，失败时抛出真实异常。工具注册、推理循环和模型配置继续沿用原 Agent；框架不会按任意对象自动猜测执行方法或回答字段。

**当前示例不能直接接收任意不可调用的 Agent 对象。** 正式 run_task 拟支持 Agent 对象或执行入口的接入，尚未实现，见第 1.4 节；不要把该设计目标当成当前示例的能力。

第 4 节用 my_agent 充当演示执行入口。接入真实业务时，可直接将 execute=my_agent 替换为已有兼容入口；不必先改写一份同名函数。如果选择修改演示函数体作为适配，也只是可选做法。执行由 run_task 内部触发，不要在外部先重复调用一次。

### 1.3 按需接入的真实工具和评价

| 何时需要 | 用户要改什么 | 使用什么已有接口 |
|---|---|---|
| 方法需要工具过程，且允许包装工具调用 | 提供真实工具 callable、名称、输入 | `memory.call_tool()`，自动记录动作和结果 |
| 工具已经由原 Agent 执行 | 在实际调用或执行钩子中取出真实动作与结果 | `memory.record_step()`，只补需要的过程 |
| 方法要求成功、奖励或文字反馈 | 接入业务评价器或人工评价，保存原任务 ID | `memory.submit_feedback()` |
| 方法返回工具、模型钩子或附件 | 修改示例中对 MemoryApplication 的处理，按原 Agent 机制接入这些字段 | `apply_contribution()` 保留完整贡献供你处理 |

工具自动记录和手动记录针对同一次调用选择一种。没有真实评价时不要填写虚构的 success 或 reward；BM25 示例不要求标签。单纯返回最终字符串不会自动带出原 Agent 内部工具过程。

### 1.4 拟开放的任务执行与记忆集成接口：run_task（未实现）

**状态：正式框架接口尚未实现，需由框架开发者封装，用户不负责实现。** 当前第 4 节只有同名的临时文本示例函数。本节描述拟开放接口的功能和传参需求，不提供实现代码，也不承诺下面的参数已可调用。

**接口用途：**组织一次完整任务，将已配置的 Memory 与用户已有的 Agent 连接起来。不同场景通过任务输入、执行器、适配方式和策略等参数复用同一入口，不要求每个场景重写任务管理流程。

**用户需要传入的内容：**

| 参数内容 | 是否需要 | 用途 |
|---|---|---|
| 已配置的 Memory | 必需 | 指定本次使用的记忆方法与运行实例；换记忆方法时更换此配置或实例 |
| 用户与 Agent 身份 | 必需 | 明确本次记忆读写的归属范围 |
| 任务输入与原有消息 | 必需 | 提供真实问题、原提示和必要历史；多模态内容按执行器与方法能力适配 |
| 已有 Agent 对象或实际执行入口 | 必需 | 提供用户已经准备好的 Agent；兼容时直接绑定，格式不兼容时才补输入输出适配，不要求另写 my_agent |
| 业务任务 ID | 可选 | 关联业务请求与延迟反馈；未提供时由框架生成并随结果返回 |
| 记忆检索与应用策略 | 按需 | 简单任务执行前检索；多步任务允许执行器在运行中按需使用本任务的记忆能力 |
| 输入输出与记忆贡献的适配方式 | 按需 | 仅在需要时转换原 Agent 的输入、返回及记忆贡献；对工具、钩子或附件等不兼容能力明确报错 |
| 真实步骤上报方式 | 按需 | 通过执行器或执行钩子上报消息、工具动作及结果；需要完整轨迹的方法必须补齐真实过程 |
| 学习与反馈策略 | 按需 | 指定在线观察、任务结束后学习、冻结执行或等待反馈；选择与方法能力兼容的策略 |

用户主要提供数据、已有 Agent 和 Memory 配置。需要特殊转换或步骤采集时，再按需提供适配函数或执行回调，而不是所有用户必须先实现新的 Agent 包装接口。具体支持的 Agent 对象、执行方法绑定规则、参数名称、默认值和调用签名待实现时确定；当前不承诺自动识别任意 SDK。

**接口应完成的功能：**

1. 建立任务 ID、记忆归属和任务边界，并检查输入与方法能力是否兼容。
2. 按传入策略组织记忆检索与输入适配；向多步执行器提供本任务内按需检索、应用与上报步骤的能力。
3. 通过已确认的执行入口调用用户已有 Agent，按需应用转换与步骤采集，收集实际提供的记忆来源，保存最终结果并结束任务。
4. 失败或取消时清理活动任务，保留已知的执行与学习状态；不将清理当作已写入记忆的自动回滚。
5. 返回任务 ID、最终结果和已记录轨迹，供用户展示结果、查询状态或后续提交反馈。

工具选择、推理循环和真实业务执行由传入的 Agent 完成。接口不猜测成功或奖励，也不自动推断 Agent 内部未上报的工具过程；真实评价稍后到达时，仍可按原任务 ID 单独提交反馈。现有低层 Memory 接口继续保留，供自定义流程使用。

在正式接口实现前，第 3—4 节继续使用可复制的同名示例函数。它要求兼容的异步 execute，固定学习时机并主要记录文本问答；可以直接传已有兼容入口，但不具备本节全部拟开放能力。用户仍需保存示例接线代码，现阶段不能只准备任意 Agent 对象就自动完成集成。

## 2. 框架已经开放、可以直接调用的接口

本节中的接口由框架提供，用户**调用即可，不修改它们的实现**。当前使用第 4 节的示例 run_task 时，其中一部分已被内部调用；下面会明确标出。第 1.4 节拟开放的正式接口不属于本节现有 API。

### 2.1 选方法、检查与打开 Memory

| 用户需要的功能 | 现有接口 | 传入什么 | 得到什么或完成什么 |
|---|---|---|---|
| 查看可选方法 | `atlas.methods()` | 无参数，atlas 先由 MemAtlas() 创建 | 全部已注册方法的说明列表 |
| 查看某个方法 | `atlas.describe(method_id)` | 方法 ID | 该方法参数、依赖、能力和限制 |
| 检查配置 | `Memory.check(config)` | 配置 | 报告，关注 ready 和 errors；不代替真实服务连通性验证 |
| 新建并打开 | `async with Memory.from_dict(config) as memory` | 配置 | 初始化实例，退出上下文自动关闭 |

未安装当前源码包时，先按第 4 节保存脚本，然后在脚本所在目录运行：

```powershell
python .\quickstart_agent.py --list-methods
python .\quickstart_agent.py --describe-method baseline_bm25
```

这两个选项调用现有方法发现接口，不运行 Agent、不调用模型。已知方法和配置时可以跳过发现，直接检查并创建。一个 Memory 实例绑定一种方法；切换时新建实例，不会自动迁移另一方法的原生数据库。

### 2.2 在任务中使用记忆

| 用户需要的功能 | 现有接口 | 传入什么、返回什么 | 已用 run_task 时怎么办 |
|---|---|---|---|
| 开始任务 | `await memory.start_episode(scope=..., episode_id=..., learning=...)` | 身份、独立任务 ID、学习策略；建立当前任务 | 已调用，不再额外开始 |
| 记录真实步骤 | `await memory.record_step(step)` | AgentStep；保存步骤，按策略交给方法观察 | 问题和最终回答已记录；只补实际缺少的中间步骤 |
| 检索 | `await memory.recall(query, scope=...)` | 查询；返回 MemoryContribution | 已调用，不重复同次检索流程 |
| 组织 Agent 输入 | `memory.apply_contribution(messages, contribution)` | 原消息与贡献；返回 MemoryApplication | 已调用；特殊输出在函数内适配 |
| 登记实际提供的记忆 | `memory.confirm_application(contribution)` | 实际提供的贡献；记录来源 | 已调用，不重复确认 |
| 结束任务 | `await memory.finish_task(output=..., learn=False)` | 最终结果；返回 AgentTrace | 已调用，不再结束一次 |
| 放弃失败任务 | `memory.discard_episode()` | 无参数；清理活动任务 | 异常和取消分支已调用 |

`MemoryApplication.messages` 是已加入文本记忆的消息；还可能有 direct_answer、tools、model_hooks、artifacts。示例采用直接答案时跳过业务执行，对未实现的结构化输出明确报错。实际裁剪贡献时也要同步保留来源关联。

`learning=False` 使默认步骤先保存在任务轨迹中，反馈提交时再进入方法；它不是全局冻结。需要原生在线观察时按方法说明改为 learning=True。一个 Memory 当前只能有一个活动任务，不要交错处理并发任务。

### 2.3 Agent 的模型与工具调用辅助

这两个接口同样已由 Memory 提供，**不是用户需要另外实现的 Agent 方法**。

| 用户需要的功能 | 现有接口 | 框架替你完成什么 | 什么时候选 |
|---|---|---|---|
| 包装一次模型调用 | `await memory.call_model(model, messages, contribution=...)` | 应用贡献、确认来源、执行 callable、记录返回步骤和追踪；返回 callable 原始结果 | 自己组织执行流程时选用；普通 run_task 路径可跳过 |
| 包装一次工具执行 | `await memory.call_tool(tool, name, arguments)` | 调用真实工具，记录动作与成功结果及追踪；返回实际工具结果 | 希望框架自动记录某次工具调用时 |

call_model 的 model 可以是同步或异步 callable，包装层不猜 SDK 回答字段。它不自动检索或管理整次任务；选它时替换 run_task 中对应的手动应用、执行和记录代码，不能对同次模型调用叠加记录。

call_tool 的 tool 必须是实际 callable，name 是记录名称而非注册表查询键。arguments 为字典时展开为关键字参数，为 None 时不传业务参数，其他值作为一个位置参数。它不选择工具，也不把工具结果自动加回模型消息。完整调用片段见第 5.1—5.3 节。

### 2.4 写入已有信息、提交经历和反馈

| 用户需要的功能 | 现有接口 | 传入什么、完成什么 |
|---|---|---|
| 直接记录一条信息 | `await memory.record(text, scope=...)` | 文本、归属和可选角色；交给方法观察，不自动结束任务 |
| 导入已有历史 | `await memory.ingest_history(events, outcome=...)` | MemoryEvent 列表与方法要求的结果；按方法策略导入 |
| 提交任务反馈与学习 | `await memory.submit_feedback(episode_id, scope=..., ...)` | 原任务 ID、真实评价；返回 EpisodeResult |
| 登记外部完整经历 | `memory.record_episode(episode)` | 已构造的 MemoryEpisode；先登记，不提交学习 |
| 直接提交外部完整经历 | `await memory.submit_episode(episode)` | 完整经历与结果；登记并提交，替代逐步任务路径 |

本文主路径使用 run_task 和 submit_feedback。不要再对同一次经历额外调用 submit_episode 或底层 learn。反馈晚到时先保留任务 ID、scope、原配置与实际运行目录，再恢复原运行提交。

### 2.5 查看状态、保存恢复与管理

| 用户需要的功能 | 现有接口 | 传入什么、完成什么 |
|---|---|---|
| 查询某次任务 | `memory.episodes.result(episode_id, scope=...)` | 任务 ID；返回 EpisodeResult，含 execution、feedback、learning |
| 查看整个运行 | `memory.stats()` | 无参数；返回运行状态、操作与可获取用量 |
| 提交方法缓冲 | `await memory.flush()` | 方法支持 commit 时提交缓冲，不等于任务学习 |
| 保存完整检查点 | `await memory.checkpoint(path)` | 路径；保存方法状态和公共任务历史，当前任务先结束或放弃 |
| 打开旧运行 | `Memory.resume(config, run_dir)` | 原配置、实际运行目录；用 async with 初始化 |
| 恢复某个检查点 | `await memory.load_state(path)` | 检查点；回到保存时状态，可能替换后续更新 |
| 临时冻结更新 | `async with memory.read_only():` | 方法需支持冻结；保留记忆，限制更新，退出后解除限制 |
| 清理记忆 | `await memory.reset(scope)` | 指定归属；请求清理记忆及对应公共任务历史，省略 scope 则请求全部清理 |
| 关闭资源 | `await memory.close()` | 显式关闭；async with 退出时已自动处理 |

冻结不会清空记忆；清理不会在退出代码块后自动恢复。只读期间不能 reset 或提交训练反馈。任务查询与整体统计不同；未知任务查询可能返回 execution="pending" 的默认结果，不代表已确认该任务正在执行。未追踪用量不代表零。

### 2.6 返回数据怎样读取

这部分是数据字段，不是要用户实现或再次调用的接口。

| 返回数据 | 用户读取的字段 |
|---|---|
| AgentTrace（run_task 的返回值） | final_output：回答；case_id：任务 ID；steps：已记录过程；retrieved：登记的记忆来源 |
| submit_feedback 返回 EpisodeResult | execution：执行状态；feedback：反馈状态；learning：学习状态；error：错误信息 |
| recall 返回 MemoryContribution | 方法输出，交给 apply_contribution；不要只处理 items 而忽略整理后的 context 或其他贡献 |

completed 表示执行结束，submitted 表示经历已提交，都不代表回答正确或效果提高。

## 3. 正常业务按这个顺序使用

当前先复制第 4 节的 run_task 和导入代码。再将下面片段放进异步函数，使用第 1.1 节准备的 config、scope。my_agent 仅用于使演示可运行；实际接入时将 agent_execute 直接换成第 1.2 节确认的已有兼容执行入口，只有不兼容才添加转换函数。

```python
report = Memory.check(config)
if not report["ready"]:
    raise RuntimeError(f"记忆环境未就绪：{report}")

async with Memory.from_dict(config) as memory:
    question = "Helios 的服务端口是多少？"
    messages = [
        {"role": "system", "content": "你是服务运维助手。"},
        {"role": "user", "content": question},
    ]
    agent_execute = my_agent  # 演示值；实际替换为原 Agent 的兼容执行入口
    trace = await run_task(
        memory, scope, question, execute=agent_execute, messages=messages,
    )
    print(trace.final_output)
    # BM25 示例不要求标签；其他方法按真实评价补充反馈字段。
    result = await memory.submit_feedback(trace.case_id, scope=scope)
    print(result.execution, result.learning)
    run_dir = Path(memory.config.runtime.artifact_dir)
    await memory.checkpoint(run_dir / "checkpoint.json")
```

需要延迟反馈时，暂不执行提交那一行，保留原任务 ID 后再提交。下一任务复用打开的 Memory；不要每条消息都新建实例。默认新建会产生新的运行，即使 artifact_dir 相同也不自动继续旧记忆。

## 4. 先复制运行这份完整示例

### 4.1 文件放在哪里

当前仓库就是 Python 包源码，目录名为 MemAtlas，而导入名是小写 mematlas，尚无根目录正式安装材料。示例包含本地源码加载，无需设置 PYTHONPATH、PYTHONCASEOK 或重命名目录。

```text
D:\Agent-RSI\
├── quickstart_agent.py
└── MemAtlas\
    ├── __init__.py
    ├── quickstart_agent.py
    └── runtime\
```

下面代码与两份本地脚本保持一致，也支持将脚本放在源码根目录。先完整运行演示；接入时填写配置和数据，将 execute 绑定到已有兼容入口，按需做转换。my_agent 只负责演示，不是新增必实现项。其他代码块按需选用，不要全部依次重复执行。

### 4.2 完整代码

```python
import argparse
import asyncio
import importlib.util
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
import sys
from uuid import uuid4

# 当前源码目录叫 MemAtlas，而 Python 包名是 mematlas。
# 未安装时，从脚本旁的源码目录加载；不修改全局 Python 环境。
if importlib.util.find_spec("mematlas") is None:
    script_dir = Path(__file__).resolve().parent
    package_dir = next((
        path for path in (script_dir / "MemAtlas", script_dir)
        if (path / "__init__.py").is_file()
        and (path / "runtime" / "memory.py").is_file()
    ), None)
    if package_dir is None:
        raise ModuleNotFoundError(
            "未找到 mematlas。请将脚本放在 MemAtlas 源码目录的父目录，"
            "或先安装可导入的 mematlas 包。"
        )
    spec = importlib.util.spec_from_file_location(
        "mematlas", package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载本地 mematlas：{package_dir}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["mematlas"] = package
    spec.loader.exec_module(package)

from mematlas import AgentStep, AgentTrace, EventKind, MemAtlas, Memory, MemoryScope


async def my_agent(messages: list[dict[str, str]]) -> str:
    """演示函数；替换为自己的 Agent 调用，不是框架要求的基类。"""
    contexts = [
        row["content"] for row in messages
        if row["role"] == "system"
        and row["content"].startswith("Relevant memory:")
    ]
    return "\n".join(contexts) or "没有历史记忆；这是演示回答。"


async def run_task(
    memory: Memory,
    scope: MemoryScope,
    question: str,
    *,
    execute: Callable[[list[dict]], Awaitable[str]],
    messages: list[dict] | None = None,
) -> AgentTrace:
    """用户侧适配示例：连接已有 Agent，不是框架内置 API。"""
    episode_id = f"task-{uuid4().hex}"
    # 本例先保存轨迹，结束后统一提交；不用每一步都在线更新。
    await memory.start_episode(
        scope=scope, episode_id=episode_id, learning=False,
    )
    try:
        await memory.record_step(AgentStep(
            kind=EventKind.MESSAGE, role="user", text=question,
        ))
        contribution = await memory.recall(question, scope=scope)
        application = memory.apply_contribution(
            messages if messages is not None else [{"role": "user", "content": question}],
            contribution,
        )
        # 示例只接入文本与直接答案；其他输出需要你的适配器处理。
        if application.tools or application.model_hooks or application.artifacts:
            raise ValueError("请先实现工具、模型钩子或附件的接入")
        if application.direct_answer is not None:
            answer = application.direct_answer
        else:
            answer = await execute([dict(row) for row in application.messages])
        if not isinstance(answer, str):
            raise TypeError("执行函数必须返回文本，请先从实际 Agent 响应中提取回答")
        memory.confirm_application(contribution)
        await memory.record_step(AgentStep(
            kind=EventKind.MESSAGE, role="assistant", text=answer,
        ))
        return await memory.finish_task(output=answer, learn=False)
    except BaseException:
        memory.discard_episode()
        raise


async def main(artifact_dir: Path, method_id: str) -> None:
    config = {
        "method": method_id,
        "runtime": {"artifact_dir": str(artifact_dir)},
    }
    report = Memory.check(config)
    if not report["ready"]:
        raise RuntimeError(f"记忆环境未就绪：{report}")
    scope = MemoryScope(user_id="alice", agent_id="assistant")

    async with Memory.from_dict(config) as memory:
        questions = (
            "请记住：Helios 的服务端口是 8080。",
            "Helios 的服务端口是多少？",
        )
        for index, question in enumerate(questions, start=1):
            trace = await run_task(memory, scope, question, execute=my_agent)
            print(f"任务 {index} 回答：", trace.final_output)
            # 本例的两个基线不要求评价标签，因此不虚构成功或奖励。
            result = await memory.submit_feedback(trace.case_id, scope=scope)
            print("任务状态：", result.execution, result.learning)

        run_dir = Path(memory.config.runtime.artifact_dir)
        await memory.checkpoint(run_dir / "checkpoint.json")
        print("运行目录：", run_dir)
        print("检查点：", run_dir / "checkpoint.json")
    # 退出 async with 后，Memory 已关闭。


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemAtlas 完整任务接入演示")
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path(".mematlas/quickstart"),
        help="运行产物根目录",
    )
    parser.add_argument(
        "--method", choices=("baseline_bm25", "baseline_none"),
        default="baseline_bm25", help="本示例验证的两种方法",
    )
    discovery = parser.add_mutually_exclusive_group()
    discovery.add_argument("--list-methods", action="store_true", help="列出框架已注册的方法")
    discovery.add_argument("--describe-method", metavar="METHOD_ID", help="查看单个方法的配置与能力")
    args = parser.parse_args()
    if args.list_methods:
        for card in MemAtlas().methods():
            print(f"{card['method_id']}: {card['name']}")
    elif args.describe_method:
        print(json.dumps(MemAtlas().describe(args.describe_method), ensure_ascii=False, indent=2))
    else:
        asyncio.run(main(args.artifact_dir, args.method))
```

### 4.3 运行与预期结果

```powershell
Set-Location 'D:\Agent-RSI'
python .\quickstart_agent.py --help
python .\quickstart_agent.py --list-methods
python .\quickstart_agent.py --describe-method baseline_bm25
python .\quickstart_agent.py
python .\quickstart_agent.py --method baseline_none
```

BM25 第一次没有历史，第二次应出现 Relevant memory: 和 8080，两次状态为 completed submitted。无记忆基线不返回检索记忆。第二次回显第一轮“没有历史记忆”的演示回答，是该条回答也被保存并匹配，不是本轮检索失败。

示例不调用真实模型，只验证事件记录、检索、任务、反馈与保存。实际运行目录以打印结果为准。脚本的本地加载仅对当前进程有效，不等于安装了包；独立使用 python -m mematlas 需先有可导入的包环境。示例 CLI 只验证两个基线，其他方法按第 1 节配置并满足其依赖后接入。

## 5. 需要功能的使用实例

本节片段放在已经打开的 Memory 中，复用完整脚本的导入和函数。普通接入不必全部使用。

### 5.1 改为包装模型调用

下面是一条替代任务路径，已包含开始、结束、异常处理和反馈，不是在同一次 run_task 之后再执行。my_agent 是演示占位，可直接替换为已有的模型 callable；call_model 支持同步或异步 callable，不要求用户实现同名包装函数。

```python
question = "Helios 的服务端口是多少？"
messages = [{"role": "user", "content": question}]
episode_id = f"model-{uuid4().hex}"
await memory.start_episode(scope=scope, episode_id=episode_id, learning=False)
try:
    await memory.record_step(AgentStep(
        kind=EventKind.MESSAGE, role="user", text=question,
    ))
    contribution = await memory.recall(question, scope=scope)
    answer = await memory.call_model(
        my_agent, messages, contribution=contribution,
    )
    trace = await memory.finish_task(output=answer, learn=False)
except BaseException:
    memory.discard_episode()
    raise
# 本片段仍以不要求评价标签的 BM25 为例。
result = await memory.submit_feedback(episode_id, scope=scope)
```

不要再手动应用、确认或记录该次返回。工具、模型钩子、附件等结构化输出要求 callable 声明 accepts_memory_application=True 并真正处理；标记本身不完成接入。

### 5.2 执行工具并自动记录

替换 get_service_config 的函数体即可接入真实查询。示例由代码指定工具，没有模型自主规划：

```python
def get_service_config(service: str) -> dict:
    """演示工具，实际接入时替换为真实查询。"""
    return {"service": service, "port": 8080}


episode_id = f"tool-{uuid4().hex}"
await memory.start_episode(scope=scope, episode_id=episode_id, learning=False)
try:
    tool_result = await memory.call_tool(
        get_service_config, "get_service_config", {"service": "Helios"},
    )
    # 本例直接以工具返回结束任务，真实 Agent 可继续推理再结束。
    trace = await memory.finish_task(output=tool_result, learn=False)
except BaseException:
    memory.discard_episode()
    raise
result = await memory.submit_feedback(episode_id, scope=scope)
```

工具抛异常时保存失败追踪，不生成成功结果。工具注册、选择、重试以及把结果加入后续推理，仍按原 Agent 的实际机制完成。

### 5.3 原 Agent 已执行工具，只补步骤

复用上一节工具，放在已开始的任务内部。这是手动记录的替代方式，同次调用不再使用 call_tool：

```python
arguments = {"service": "Helios"}
await memory.record_step(AgentStep(
    kind=EventKind.ACTION, role="assistant", text="查询服务配置",
    tool_name="get_service_config", tool_input=arguments,
))
tool_result = get_service_config(**arguments)  # 实际执行一次，不编造结果
await memory.record_step(AgentStep(
    kind=EventKind.TOOL_RESULT, role="tool",
    text=json.dumps(tool_result, ensure_ascii=False),
    tool_name="get_service_config", tool_output=tool_result,
))
```

异步工具按实际接口 await；失败时走真实错误处理，不编造工具结果。run_task 已记录问题和最终回答，只补它没有记录的真实过程。

### 5.4 导入已有历史

新任务开始前导入，避免每轮重复导入同一份历史：

```python
from mematlas import ContentBlock, MemoryEvent

await memory.ingest_history([
    MemoryEvent(
        kind=EventKind.MESSAGE, scope=scope, role="user",
        content=(ContentBlock.text("Helios 的服务端口是 8080。"),),
    ),
])
```

上例适用于 BM25。其他方法可能要求 outcome=EpisodeOutcome(...) 以及真实标签；检查方法的导入策略和反馈字段，不生成假的成功或奖励。

### 5.5 反馈稍后到达

任务已结束但未提交时，使用原 episode_id、scope 和真实评价。以下 verified_* 值由你的业务评价器或人工提供，缺少时按方法要求等待或保留 None：

```python
result = await memory.submit_feedback(
    episode_id, scope=scope,
    success=verified_success, reward=verified_reward, feedback=verified_feedback,
)
print(result.execution, result.feedback, result.learning)
```

重启后先按第 5.7 节恢复。任务执行失败与学习提交失败分开处理：前者放弃活动任务；后者检查保存的学习状态，不要重新提交整段任务。状态不确定时先恢复已确认的检查点，不无条件重试。

### 5.6 方法需要 LLM 或 Embedding 时

这是配置模板，不是已配置好的服务示例。按真实服务设置环境变量，安装所选方法的依赖，并只提供该方法需要的服务：

```python
import os

config["providers"] = {
    "llm": {
        "model": os.environ["MEMORY_LLM_MODEL"],
        "base_url": os.environ["MEMORY_LLM_BASE_URL"],
        "api_key_env": "MEMORY_LLM_API_KEY",
    },
    "embedding": {
        "model": os.environ["MEMORY_EMBEDDING_MODEL"],
        "base_url": os.environ["MEMORY_EMBEDDING_BASE_URL"],
        "api_key_env": "MEMORY_EMBEDDING_API_KEY",
    },
}
```

api_key_env 填密钥环境变量名。不要与 method.params 中同名模型配置冲突；Agent 自己的模型仍在原 Agent 初始化中配置。

### 5.7 重启继续使用旧记忆

使用原请求 config、原 scope，把 run_dir 替换为完整示例打印的真实运行目录，而不是 artifact_dir 根目录：

```python
# 替换为主示例实际打印的目录。
run_dir = Path("上次打印的完整运行目录")
async with Memory.resume(config, run_dir) as memory:
    contribution = await memory.recall("Helios", scope=scope)
    print([item.text for item in contribution.items])
    # 如需退回显式保存的恢复点，再调用：
    # await memory.load_state(run_dir / "checkpoint.json")
```

resume 打开运行的现有状态；load_state 可以另行退回检查点。方法需支持相应能力，不同实例不能默认安全共享同一原生数据库。

### 5.8 测试时不更新记忆

在任务边界进入只读上下文；方法需支持冻结。run_task 已用 learn=False 结束，下面不提交训练反馈：

```python
async with memory.read_only():
    trace = await run_task(
        memory, scope, question, execute=my_agent, messages=messages,
    )
    print(trace.final_output)
```

## 6. 遇到问题检查哪里

| 问题 | 检查位置 |
|---|---|
| 导入 mematlas 失败 | 是否复制完整源码加载代码，脚本是否放在第 4.1 节的位置 |
| 原 Agent 执行入口的格式不兼容 | 先检查能否直接传入已有异步执行函数或绑定方法；确需转换时，按实际 SDK 写适配，不强制命名 my_agent |
| 原提示或历史没传入 | 给 run_task 传完整 messages，不只传 question |
| 工具步骤缺失 | 添加真实执行钩子的 record_step，或选用 call_tool |
| 回答或工具步骤重复 | 同一次模型执行和工具执行各只选一种记录路径 |
| 重启后记忆像是丢了 | 是否默认创建了新运行；应使用原配置与真实目录 resume |
| 方法缺少输入或反馈 | 看 describe 中的事件、模态、scope、反馈与输出要求 |
| 方法还有批次或 epoch 学习阶段 | 按方法说明使用 learning_boundary 或 command，不将 submitted 视为完成所有额外训练阶段 |

原 Agent 的模型切换、工具注册、会话重置和资源关闭，继续调用其真实 SDK；本框架没有要求统一的 Agent 类。本文没有提供 run_task 以外的第二套普通业务适配函数，也不把 AgentTrace 字段当作执行接口。

源码对照：[方法发现](../api.py)、[Memory](../runtime/memory.py)、[任务与反馈](../runtime/episodes.py)、[模型与工具辅助](../runtime/application.py)、[公共数据](../core/models.py)、[步骤与轨迹](../core/trajectory.py)、[配置](../core/config.py)。Benchmark 回调和评分流程另写文档，不与本文普通业务的任务边界混用。

文档核验（2026-10-28）：完整代码与两份本地脚本一致；11 个 Python 片段语法和本地链接检查通过。在不设置额外导入环境变量的子进程中验证了帮助、方法查看和两个基线的完整任务；已有异步函数、绑定方法及异步可调用对象均可直接作为 execute 使用，无需另写 my_agent。验证使用临时目录，未调用真实模型或网络；任意 SDK 对象接入和正式 run_task 仍不是当前已实现能力。
