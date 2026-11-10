"""Agent 接入示例：execute(task, memory) -> AgentTrace。"""

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from functools import partial
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, TypedDict
from uuid import uuid4

# 当前团队仓库即包源码；本段只为当前进程加载源码，不安装或修改全局环境。
if importlib.util.find_spec("mematlas") is None:
    script_dir = Path(__file__).resolve().parent
    package_dir = next((
        path for path in (script_dir / "MemAtlas", script_dir)
        if (path / "__init__.py").is_file()
        and (path / "runtime" / "memory.py").is_file()
    ), None)
    if package_dir is None:
        raise ModuleNotFoundError(
            "未找到 mematlas。请将脚本放在 MemAtlas 源码目录或其父目录，"
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

Messages = list[dict[str, Any]]
# 业务执行入口统一接收原始任务和活动 Memory，返回真实轨迹。
AgentExecute = Callable[[Any, Memory], Awaitable[AgentTrace]]
# 仅供下方文本适配器绑定原 Agent；不是第二种任务执行接口。
MessageAgent = Callable[[Messages], Awaitable[str]]


class TaskInput(TypedDict):
    """默认文本适配器的数据格式；其他场景可使用自己的业务输入。"""

    query: str
    messages: Messages


def to_text_task(row: Mapping[str, Any]) -> TaskInput:
    """接受完整 query/messages，或简写 question + 可选 system_prompt。"""
    if not isinstance(row, Mapping):
        raise TypeError("任务必须是 JSON 对象")
    if "query" in row or "messages" in row:
        if "query" not in row or "messages" not in row:
            raise ValueError("完整格式必须同时包含 query 和 messages")
        query, messages = row["query"], row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 必须是非空消息列表")
        if any(not isinstance(message, Mapping) or not isinstance(message.get("role"), str)
               for message in messages):
            raise ValueError("messages 中每条消息必须是带字符串 role 的对象")
    else:
        query = row.get("question")
        system_prompt = row.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt 必须是字符串")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 或 question 必须是非空字符串")
    return {"query": query, "messages": deepcopy([dict(message) for message in messages])}


def load_tasks(path: str | Path) -> Iterator[TaskInput]:
    """逐行读取 UTF-8 JSONL，不一次加载整个数据集；仅用于文本任务。"""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield to_text_task(json.loads(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"任务文件 {path}:{line_number}: {exc}") from exc


async def demo_agent(messages: Messages) -> str:
    """仅回显检索记忆，不调用模型。真实接入时替换 D 中的 agent_call。"""
    contexts = [
        row["content"] for row in messages
        if row.get("role") == "system"
        and isinstance(row.get("content"), str)
        and row["content"].startswith("Relevant memory:")
    ]
    return "\n".join(contexts) or "没有历史记忆；这是演示回答。"


async def execute_text(
    task: Any, memory: Memory, *, agent_call: MessageAgent,
) -> AgentTrace:
    """文本执行接入：原 Agent 只处理消息；本函数负责记忆接线和真实步骤。

    agent_call 在 D 处绑定后，入口即 execute(task, memory) -> AgentTrace。
    本函数不开始/结束任务，不提交学习；支持文本贡献和直接答案。
    """
    text_task = to_text_task(task)
    question = text_task["query"]
    await memory.record_step(AgentStep(
        kind=EventKind.MESSAGE, role="user", text=question,
    ))
    contribution = await memory.recall(question, scope=memory.scope)
    application = memory.apply_contribution(text_task["messages"], contribution)
    if application.tools or application.model_hooks or application.artifacts:
        raise ValueError("文本适配器不支持工具、模型钩子或附件；请接入实际 Agent 后再使用")
    if application.direct_answer is not None:
        answer = application.direct_answer
    else:
        answer = await agent_call(deepcopy([dict(row) for row in application.messages]))
    if not isinstance(answer, str):
        raise TypeError("文本 Agent 需返回字符串；请按实际 SDK 提取最终回答")
    memory.confirm_application(contribution)
    await memory.record_step(AgentStep(
        kind=EventKind.MESSAGE, role="assistant", text=answer,
    ))
    return AgentTrace(
        case_id=None, scope=memory.scope, steps=memory.steps,
        final_output=answer, metadata={"task": question},
    )


# ===== A：记忆方法与参数；方法需要模型服务时再补 providers =====
MEMORY_CONFIG: dict[str, Any] = {
    "method": {"method_id": "baseline_bm25", "params": {}},
    "runtime": {"artifact_dir": ".mematlas/quickstart"},
}

# ===== B：填写实际记忆归属；不是登录信息或模型名称 =====
SCOPE = MemoryScope(user_id="alice", agent_id="assistant")

# ===== C：默认文本数据；也可用 --tasks tasks.jsonl 或 load_tasks(path) =====
TASKS: Iterable[Any] = [
    {
        "query": "请记住：Helios 的服务端口是 8080。",
        "messages": [
            {"role": "system", "content": "你是服务运维助手。"},
            {"role": "user", "content": "请记住：Helios 的服务端口是 8080。"},
        ],
    },
    {
        "query": "Helios 的服务端口是多少？",
        "messages": [
            {"role": "system", "content": "你是服务运维助手。"},
            {"role": "user", "content": "Helios 的服务端口是多少？"},
        ],
    },
]

# ===== D：统一执行入口；文本场景只替换 agent_call 为原 Agent 的兼容入口 =====
# partial 预先绑定原 Agent，运行时仍只传 task 和 memory。
# 新场景可直接设置 AGENT_EXECUTE = 自己的异步 execute，返回 AgentTrace。
AGENT_EXECUTE: AgentExecute = partial(execute_text, agent_call=demo_agent)


async def run_task(
    memory: Memory,
    scope: MemoryScope,
    task: Any,
    *,
    execute: AgentExecute,
    task_id: str | None = None,
    learning: bool = False,
) -> AgentTrace:
    """本脚本的统一任务包装；不是 mematlas 包导出的 API。

    拥有任务边界；execute 负责业务、检索与轨迹，不能自行结束/提交任务。
    步骤支持增量记录或完整返回；增量记录与返回 steps 必须一致。
    """
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
        raise ValueError("task_id 必须是非空字符串")
    # 公共 start_episode 对相同活动 ID 允许复用；任务包装不能因此嵌套执行。
    if memory.case_started and not memory.finalized:
        raise ValueError("已有未结束任务；请先结束或放弃它，再调用 run_task")
    episode_id = task_id or f"task-{uuid4().hex}"
    await memory.start_episode(scope=scope, episode_id=episode_id, learning=learning)
    try:
        trace = await execute(task, memory)
        if not isinstance(trace, AgentTrace):
            raise TypeError("execute(task, memory) 必须返回 AgentTrace")
        if memory.finalized:
            raise ValueError("execute 不应自行结束任务；由 run_task 管理任务边界")
        if trace.case_id not in (None, episode_id) or trace.scope != memory.scope:
            raise ValueError("返回轨迹的任务 ID 或 scope 与当前任务不一致")
        if memory.steps and memory.steps != trace.steps:
            raise ValueError("已记录步骤与返回 AgentTrace.steps 不一致")
        if not memory.steps:
            for step in trace.steps:
                await memory.record_step(step)
        return await memory.finish_task(
            output=trace.final_output, success=trace.success, reward=trace.reward,
            feedback=trace.feedback, metadata=trace.metadata,
            retrieved=trace.retrieved or None, learn=False,
        )
    except BaseException:
        memory.discard_episode()
        raise


async def main(
    artifact_dir: Path | None = None,
    method_id: str | None = None,
    tasks_file: Path | None = None,
) -> None:
    """打开一次 Memory，串行执行任务，提交真实经历，保存并关闭。"""
    config = deepcopy(MEMORY_CONFIG)
    if artifact_dir is not None:
        config["runtime"]["artifact_dir"] = str(artifact_dir)
    if method_id is not None:
        config["method"] = {"method_id": method_id, "params": {}}
    report = Memory.check(config)
    if not report["ready"]:
        raise RuntimeError(f"记忆环境未就绪：{report}")
    tasks = load_tasks(tasks_file) if tasks_file is not None else TASKS
    async with Memory.from_dict(config) as memory:
        for index, task in enumerate(tasks, start=1):
            trace = await run_task(memory, SCOPE, task, execute=AGENT_EXECUTE)
            print(f"任务 {index} 回答：", trace.final_output)
            # 按需 E：可在此接真实评价器；晚到时先 checkpoint，再暂缓提交。
            # 默认基线无需评价标签，不虚构 success/reward；保留执行器已有评价。
            result = await memory.submit_feedback(
                trace.case_id, scope=trace.scope, success=trace.success,
                reward=trace.reward, feedback=trace.feedback,
            )
            print("任务状态：", result.execution, result.learning)
        run_dir = Path(memory.config.runtime.artifact_dir)
        await memory.checkpoint(run_dir / "checkpoint.json")
        print("运行目录：", run_dir)
        print("检查点：", run_dir / "checkpoint.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemAtlas 接入：execute(task, memory) -> AgentTrace")
    parser.add_argument("--tasks", type=Path, metavar="JSONL", help="替代 C 处文本数据；每行 question 或 query/messages")
    parser.add_argument("--artifact-dir", type=Path, help="覆盖 A 处产物根目录；默认每次新建运行")
    parser.add_argument("--method", metavar="METHOD_ID", help="覆盖方法 ID 并清空方法参数；先满足所选方法要求")
    discovery = parser.add_mutually_exclusive_group()
    discovery.add_argument("--list-methods", action="store_true", help="列出已注册方法")
    discovery.add_argument("--describe-method", metavar="METHOD_ID", help="查看方法参数、依赖与能力")
    args = parser.parse_args()
    if args.list_methods:
        for card in MemAtlas().methods():
            print(f"{card['method_id']}: {card['name']}")
    elif args.describe_method:
        print(json.dumps(MemAtlas().describe(args.describe_method), ensure_ascii=False, indent=2))
    else:
        asyncio.run(main(args.artifact_dir, args.method, args.tasks))
