"""AWM 网页测评适配：连接外部环境、执行 Agent 动作并整理完整学习经历。

通过 MethodServices.objects['awm_environment_factory'](task, public_options) 提供环境。
工厂返回实现 reset/step/close 的适配器，只接收任务与公开选项，不接收参考答案或目标动作。
浏览器部署、观察数据处理和真实成败评价由外部环境负责。
"""
from __future__ import annotations
from dataclasses import replace
import re
from typing import Any, Mapping

from mematlas.core.models import MemoryEpisode,MemoryEvent,content_text
from mematlas.core.contributions import contribution_text
from mematlas.core.plugin import call_maybe_async
from mematlas.core.telemetry import active_usage
from mematlas.providers.instrumentation import MeteredService
from mematlas.evaluation.qa import DefaultQA
from .environment import AWMEnvironment, BrowserState

__all__ = ["AWMQA"]


def _state(value):
    if (not isinstance(value, Mapping)
            or not isinstance(value.get("observation"), str)
            or type(value.get("done")) is not bool):
        raise ValueError("AWM environment must return real observation:str and done:bool")
    if "success" in value and type(value["success"]) is not bool:
        raise ValueError("AWM environment success must be a real bool")
    return value


def _options(config):
    options = [
        method.method.params
        for method in config.methods
        if method.method.method_id == "awm"
    ]
    if not options or len({value["profile"] for value in options}) != 1:
        raise ValueError("Use separate experiments for mixed AWM profiles")
    return options[0]


def _case_options(metadata):
    values = dict(metadata.get("awm", {}))
    website = metadata.get("memory_extensions", {}).get("awm", {}).get("website")
    if website is not None:
        if "website" in values and values["website"] != website:
            raise ValueError("AWM website metadata conflicts")
        values["website"] = website
    return values


class AWMQA(DefaultQA):
    """Execute web tasks through an injected real environment and preserve traces."""

    # Native Mind2Web scoring needs all step results, including skipped actions.
    include_episode_in_prediction = True

    @staticmethod
    def collect_training_episode(case):
        """Execute raw web tasks before learn; imported episodes/history take priority."""
        return case.episode is None and not case.history

    @staticmethod
    def retrieval_extensions(metadata):
        extensions = DefaultQA.retrieval_extensions(metadata)
        website = _case_options(metadata).get("website")
        if website:
            extensions["awm"] = {"website": website}
        return extensions

    @staticmethod
    def prepare_episode(case,episode,*,last_for_scope):
        values = {
            key: value
            for key, value in _case_options(case.metadata).items()
            if key in {"website", "domain", "subdomain", "template_id", "canonical"}
        }
        options = {**values, **episode.extensions.get("awm", {})}
        if last_for_scope:
            options["end_of_training"] = True
        return replace(episode, extensions={**episode.extensions, "awm": options})

    @classmethod
    def validate_cases(cls,config,train,tests):
        profile = _options(config)["profile"]
        if profile == "medicine":
            raise ValueError("AWM medicine requires an external clinical Agent, not browser QA")
        for case in train:
            if case.metadata.get("benchmark_id") == "mind2web":
                canonical = bool(case.episode and case.episode.extensions.get("awm", {}).get("canonical"))
                offline = _options(config).get("induction_mode", "online") == "offline"
                if canonical != offline:
                    raise ValueError("Mind2Web canonical training requires offline induction; raw Agent training requires online induction")
        for case in (*train,*tests):
            imported = case.episode.extensions.get("awm", {}) if case.episode else {}
            options = {**_case_options(case.metadata), **imported}
            website = options.get("website")
            if not isinstance(website, str) or not website.strip():
                raise ValueError(
                    "AWM web navigation requires website metadata and a compatible browser "
                    "environment; ordinary conversation QA is unsupported"
                )
            if profile == "mind2web" and any(
                not options.get(key) for key in ("domain", "subdomain")
            ):
                raise ValueError("AWM Mind2Web QA requires real domain/subdomain")
            template = options.get("template_id")
            if profile == "webarena" and not (
                case.episode and case.episode.success is False
            ) and (
                isinstance(template, bool)
                or not isinstance(template, (str, int))
                or isinstance(template, str) and not template
            ):
                raise ValueError("AWM WebArena QA requires real template_id before acting")
            max_steps = options.get("max_steps", 30)
            if type(max_steps) is not int or not 1 <= max_steps <= 100:
                raise ValueError("AWM max_steps must be within 1..100")

    async def __call__(self,case,memory):
        if case.metadata.get("benchmark_id") == "mind2web":
            from .mind2web_qa import execute_mind2web
            return await execute_mind2web(self, case, memory)
        factory = self.services.object("awm_environment_factory")
        if not callable(factory):
            raise ValueError(
                'AWMQA requires MethodServices.objects["awm_environment_factory"]; '
                "no plain-text QA fallback"
            )
        if self.llm is None:
            raise ValueError("AWMQA requires an answer model for the external Agent")

        profile = _options(self.config)["profile"]
        options = _case_options(case.metadata)
        self.validate_cases(self.config, (), (case,))
        public = {
            key: options[key]
            for key in ("website", "domain", "subdomain")
            if key in options
        }
        public["profile"] = profile
        environment = await call_maybe_async(
            factory,
            content_text(case.query),
            public,
            run_sync_in_thread=True,
        )

        events = []
        applied = []
        output = ""
        records = []
        model = self.llm
        active = active_usage()
        if active is not None and not isinstance(model, MeteredService):
            model = MeteredService(model, active[0], "llm")
        try:
            state = _state(await call_maybe_async(
                environment.reset,
                run_sync_in_thread=True,
            ))
            if state["done"]:
                raise ValueError("AWM environment reset must start a real unfinished task")
            events.append(MemoryEvent("observation", state["observation"], role="tool"))

            # Retrieve before every action so a multi-step Agent sees the latest
            # observation and may use the same workflow throughout the task.
            for _ in range(options.get("max_steps", 30)):
                used = await memory.search(
                    case.query,
                    scope=case.scope,
                    extensions={"awm": {"website": options["website"]}},
                )
                applied.append(used)
                if profile == "mind2web":
                    action_space = (
                        "Use the literal square-bracket format below. Do not omit the brackets. "
                        "Output exactly one raw action and no explanation, Markdown, backticks, or code block. "
                        "Replace 143 with the exact element id from the current observation. Valid examples: "
                        "CLICK [143]; TYPE [143] [text]; SELECT [143] [option]. Invalid examples: "
                        "CLICK 143; CLICK [id=143].\n"
                        "CLICK [id] means click an element. TYPE [id] [value] means type text. "
                        "SELECT [id] [value] means select an option."
                    )
                else:
                    action_space = (
                        "Python-shaped browser actions such as click('id'), fill('id', 'value'), "
                        "select_option('id', 'value')"
                    )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a web navigation Agent. Output the next action and wait for "
                            "the next observation. Use only the actual current observation. "
                            f"Action space: {action_space}\nWorkflow memory:\n"
                            f"{contribution_text(used)}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task: {content_text(case.query)}\nTrajectory:\n"
                            + "\n".join(content_text(event.content) for event in events)
                        ),
                    },
                ]
                raw = str(await model.complete(messages, model=self.model, temperature=0))
                blocks = re.findall(r"```(?:python\s*\n)?(.*?)```", raw, re.S)
                action = blocks[-1].strip() if blocks else raw.strip()
                if not action:
                    raise ValueError("AWM Agent returned an empty action")
                thought = raw[:raw.find("```")].strip() if blocks else ""
                if profile == "webarena":
                    data = {
                        "thought": thought,
                        "actions": [line for line in action.splitlines() if line.strip()],
                    }
                else:
                    data = {}
                events.append(MemoryEvent(
                    "action", action, role="assistant", extensions={"awm": data}
                ))
                records.append({"messages": messages, "raw_output": raw, "action": action})

                state = _state(await call_maybe_async(
                    environment.step,
                    action,
                    run_sync_in_thread=True,
                ))
                events.append(MemoryEvent("tool_result", state["observation"], role="tool"))
                output = action
                if state["done"]:
                    break

            if not state["done"] or "success" not in state:
                raise ValueError(
                    "AWMQA needs terminal real success feedback; keep incomplete task outside learn"
                )
            episode = MemoryEpisode(
                task=content_text(case.query),
                scope=case.scope,
                events=events,
                output=output,
                success=state["success"],
                applied_memory=applied,
                feedback=(
                    "External web environment returned terminal success="
                    + str(state["success"])
                ),
                extensions={"awm": {"canonical": False}},
            )
            return self.prepare_episode(case, episode, last_for_scope=False)
        finally:
            sink = self.services.object("awm_trace_sink")
            try:
                if sink:
                    sink({
                        "case_id": case.case_id,
                        "model_calls": records,
                        "events": [event.to_dict() for event in events],
                    })
            finally:
                await call_maybe_async(environment.close, run_sync_in_thread=True)
