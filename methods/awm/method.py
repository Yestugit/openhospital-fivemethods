"""AWM 公共接口适配：按网站收集轨迹、自动归纳并返回完整工作流文件。

Agent 执行动作并提供真实反馈；归纳在 learn 内部触发，状态保存和检索不触发归纳。
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import random
from typing import Any,Mapping,TypedDict

from mematlas.core.config import MethodConfig
from mematlas.core.models import MemoryContribution,MemoryEpisode,MemoryEvent,content_text
from mematlas.core.plugin import BaseMemoryMethod,MethodPlugin,MethodContext,MethodServices,call_maybe_async
from mematlas.core.scope import scope_key
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services
from mematlas.methods.storage import atomic_write_json,configured_file,read_json
from .core import AWMCore

DESCRIPTOR=load_method_descriptor(__file__)


class AddOptions(TypedDict,total=False):
    """learn 的专有数据，放入 MemoryUnit.extensions['awm']。

    Attributes:
        website: str，Mind2Web/WebArena 必填，来自实际任务的网站标识；
            medicine 可省略，默认 "medicine"。用于轨迹分组和工作流归属。
        domain: str，Mind2Web 必填，来自任务数据的网站领域标签。
        subdomain: str，Mind2Web 必填，来自任务数据的网站子领域标签。
        template_id: str | int，WebArena 成功经历必填，来自真实任务模板；
            用于按模板抽样，不能用 episode_id 替代。
        canonical: bool，默认 False；offline 必须为 True，表示输入为实际标注轨迹。
        end_of_training: bool，默认 False；由 Agent 或调度器在最后一份训练经历设置，
            通知 learn 处理各网站尚未归纳的轨迹。
        mind2web: dict，原生 Mind2Web QA 自动附带的逐步评分与执行协议；
            用户无需填写，不参与工作流归纳，仅随实际经历保存以便评测和审计。

    task、events、success 使用 MemoryEpisode 公共字段。WebArena/medicine 需要真实
    success；Mind2Web online 使用实际 Agent 经历，不按成败过滤。
    """
    website:str
    domain:str
    subdomain:str
    template_id:str|int
    canonical:bool
    end_of_training:bool
    mind2web:dict[str,Any]


class EventOptions(TypedDict,total=False):
    """动作事件的专有数据，放入 MemoryEvent.extensions['awm']。

    Attributes:
        action_repr: str，Mind2Web 标注动作的可读表示，由任务数据或动作解析器提供；
            可省略，默认使用 action 事件正文。
        parsed_action: str，原生 Mind2Web QA 从真实模型回复提取的动作；
            QA 自动填写，原始回复保留在事件正文中用于在线归纳。
        thought: str，WebArena Agent 对当前动作实际生成的思考；可省略，默认空字符串。
        actions: list[str]，WebArena 当前事件包含的实际动作文本；可省略，
            默认将 action 事件正文作为一条动作。
    """
    action_repr:str
    parsed_action:str
    thought:str
    actions:list[str]


class SearchOptions(TypedDict):
    """检索条件，传入 retrieve(..., extensions={'awm': ...})。

    Attributes:
        website: str，必填，来自当前任务的目标网站标识，应与学习时的网站标识一致。

    返回该网站的完整工作流文件，不做语义排序；单站最多返回一个文件。
    """
    website:str


class AWMMemory(BaseMemoryMethod):
    def __init__(self,config:MethodConfig,services:MethodServices):
        self.params = {
            name: spec["default"]
            for name, spec in DESCRIPTOR.config_schema["properties"].items()
            if "default" in spec
        }
        self.params.update(config.params)
        self.services = services
        self.core = AWMCore(self.params["profile"])
        self._episodes = {}
        self._order = []
        self._pending = {}
        self._workflow_text = {}
        self._workflow_sources = {}
        self._history = []
        self._active_scope = None
        self._storage_path = None
        self._persistence_ready = False

    def _ensure_scope(self,scope,*,read_only=False):
        identity=scope_key(scope)
        if self._active_scope is None:
            if not read_only:self._active_scope=identity
        elif identity != self._active_scope:
            raise ValueError("AWM supports one Memory per scope")

    async def setup(self,context:MethodContext):
        if self.services.object("awm.inducer") is None:
            prepare_services(self.params, self.services, method_id="awm", llm=True, embedding=False)
        self._storage_path = configured_file(self.services, "awm", None, "awm.json")
        if self._storage_path.exists():
            # Restore validates provenance and recomputes all derived workflow state.
            self._restore(read_json(self._storage_path, {}))
        self._persistence_ready = True

    async def _observe(self,event:MemoryEvent):
        self._ensure_scope(event.scope)  # Public runtime journals the original event.

    def _example(self,episode:MemoryEpisode):
        if not episode.task or not episode.task.strip() or not episode.events:
            raise ValueError("AWM requires task and real trajectory events")

        profile = self.params["profile"]
        options = episode.extensions.get("awm", {})
        website = options.get("website", "medicine" if profile == "medicine" else None)
        if not isinstance(website, str) or not website.strip():
            raise ValueError("AWM requires a real website")

        # WebArena and the clinical extension induce only from real successes.
        if profile in {"webarena", "medicine"}:
            if episode.success is None:
                raise ValueError("AWM requires real boolean success")
            if episode.success is not True:
                return website, None

        if self.params["induction_mode"] == "offline" and options.get("canonical") is not True:
            raise ValueError("AWM offline requires canonical=True for actual annotated/canonical experiences")

        actions = [event for event in episode.events if event.kind.value == "action"]
        if profile == "mind2web":
            required_labels = ("domain", "subdomain")
            if any(not isinstance(options.get(key), str) or not options[key].strip()
                   for key in required_labels):
                raise ValueError("AWM Mind2Web requires domain and subdomain")

            if self.params["induction_mode"] == "offline":
                representations = [
                    event.extensions.get("awm", {}).get(
                        "action_repr", content_text(event.content)
                    )
                    for event in actions
                ]
            else:
                # Pair each action with the immediately preceding real observation.
                representations = []
                observation = None
                for event in episode.events:
                    if event.kind.value in {"observation", "tool_result"}:
                        observation = content_text(event.content)
                    elif event.kind.value == "action":
                        if not observation:
                            raise ValueError(
                                "AWM Mind2Web online needs an actual observation before each action"
                            )
                        representations.append(f"# {observation}\n{content_text(event.content)}")
                        observation = None

                # Native evaluation may skip every step before any actor call.
                # Retain the completed episode, but do not induce workflows from
                # candidate/context failures with no actual Agent experience.
                scores = options.get("mind2web", {}).get("step_scores", [])
                if not representations and scores and all(
                    row.get("prediction") is None for row in scores
                ):
                    return website, None

            if not representations or any(
                not isinstance(value, str) or not value.strip() for value in representations
            ):
                raise ValueError("AWM requires nonempty real action representations")
            result = {
                "confirmed_task": episode.task,
                "action_reprs": representations,
                "domain": options["domain"],
                "subdomain": options["subdomain"],
                "website": website,
            }
        elif profile == "webarena":
            template = options.get("template_id")
            if (isinstance(template, bool) or not isinstance(template, (str, int))
                    or isinstance(template, str) and not template):
                raise ValueError("AWM WebArena requires an actual template_id")

            thoughts, action_lists = [], []
            for event in actions:
                data = event.extensions.get("awm", {})
                thought = data.get("thought", "")
                values = data.get("actions", [content_text(event.content)])
                if (not isinstance(thought, str) or not isinstance(values, list)
                        or any(not isinstance(value, str) for value in values)):
                    raise ValueError("AWM thought/actions must be real text and list[str]")
                valid_actions = self.core.valid_actions(values)
                if valid_actions:
                    thoughts.append(thought)
                    action_lists.append(valid_actions)
            if not action_lists:
                raise ValueError("AWM WebArena requires actual eligible actions")
            result = {
                "query": episode.task,
                "think_list": thoughts,
                "action_list": action_lists,
                "website": website,
                "template_id": template,
            }
        else:
            if website != "medicine":
                raise ValueError("AWM medicine uses website=medicine")
            trajectory = "\n".join(
                content_text(event.content)
                for event in episode.events
                if event.kind.value != "feedback"
            )
            if not trajectory.strip():
                raise ValueError("AWM medicine needs a nonempty real encounter")
            result = {"query": episode.task, "trajectory": trajectory, "website": website}
        return website,result

    def _examples(self,website,ids=None):
        result=[]
        for identity in self._order if ids is None else ids:
            value=MemoryEpisode.from_dict(self._episodes[identity]);site,example=self._example(value)
            if site==website and example is not None:result.append((identity,example))
        return result

    def _select(self,examples):
        if self.params["profile"] != "webarena":
            return examples
        groups = defaultdict(list)
        for row in examples:groups[(type(row[1]['template_id']).__name__,row[1]['template_id'])].append(row)
        generator = random.Random(self.params["seed"])
        return [
            row
            for values in groups.values()
            for row in generator.sample(values, min(self.params["num_samples"], len(values)))
        ]

    async def _learn(self,episode:MemoryEpisode,*,observed_event_ids:frozenset[str]):
        self._ensure_scope(episode.scope)
        if episode.episode_id in self._episodes:
            return

        website, example = self._example(episode)
        self._episodes[episode.episode_id] = episode.to_dict()
        self._order.append(episode.episode_id)
        if example is not None:
            self._pending[website] = self._pending.get(website, 0) + 1

        # Online mode induces at the configured batch boundary. The explicit
        # end marker also flushes incomplete website batches for both modes.
        due = (
            self.params["induction_mode"] == "online"
            and self._pending.get(website, 0) >= self.params["induce_steps"]
        )
        if due:
            await self._induce(website)
        if episode.extensions.get("awm", {}).get("end_of_training", False):
            for site in sorted(self._pending):
                await self._induce(site)
        self._persist()

    async def _induce(self,website):
        candidates = self._examples(website)
        selected = self._select(candidates)
        if not selected:
            return
        prompt = self.core.induction_prompt(
            [example for _, example in selected], self.params["induction_mode"]
        )
        options = {
            "temperature": 0.0 if self.params["profile"] == "mind2web" else 1.0
        }
        if self.params["profile"] == "webarena":
            options["max_tokens"] = 2048
        elif self.params["induction_mode"] == "offline":
            options["max_tokens"] = 1024
        messages = [{"role": "user", "content": prompt}]
        inducer=self.services.object('awm.inducer')
        response=(await call_maybe_async(inducer,prompt,self.params['profile'],run_sync_in_thread=True) if inducer else
                  await self.services.llm.complete(messages,**options))
        if not isinstance(response,str) or not response.strip():
            raise ValueError('AWM induction returned empty/nontext output')
        filtered=self.core.filter_response(response,website)
        source_ids=list(dict.fromkeys(event['event_id'] for identity,_ in selected for event in self._episodes[identity]['events']))
        self._workflow_text[website]=filtered;self._workflow_sources[website]=source_ids
        self._history.append({'website':website,'selected_episode_ids':[identity for identity,_ in selected],
            'absorbed_episode_ids':[identity for identity,_ in candidates], 'messages':messages,'options':options,
            'output':response,'filtered':filtered,'source_event_ids':source_ids})
        self._pending.pop(website,None)

    async def search(self,query,*, level_ref=None,top_k,scope,extensions):
        self._ensure_scope(scope, read_only=True)
        website = extensions["awm"]["website"]
        text = self._workflow_text.get(website, "")
        items = []
        if text:
            items.append({
                "memory_id": "awm-" + hashlib.sha256((website + "\n" + text).encode()).hexdigest(),
                "content": text,
                "memory_type": "workflow_file",
                "source_event_ids": list(self._workflow_sources[website]),
                "metadata": {
                    "method_id": "awm",
                    "website": website,
                    "profile": self.params["profile"],
                    "parsed_workflows": (
                        self.core.parse_clinical_workflows(text)
                        if self.params["profile"] == "medicine"
                        else self.core.parse_workflows(text, website)
                    ),
                    "provenance": "conservative_induction_input_dependencies",
                },
            })
            # Parser IDs are presentation-only; exclude random IDs to keep frozen results deterministic.
            for block in items[0]['metadata']['parsed_workflows']:block.pop('id',None)
        return MemoryContribution(
            items=items,
            extensions={
                "awm": {
                    "website": website,
                    "pending_induction": website in self._pending,
                    "retrieval": "complete_website_workflow_file",
                    "top_k_applies_to": "website files; one requested website",
                }
            },
        )

    def _serialize(self):
        return deepcopy({
            "version": 2,
            "profile": self.params["profile"],
            "induction_mode": self.params["induction_mode"],
            "episodes": self._episodes,
            "episode_order": self._order,
            "pending": self._pending,
            "workflow_text": self._workflow_text,
            "workflow_sources": self._workflow_sources,
            "induction_history": self._history,
            "active_scope": self._active_scope,
        })

    async def save(self):
        return self._serialize()

    def _restore(self,state):
        state = deepcopy(state)
        if (state.get("version") != 2
                or state.get("profile") != self.params["profile"]
                or state.get("induction_mode") != self.params["induction_mode"]):
            raise ValueError("AWM requires matching v2 state with complete provenance")

        episodes = state["episodes"]
        order = state["episode_order"]
        owner = state["active_scope"]
        if len(order) != len(episodes) or set(order) != set(episodes):
            raise ValueError("AWM episode order is incomplete")
        if owner is not None and scope_key(json.loads(owner)) != owner:
            raise ValueError("AWM invalid saved scope")
        if episodes and owner is None:
            raise ValueError("AWM saved scope is missing")

        examples = {}
        eligible = defaultdict(list)
        for identity in order:
            value = MemoryEpisode.from_dict(episodes[identity])
            if (value.episode_id != identity or value.scope is None
                    or scope_key(value.scope) != owner):
                raise ValueError("AWM saved episode identity/scope is invalid")
            website, example = self._example(value)
            if example is not None:
                examples[identity] = example
                eligible[website].append(identity)

        # Rebuild derived workflow data from its recorded inputs. Persisted text
        # is accepted only when it matches the original induction history.
        expected_text = {}
        expected_sources = {}
        absorbed = {}
        for row in state['induction_history']:
            website = row["website"]
            absorbed_ids = row["absorbed_episode_ids"]
            selected_ids = row["selected_episode_ids"]
            if (not absorbed_ids or len(set(absorbed_ids)) != len(absorbed_ids)
                    or any(identity not in examples
                           or examples[identity]["website"] != website
                           for identity in absorbed_ids)):
                raise ValueError("AWM induction source episodes are invalid")

            expected_selected = [
                identity
                for identity, _ in self._select([
                    (identity, examples[identity]) for identity in absorbed_ids
                ])
            ]
            if selected_ids != expected_selected:
                raise ValueError("AWM template/source selection is invalid")
            prompt = self.core.induction_prompt(
                [examples[identity] for identity in selected_ids],
                self.params["induction_mode"],
            )
            source_ids = list(dict.fromkeys(
                event["event_id"]
                for identity in selected_ids
                for event in episodes[identity]["events"]
            ))
            filtered = self.core.filter_response(row["output"], website)
            expected_messages = [{"role": "user", "content": prompt}]
            if (row["messages"] != expected_messages
                    or row["filtered"] != filtered
                    or row["source_event_ids"] != source_ids):
                raise ValueError("AWM workflow induction history/provenance is invalid")
            expected_text[website] = filtered
            expected_sources[website] = source_ids
            absorbed[website] = absorbed_ids

        pending = {
            website: len([
                identity for identity in identities
                if identity not in absorbed.get(website, [])
            ])
            for website, identities in eligible.items()
        }
        pending = {website: count for website, count in pending.items() if count}
        if (state["pending"] != pending
                or any(type(count) is not int for count in state["pending"].values())):
            raise ValueError("AWM pending induction state is invalid")
        if (state["workflow_text"] != expected_text
                or state["workflow_sources"] != expected_sources):
            raise ValueError("AWM saved workflows differ from induction history")

        self._episodes = episodes
        self._order = order
        self._pending = pending
        self._workflow_text = expected_text
        self._workflow_sources = expected_sources
        self._history = state["induction_history"]
        self._active_scope = owner

    async def load(self,state):
        self._restore(state)
        self._persist()

    async def reset(self,scope):
        self._ensure_scope(scope)
        self._episodes = {}
        self._order = []
        self._pending = {}
        self._workflow_text = {}
        self._workflow_sources = {}
        self._history = []
        self._active_scope = None
        self._persist()

    def _persist(self):
        if self._storage_path is not None and self._persistence_ready:
            atomic_write_json(self._storage_path,self._serialize())

    async def close(self):
        self._persist()


class AWMPlugin(MethodPlugin):
    descriptor = DESCRIPTOR
    qa_factory = "mematlas.methods.awm.qa:AWMQA"

    def create_method(self, config, services):
        return AWMMemory(config, services)

    def healthcheck(self, config, services):
        result = dict(super().healthcheck(config, services))
        if callable(services.object("awm.inducer")):
            result["missing_services"] = [
                name for name in result["missing_services"] if name != "llm"
            ]
            if (not result["missing_services"] and not result["missing_packages"]
                    and not result["config_errors"]):
                result.update(ready=True, status="ready")
        return result


PLUGIN=AWMPlugin()
