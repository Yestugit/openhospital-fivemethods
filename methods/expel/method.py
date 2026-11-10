"""ExpeL 公共接口适配：收集真实经历、在训练结束时提炼规则并检索成功示例。

规则提炼由 learn 内部触发；Agent 执行、工具调用和成败评价在 Memory 外部完成。
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import math
import random
from typing import Any, Mapping, TypedDict

from mematlas.core.config import MethodConfig
from mematlas.core.models import Content, MemoryContribution, MemoryEpisode, MemoryEvent, Scope, content_text
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodPlugin, MethodServices
from mematlas.core.scope import scope_key
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services
from mematlas.methods.storage import ascii_database_path, atomic_write_json, configured_file, read_json
from .core import ExpeLCore
from .retrieval import ExperienceIndex
from .utils import environment_name, strip_task, strip_component

DESCRIPTOR = load_method_descriptor(__file__)


class AddOptions(TypedDict, total=False):
    """learn 的专有数据，放入 MemoryUnit.extensions['expel']。

    Attributes:
        end_of_training: bool，默认 False；由 Agent 或调度器在最后一份训练经历设置。
            True 时 learn 对当前已收集经历执行完整规则提炼；save/retrieve 不触发。
        reflections: list[str]，默认 []；使用 Agent 实际生成的反思，作为反思检索键。
            Memory 不生成反思；只使用规则或其他检索策略时可省略。
        env_name: str，可省略，来自实际任务的环境类别，用于限定示例检索范围。
            ALFWorld 默认按任务词推断，其他任务默认使用配置中的 benchmark。

    success、task、events 使用 MemoryEpisode 公共字段，不放入本结构。
    """
    end_of_training: bool
    reflections: list[str]
    env_name: str


class SearchOptions(TypedDict, total=False):
    """检索时的执行阶段数据，传入 retrieve(..., extensions={'expel': ...})。

    Attributes:
        thought: str，默认空字符串，来自 Agent 当前实际思考；思考及反思相似度检索使用。
        action: str，默认空字符串，来自 Agent 当前实际动作；动作相似度检索使用。
        step: str，默认空字符串，来自 Agent 当前实际完整步骤；步骤相似度检索使用。
        last_step_type: str，可省略，来自 Agent 当前阶段标记；rotation 据此选择
            thought 或 observation/step，其他阶段使用 task。
        env_name: str，可省略，来自当前任务的环境类别，应与学习时一致；
            默认按 query 和配置中的 benchmark 推断。

    按 fewshot_strategy 提供对应字段；直接按组件检索时，空组件不会用任务文本替代。
    reflection_similarity 为显式扩展，使用当前 thought 检索已保存的真实反思。
    """
    thought: str
    action: str
    step: str
    last_step_type: str
    env_name: str


class ExpeLMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params, self.services = dict(config.params), services
        self.core = ExpeLCore(self.params)
        self._successes, self._failures = defaultdict(list), defaultdict(list)
        self._task_order: list[str] = []
        self._seen: set[str] = set()
        self._rules_with_count: list[tuple[str, int]] = []
        self._rule_sources: dict[str, list[str]] = {}
        self._critique_history: list[dict[str, Any]] = []
        self._active_scope: str | None = None
        self._dirty = False
        self._storage_path = None
        self.index = None

    def _ensure_scope(self, scope: Scope, *, read_only: bool = False) -> None:
        key = scope_key(scope)
        if self._active_scope is None:
            if not read_only:
                self._active_scope = key
        elif key != self._active_scope:
            raise ValueError('ExpeL uses one Memory instance per scope')

    async def setup(self, context: MethodContext) -> None:
        prepare_services(self.params, self.services, method_id='expel', llm=True,
                         embedding=self.params.get('fewshot_strategy', 'task_similarity') != 'none')
        self._storage_path = configured_file(self.services, 'expel', self.params.get('storage_path'), 'expel.json')
        self.index = ExperienceIndex(self.services, self.params['benchmark'],
            ascii_database_path(self._storage_path, 'expel.faiss', services=self.services))
        if self._storage_path.exists():
            self._restore(read_json(self._storage_path, {}))

    async def _observe(self, event: MemoryEvent) -> None:
        # The public runtime journals the real event, pending explicit outcome feedback.
        self._ensure_scope(event.scope)

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids: frozenset[str]) -> None:
        self._ensure_scope(episode.scope)
        if episode.episode_id in self._seen:
            return
        if episode.success is None or not episode.task or not episode.task.strip() or not episode.events:
            raise ValueError('ExpeL requires real success, task and trajectory events')
        text = '\n'.join(content_text(event.content) for event in episode.events if content_text(event.content).strip())
        if not text.strip():
            raise ValueError('ExpeL requires a nonempty real trajectory')
        options = episode.extensions.get('expel', {})
        if any(not isinstance(value,str) for value in options.get('reflections', [])):
            raise ValueError('ExpeL reflections must contain real text strings')
        trial = dict(id=episode.episode_id, task=episode.task, content=text, success=episode.success,
            reward=episode.reward, feedback=episode.feedback,
            reflections=list(options.get('reflections', [])),
            env_name=options.get('env_name', environment_name(episode.task, self.params['benchmark'])),
            source_event_ids=[event.event_id for event in episode.events])
        if episode.task not in self._task_order:
            self._task_order.append(episode.task)
        if trial['success'] and self.params.get('fewshot_strategy', 'task_similarity') != 'none':
            await self.index.add_success(trial)
        (self._successes if trial['success'] else self._failures)[trial['task']].append(trial)
        self._seen.add(episode.episode_id)
        self._dirty = True
        if options.get('end_of_training', False):
            await self._compile_rules()
        self._persist()

    async def _compile_rules(self) -> None:
        # Match a fresh upstream create_rules pass over the collected train set.
        # All work is staged locally; public Memory rolls back learn on exceptions.
        rules, sources, history = [], {}, []

        async def extend(successes, failures=(), task=None):
            nonlocal rules, sources
            success_text = (successes[0]['content'].strip() if failures else '\n\n'.join(
                f"{strip_task(t['task'], self.params['benchmark'])}\n{t['content']}" for t in successes).strip())
            failure_text = failures[0]['content'].strip() if failures else None
            messages = self.core.critique_messages(benchmark=self.params['benchmark'], success_history=success_text,
                fail_history=failure_text, task=strip_task(task, self.params['benchmark']) if task else None,
                existing_rules=[rule for rule, _ in rules], max_num_rules=self.params.get('max_num_rules', 20))
            output = await self.services.llm.complete(messages, temperature=0)
            operations = self.core.parse_rules(output)
            before = deepcopy(rules)
            dependencies = list(dict.fromkeys([*(value for values in sources.values() for value in values),
                *(identity for trial in [*successes, *failures] for identity in trial['source_event_ids'])]))
            rules = self.core.update_rules(rules, operations,
                list_full=len(rules) >= self.params.get('max_num_rules', 20) + 5)
            # Conservative decision-input lineage, not a claim of textual fact coverage.
            sources = {rule: list(dependencies) for rule, _ in rules}
            history.append(dict(stage='compare' if failures else 'success', task=task,
                source_event_ids=dependencies, messages=messages, output=str(output), operations=operations,
                before=[list(row) for row in before], after=[list(row) for row in rules]))

        for task in self._task_order:
            for success in self._successes.get(task, []):
                for failure in self._failures.get(task, []):
                    await extend([success], [failure], task)
        first_successes = [self._successes[task][0] for task in self._task_order if self._successes.get(task)]
        random.Random(self.params.get('seed', 0)).shuffle(first_successes)
        limit = self.params.get('success_critique_num', 8)
        if first_successes:
            chunk_size = math.ceil(len(first_successes) / math.ceil(len(first_successes) / limit))
            for offset in range(0, len(first_successes), chunk_size):
                await extend(first_successes[offset:offset+chunk_size])
        self._rules_with_count, self._rule_sources = rules, sources
        self._critique_history.extend(history)
        self._dirty = False

    async def search(self, query: Content, *, level_ref=None, top_k: int | None, scope: Scope,
                       extensions: Mapping[str, Any]) -> MemoryContribution:
        self._ensure_scope(scope, read_only=True)
        items, contexts = [], []
        for rule, count in self._rules_with_count:
            items.append(dict(content=rule, memory_id='rule-'+hashlib.sha256(rule.encode()).hexdigest(),
                score=float(count), memory_type='insight', source_event_ids=self._rule_sources[rule],
                metadata={'method_id':'expel', 'count':count, 'provenance':'conservative_rule_input_dependencies'}))
        if items:
            contexts.append(dict(content=self.core.format_rules(self.params['benchmark'], [r for r, _ in self._rules_with_count]),
                                 source_item_indices=list(range(len(items))), kind='rules'))
        options = extensions.get('expel', {})
        task = content_text(query)
        strategy = self.params.get('fewshot_strategy', 'task_similarity')
        number = top_k if top_k is not None else self.params.get('num_fewshots')
        if number is None:
            number = self.core.DEFAULT_FEWSHOTS[self.params['benchmark']]
        field = strategy.replace('_similarity', '')
        if strategy == 'task_thought_similarity':
            field = 'thought' if options.get('thought') else 'task'
        elif strategy == 'rotation':
            last = options.get('last_step_type')
            field = ('thought' if last == 'thought' else 'step' if last in {'observation','step'} else 'task') if options.get('thought') else 'task'
        elif strategy in {'random','none'}:
            field = 'task'
        query_text = strip_task(task, self.params['benchmark']) if field == 'task' else options.get('thought' if field == 'reflection' else field, '')
        query_text = strip_component(query_text, 'thought' if field == 'reflection' else field, self.params['benchmark'])
        env = options.get('env_name', environment_name(task, self.params['benchmark']))
        candidates = (await self.index.search(query_text, field, env, self._successes,
            number*self.params.get('buffer_retrieve_ratio', 4)) if strategy != 'none' and query_text and number else [])
        if strategy == 'random':
            random.Random(self.params.get('seed', 0)).shuffle(candidates)
        import tiktoken
        encoding = tiktoken.get_encoding('cl100k_base')
        tokens = lambda text: len(encoding.encode(text, disallowed_special=()))
        reranker = self.params.get('reranker', 'none')
        distance_kind = 'l2'
        if reranker == 'len':
            candidates.sort(key=lambda row: tokens(self._successes[row[0]['task']][0]['content']), reverse=True)
        elif candidates and reranker in {'thought','task'} and (options.get('thought') or reranker == 'task'):
            rerank_query = strip_component(options['thought'] if reranker == 'thought' else strip_task(task,self.params['benchmark']),
                                          reranker, self.params['benchmark'])
            candidates = await self.index.rerank(candidates, rerank_query)
            distance_kind = 'cosine'
        max_tokens = self.params.get('max_fewshot_tokens') or self.core.get_fewshot_max_tokens(self.params['benchmark'])
        selected = set()
        for doc, distance in candidates:
            if doc['task'] == task or doc['task'] in selected:
                continue
            trial = min(self._successes[doc['task']], key=lambda row: len(row['content']))
            if tokens(trial['content']) > max_tokens:
                continue
            text = strip_task(trial['task'], self.params['benchmark'])+'\n'+trial['content']
            contexts.append(dict(content=text, source_item_indices=[len(items)], kind='successful_trial'))
            items.append(dict(content=text, memory_id=trial['id'], score=1.0/(1.0+max(0.0, distance)),
                source_event_ids=trial['source_event_ids'], memory_type='successful_trial',
                metadata={'method_id':'expel','task':trial['task'], 'distance':distance,'retrieval_field':field,
                    'matched_document_id':doc['id'], 'distance_kind':distance_kind}))
            selected.add(trial['task'])
            if len(selected) >= number:
                break
        return MemoryContribution(items=items, context=contexts,
            extensions={'expel':{'profile':self.params['benchmark'], 'pending_induction':self._dirty,
                'top_k_applies_to':'successful_trials; all learned rules are preserved'}})

    async def reset(self, scope: Scope) -> None:
        self._ensure_scope(scope)
        self._restore(dict(version=2, successes={}, failures={}, task_order=[], seen_episodes=[],
            rules_with_count=[], rule_sources={}, critique_history=[], documents=[], dirty=False, active_scope=None))
        self._persist()

    async def save(self) -> Mapping[str, Any]:
        return deepcopy(dict(version=2, successes=dict(self._successes), failures=dict(self._failures),
            task_order=self._task_order, seen_episodes=sorted(self._seen), rules_with_count=[list(row) for row in self._rules_with_count],
            rule_sources=self._rule_sources, critique_history=self._critique_history, documents=self.index.documents,
            dirty=self._dirty, active_scope=self._active_scope))

    async def load(self, state: Mapping[str, Any]) -> None:
        self._restore(state)
        self._persist()

    def _restore(self, state):
        state = deepcopy(state)
        if state.get('version') != 2:
            raise ValueError('ExpeL requires state version 2 with vectors and source dependencies')
        trials = [trial for group in ('successes','failures') for rows in state[group].values() for trial in rows]
        if {trial['id'] for trial in trials} != set(state['seen_episodes']) or len(trials) != len(state['seen_episodes']):
            raise ValueError('ExpeL saved trial identities are inconsistent')
        for group, success in [('successes',True),('failures',False)]:
            if any(not isinstance(task,str) or not task.strip() or not rows for task,rows in state[group].items()):
                raise ValueError('ExpeL saved task groups are invalid')
            if any(trial['success'] is not success or trial['task'] != task or not trial['source_event_ids']
                   or not isinstance(trial['content'],str) or not trial['content'].strip()
                   or not isinstance(trial['reflections'],list) or any(not isinstance(r,str) for r in trial['reflections'])
                   or any(not isinstance(identity,str) or not identity for identity in trial['source_event_ids'])
                   for task, rows in state[group].items() for trial in rows):
                raise ValueError('ExpeL saved trial feedback/task/provenance is invalid')
        task_keys = set(state['successes']) | set(state['failures'])
        if len(state['task_order']) != len(task_keys) or set(state['task_order']) != task_keys:
            raise ValueError('ExpeL saved task_order must cover every task exactly once')
        if type(state['dirty']) is not bool or trials and state['active_scope'] is None:
            raise ValueError('ExpeL saved training/scope state is invalid')
        source_ids = {identity for trial in trials for identity in trial['source_event_ids']}
        rules = state['rules_with_count']
        if any(not isinstance(rule, str) or type(count) is not int or count < 1 for rule,count in rules):
            raise ValueError('ExpeL rule counts are invalid')
        if set(state['rule_sources']) != {rule for rule,_ in rules} or any(
            not set(ids) <= source_ids for ids in state['rule_sources'].values()):
            raise ValueError('ExpeL rule sources are invalid')
        owner = state['active_scope']
        import json
        if owner is not None and scope_key(json.loads(owner)) != owner:
            raise ValueError('ExpeL saved scope is invalid')
        expected = []
        if self.params.get('fewshot_strategy', 'task_similarity') != 'none':
            for rows in state['successes'].values():
                for index, trial in enumerate(rows):
                    expected.extend(self.index.trial_documents(trial, include_task=index==0))
        actual = [{k:v for k,v in doc.items() if k!='embedding'} for doc in state['documents']]
        by_id = lambda docs: sorted(docs, key=lambda doc:doc['id'])
        if by_id(actual) != by_id(expected):
            raise ValueError('ExpeL index documents do not match complete native successful-trial components')
        self.index.restore(state['documents'])
        self._successes, self._failures = defaultdict(list,state['successes']), defaultdict(list,state['failures'])
        self._task_order, self._seen = state['task_order'], set(state['seen_episodes'])
        self._rules_with_count, self._rule_sources = [tuple(row) for row in rules], state['rule_sources']
        self._critique_history, self._dirty, self._active_scope = state['critique_history'], state['dirty'], owner

    def _persist(self):
        if self._storage_path is not None:
            state = dict(version=2,successes=dict(self._successes),failures=dict(self._failures),task_order=self._task_order,
                seen_episodes=sorted(self._seen),rules_with_count=self._rules_with_count,rule_sources=self._rule_sources,
                critique_history=self._critique_history,documents=self.index.documents,dirty=self._dirty,active_scope=self._active_scope)
            atomic_write_json(self._storage_path, state)
            self.index.persist()

    async def close(self) -> None:
        # A pending finite batch is saved exactly; closing never induces rules.
        self._persist()


class ExpeLPlugin(MethodPlugin):
    descriptor = DESCRIPTOR
    qa_factory = 'mematlas.methods.expel.qa:ExpeLQA'

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return ExpeLMemory(config, services)

    def healthcheck(self, config, services):
        result = dict(super().healthcheck(config, services))
        if config.params.get('fewshot_strategy') == 'none':
            result['missing_services'] = [name for name in result['missing_services'] if name != 'embedding']
            if not result['missing_services'] and not result['missing_packages'] and not result['config_errors']:
                result.update(ready=True,status='ready')
        return result


PLUGIN = ExpeLPlugin()
