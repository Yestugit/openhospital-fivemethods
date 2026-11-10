"""Mind2Web task data, canonical training episodes and native web-action scoring.

The JSON manifest has disjoint ``train``/``test`` arrays of official records
with candidate ``rank`` values already attached. ``exemplars`` is an upstream
message-list bank or a JSON path relative to the manifest. Defaults match AWM's
runner: top_k_elements=5, previous_top_k_elements=3, retrieve_top_k=1.
``training_mode`` is ``canonical`` (annotated offline training) or ``agent``
(collect real predictions through the registered QA before learning).
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.models import MemoryEpisode, MemoryEvent
from .base import BenchmarkAdapter

UPSTREAM_REVISION = '8c0ff8cd11d648c8fceb99e4e42f37e3b75381b1'
PROTOCOL = 'mind2web_teacher_forcing_v1'


class Mind2WebBenchmark(BenchmarkAdapter):
    benchmark_id = 'mind2web'

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding='utf-8-sig'))
        if not isinstance(self.data, dict):
            raise ValueError('Mind2Web requires a manifest with train/test arrays and ranked official records')
        self.top_k = self.data.get('top_k_elements', 5)
        self.previous_top_k = self.data.get('previous_top_k_elements', 3)
        self.retrieve_top_k = self.data.get('retrieve_top_k', 1)
        self.max_context_tokens = self.data.get('max_context_tokens', 16385)
        self.seed = self.data.get('seed', 0)
        if any(type(n) is not int or n < 1 for n in (self.top_k, self.previous_top_k, self.max_context_tokens)):
            raise ValueError('Mind2Web candidate/context budgets must be positive integers')
        if type(self.retrieve_top_k) is not int or self.retrieve_top_k < 0 or type(self.seed) is not int:
            raise ValueError('Mind2Web retrieve_top_k must be nonnegative and seed must be an integer')
        self.training_mode = self.data.get('training_mode', 'canonical')
        if self.training_mode not in {'canonical', 'agent'}:
            raise ValueError('Mind2Web training_mode must be canonical or agent')
        self.rows: dict[str, dict[str, Any]] = {}
        for split in ('train', 'test'):
            values = self.data.get(split, [])
            if not isinstance(values, list):
                raise ValueError(f'Mind2Web {split} must be an array')
            for row in values:
                self._validate_row(row)
                identity = row['annotation_id']
                if identity in self.rows:
                    raise ValueError('Mind2Web task IDs must be unique and train/test disjoint')
                self.rows[identity] = row
        bank = self.data.get('exemplars', [])
        self.exemplar_path = None
        if isinstance(bank, str):
            self.exemplar_path = self.path.parent / bank
            bank = json.loads(self.exemplar_path.read_text(encoding='utf-8-sig'))
        if not isinstance(bank, list) or any(
            not isinstance(example, list) or not example or any(
                not isinstance(message, dict) or message.get('role') not in {'user', 'assistant'}
                or not isinstance(message.get('content'), str)
                or not isinstance(message.get('specifier', ''), str) for message in example)
            for example in bank
        ):
            raise ValueError('Mind2Web exemplars must be lists of user/assistant text messages')
        # Do not expose a test trajectory from a concrete-example bank. The
        # official bank can contain other training tasks not selected for learn.
        test_tasks = {row['confirmed_task'] for row in self.data.get('test', [])}
        if any(any('Task: ' + task in m.get('specifier', '') or 'Task: ' + task in m['content']
                   for m in example) for example in bank for task in test_tasks):
            raise ValueError('Mind2Web exemplar bank overlaps a selected test task')
        self.exemplars = bank

    @staticmethod
    def _validate_row(row: Any) -> None:
        for name in ('annotation_id', 'confirmed_task', 'website', 'domain', 'subdomain'):
            if not isinstance(row, dict) or not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f'Mind2Web record requires {name}:str')
        actions, representations = row.get('actions'), row.get('action_reprs')
        if not isinstance(actions, list) or not actions or not isinstance(representations, list) or len(actions) != len(representations):
            raise ValueError('Mind2Web actions/action_reprs must be nonempty aligned arrays')
        for step, representation in zip(actions, representations):
            if not isinstance(representation, str) or not representation:
                raise ValueError('Mind2Web action_reprs must contain text')
            if any(not isinstance(step.get(k), str) or not step[k] for k in ('action_uid', 'raw_html', 'cleaned_html')):
                raise ValueError('Mind2Web steps require action_uid/raw_html/cleaned_html')
            operation = step.get('operation', {})
            if operation.get('op') not in {'CLICK', 'TYPE', 'SELECT'} or not isinstance(operation.get('value'), str):
                raise ValueError('Mind2Web operation requires native op/value')
            for name in ('pos_candidates', 'neg_candidates'):
                if not isinstance(step.get(name), list):
                    raise ValueError(f'Mind2Web steps require {name}')
                for candidate in step[name]:
                    if (not isinstance(candidate.get('backend_node_id'), str)
                            or type(candidate.get('rank')) is not int or candidate['rank'] < 0):
                        raise ValueError('Mind2Web candidates need backend_node_id:str and rank:int; attach official ranks first')

    def cases(self, split: str = 'test'):
        if split not in {'train', 'test', 'all'}:
            raise ValueError('Mind2Web split must be train/test/all')
        for part in ('train', 'test'):
            if split not in {part, 'all'}:
                continue
            for row in self.data.get(part, []):
                scope = {'user_id': 'awm-mind2web'}
                episode = None
                options = {k: row[k] for k in ('website', 'domain', 'subdomain')}
                if part == 'train' and self.training_mode == 'canonical':
                    episode = MemoryEpisode(episode_id='mind2web-canonical:' + row['annotation_id'],
                        task=row['confirmed_task'], scope=scope,
                        events=[MemoryEvent('action', text, role='assistant',
                            event_id=f"mind2web:{row['annotation_id']}:{i}",
                            extensions={'awm': {'action_repr': text}})
                            for i, text in enumerate(row['action_reprs'])],
                        extensions={'awm': {**options, 'canonical': True}})
                yield BenchmarkCase(row['annotation_id'], row['confirmed_task'], scope=scope,
                    episode=episode, metadata={'benchmark_id': 'mind2web', 'split': part,
                        'protocol': PROTOCOL, 'awm': options,
                        'memory_extensions': {'awm': {'website': row['website']}}})

    def environment(self, case_id: str):
        from .mind2web_env import Mind2WebEnvironment
        return Mind2WebEnvironment(self.rows[case_id], top_k=self.top_k, previous_top_k=self.previous_top_k)

    def concrete_examples(self, case_id: str) -> list[list[dict[str, str]]]:
        """Use upstream website/subdomain fallback, with a reproducible local RNG."""
        row = self.rows[case_id]
        values = self.exemplars
        if any(row['website'] in example[0].get('specifier', '') for example in values):
            values = [e for e in values if all(row[k] in e[0].get('specifier', '')
                      for k in ('domain', 'subdomain', 'website'))]
        elif any(row['subdomain'] in example[0].get('specifier', '') for example in values):
            values = [e for e in values if all(row[k] in e[0].get('specifier', '')
                      for k in ('domain', 'subdomain'))]
        # Exclude the task being acted on, including online training queries.
        values = [e for e in values if all('Task: ' + row['confirmed_task'] not in m.get('specifier', '')
                  and 'Task: ' + row['confirmed_task'] not in m['content'] for m in e)]
        selected = random.Random(f'{self.seed}:{case_id}').sample(values, min(self.retrieve_top_k, len(values)))
        return [[{k: v for k, v in m.items() if k != 'specifier'} for m in e] for e in selected]

    def native_score(self, predictions):
        scores, successes = [], []
        for prediction in predictions:
            expected = self.rows[prediction.case_id]['actions']
            data = prediction.metadata.get('episode', {}).get('extensions', {}).get('awm', {}).get('mind2web', {})
            steps = data.get('step_scores', [])
            if data.get('protocol') != PROTOCOL or len(steps) != len(expected):
                if not prediction.metadata.get('evaluation_error'):
                    raise ValueError('Mind2Web scoring needs a complete registered native QA episode')
                steps = [{'element_acc': 0.0, 'action_f1': 0.0, 'step_success': 0.0,
                          'reason': 'execution_error'} for _ in expected]
            scores.extend(steps)
            successes.append(float(all(s['step_success'] == 1 for s in steps)))
        count = len(scores)
        return {'element_acc': sum(s['element_acc'] for s in scores) / count if count else 0.0,
                'action_f1': sum(s['action_f1'] for s in scores) / count if count else 0.0,
                'step_success_rate': sum(s['step_success'] for s in scores) / count if count else 0.0,
                'success_rate': sum(successes) / len(successes) if successes else 0.0,
                'task_count': len(successes), 'step_count': count,
                'candidate_misses': sum(s['reason'] == 'candidate_rank' for s in scores)}

    def describe(self):
        return {'benchmark_id': self.benchmark_id, 'path': str(self.path),
                'sha256': self.file_sha256(self.path), 'protocol': PROTOCOL,
                'live_browser': False, 'upstream_revision': UPSTREAM_REVISION,
                'train_count': len(self.data.get('train', [])), 'test_count': len(self.data.get('test', [])),
                'training_mode': self.training_mode, 'top_k_elements': self.top_k,
                'previous_top_k_elements': self.previous_top_k, 'retrieve_top_k': self.retrieve_top_k,
                'max_context_tokens': self.max_context_tokens, 'seed': self.seed,
                'exemplar_count': len(self.exemplars),
                'exemplars_sha256': self.file_sha256(self.exemplar_path) if self.exemplar_path else None,
                'sampling': 'local seeded RNG per case; not upstream global random stream',
                'requirements': ['ranked official Mind2Web JSON', 'lxml', 'tiktoken', 'answer LLM']}
