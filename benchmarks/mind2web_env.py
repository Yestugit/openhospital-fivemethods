"""Mind2Web's offline teacher-forced evaluator, independent of Memory and LLMs.

Follows AWM 8c0ff8c mind2web/memory.py::eval_sample: candidate misses score
zero without acting; every subsequent state/history comes from annotations.
Gold actions are exposed only as *past* history, after that step is scored.
This is the dataset's evaluation protocol, not a live browser simulator.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from .mind2web_dom import (
    calculate_f1, construct_act_str, get_target_obs_and_act, get_top_k_obs, parse_act_str,
)


def extract_action(response: str) -> str:
    """Match upstream extract_from_response(response, '`'), including failures."""
    match = re.search(r"`(.*?)`", response, re.DOTALL)
    return match.group(1) if match else ""


class Mind2WebEnvironment:
    """Score all recorded steps, even after an invalid or incorrect action."""

    def __init__(self, row: dict[str, Any], *, top_k: int = 5, previous_top_k: int = 3):
        self.row = row
        self.top_k, self.previous_top_k = top_k, previous_top_k
        self.index = 0
        self.history: list[dict[str, str]] = []
        self.scores: list[dict[str, Any]] = []

    def reset(self) -> dict[str, Any]:
        self.index, self.history, self.scores = 0, [], []
        return self._state()

    def _eligible(self) -> bool:
        return any(c['rank'] < self.top_k for c in self.row['actions'][self.index]['pos_candidates'])

    def _state(self) -> dict[str, Any]:
        if self.index == len(self.row['actions']):
            return {'done': True, 'success': all(s['step_success'] == 1 for s in self.scores),
                    'step_scores': deepcopy(self.scores)}
        can_act = self._eligible()
        # Upstream requests cleaned HTML for eligible current steps; raw fallback
        # is used only for canonical history, including missing-positive steps.
        observation = (get_top_k_obs(self.row['actions'][self.index], self.top_k, use_raw=False)[0]
                       if can_act else '')
        return {'done': False, 'can_act': can_act, 'observation': observation,
                'history': deepcopy(self.history), 'step_index': self.index}

    def step(self, action: str | None, *, reason: str | None = None) -> dict[str, Any]:
        if self.index >= len(self.row['actions']):
            raise ValueError('Mind2Web task is already finished')
        step = self.row['actions'][self.index]
        eligible = self._eligible()
        if not eligible and action is not None:
            raise ValueError('Candidate rank miss must be skipped without a model action')
        _, target = get_target_obs_and_act(step)
        if eligible and action is not None:
            op, node, value = parse_act_str(action)
            target_op, _, target_value = parse_act_str(target)
            element = float(node in [c['backend_node_id'] for c in step['pos_candidates']][:1])
            f1 = float(calculate_f1(construct_act_str(op, value),
                                    construct_act_str(target_op, target_value)))
            success = float(action == target)
            reason = reason or ('correct' if success else 'invalid_action' if op is None else 'wrong_action')
        else:
            element = f1 = success = 0.0
            reason = 'candidate_rank' if not eligible else reason or 'context_limit'
        self.scores.append({'step_index': self.index, 'action_uid': step['action_uid'],
                            'prediction': action, 'target': target, 'element_acc': element,
                            'action_f1': f1, 'step_success': success, 'reason': reason})
        previous = get_top_k_obs(step, self.previous_top_k, use_raw=True)[0]
        self.history.extend([
            {'role': 'user', 'content': 'Observation: `' + previous + '`'},
            {'role': 'assistant', 'content': 'Action: `' + target + '` ('
             + self.row['action_reprs'][self.index] + ')'},
        ])
        self.index += 1
        return self._state()
