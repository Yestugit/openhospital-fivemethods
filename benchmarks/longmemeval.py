from __future__ import annotations

import inspect
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Awaitable, Callable, Iterable, Mapping

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.core.evaluation import BenchmarkCase, Prediction
from mematlas.core.models import EventKind, MemoryEvent, Scope
from mematlas.core.contributions import applied_sources
from mematlas.core.plugin import LLMService


Judge = Callable[[Prediction], bool | str | Mapping[str, Any] | Awaitable[Any]]


def _timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for pattern in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def answer_check_prompt(task: str, question: str, answer: str, response: str, *, abstention: bool) -> str:
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. The model "
            "could say that the information is incomplete, or some other information is given but the asked "
            "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        ).format(question, answer, response)
    common = (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
        "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
        "the correct answer or contains all the intermediate steps to get the correct answer, you should also "
        "answer yes. If the response only contains a subset of the information required by the answer, answer no."
    )
    if task == "temporal-reasoning":
        common += (
            " In addition, do not penalize off-by-one errors for the number of days. If the question asks for "
            "the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still correct."
        )
    elif task == "knowledge-update":
        common = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response should be considered as correct "
            "as long as the updated answer is the required answer."
        )
    elif task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response from a "
            "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
            "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        ).format(question, answer, response)
    elif task not in {"single-session-user", "single-session-assistant", "multi-session"}:
        raise ValueError(f"Unsupported LongMemEval question type: {task}")
    return (
        f"{common}\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    )


class LongMemEvalLLMJudge:
    def __init__(self, llm: LLMService, *, model: str = "qwen3.5-flash") -> None:
        self.llm = llm
        self.model = model

    async def __call__(self, prediction: Prediction) -> bool:
        metadata = prediction.metadata.get("case_metadata", {})
        prompt = answer_check_prompt(
            str(metadata["question_type"]),
            str(metadata["question"]),
            str(prediction.reference),
            str(prediction.output),
            abstention=bool(metadata.get("abstention")),
        )
        from mematlas.providers.instrumentation import MeteredService
        from mematlas.core.telemetry import active_usage
        active = active_usage()
        provider = (MeteredService(self.llm, active[0], "llm")
                    if active is not None and not isinstance(self.llm, MeteredService) else self.llm)
        response = await provider.complete(
            [{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0,
            max_tokens=10,
        )
        return _judge_label(response)


def _judge_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        if "label" in value:
            return _judge_label(value["label"])
        if "correct" in value:
            return _judge_label(value["correct"])
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        label = value.strip().lower().strip(" .!?`'\"")
        if label in {"yes", "true", "1"}:
            return True
        if label in {"no", "false", "0"}:
            return False
    raise ValueError(f"Unrecognized LongMemEval judge label: {value!r}; expected yes/no or true/false")


RETRIEVAL_K_VALUES = (1, 3, 5, 10, 30, 50)


def _ranked_retrieved_sessions(prediction: Prediction) -> list[str]:
    sessions: list[str] = []
    for item in applied_sources(prediction.applied_memory):
        candidates: list[str] = []
        metadata = item.get("metadata") or {}
        for key in ("session_id", "source_session_id"):
            if metadata.get(key):
                candidates.append(str(metadata[key]))
        for source_id in item.get("source_event_ids", []):
            candidates.append(str(source_id).split(":turn:", 1)[0])
        for key in ("source_event_ids", "mematlas_event_ids", "event_ids"):
            values = metadata.get(key) or []
            if isinstance(values, str):
                values = [values]
            candidates.extend(str(value).split(":turn:", 1)[0] for value in values)
        for candidate in candidates:
            if candidate and candidate not in sessions:
                sessions.append(candidate)
    return sessions


def _retrieval_metrics(ranked: list[str], expected: set[str], k: int) -> tuple[float, float, float]:
    selected = ranked[:k]
    selected_set = set(selected)
    recall_any = float(bool(expected & selected_set))
    recall_all = float(expected <= selected_set)
    gains = [1.0 if item in expected else 0.0 for item in selected]

    def dcg(values: list[float]) -> float:
        return sum(
            value / math.log2(index + 2)
            for index, value in enumerate(values)
        )

    ideal_dcg = dcg([1.0] * min(len(expected), k))
    ndcg = dcg(gains) / ideal_dcg if ideal_dcg else 0.0
    return recall_any, recall_all, ndcg


class LongMemEvalBenchmark(BenchmarkAdapter):
    benchmark_id = "longmemeval"

    def __init__(
        self,
        path: str | Path,
        *,
        judge: Judge | None = None,
        limit: int | None = None,
        top_k: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.judge = judge
        self.limit = limit
        self.top_k = top_k
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("LongMemEval data file must contain a list")
        self._data = payload[:limit] if limit is not None else payload

    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]:
        if split not in {"test", "all"}:
            return
        for row in self._data:
            question_id = str(row["question_id"])
            scope = {"user_id": f"longmemeval:{question_id}", "agent_id": "longmemeval-reader"}
            events: list[MemoryEvent] = []
            sessions = row.get("haystack_sessions", [])
            session_ids = row.get("haystack_session_ids", [])
            dates = row.get("haystack_dates", [])
            for session_index, session in enumerate(sessions):
                session_id = str(session_ids[session_index]) if session_index < len(session_ids) else str(session_index)
                session_date = dates[session_index] if session_index < len(dates) else None
                base_time = _timestamp(session_date)
                for turn_index, turn in enumerate(session):
                    event_id = f"{session_id}:turn:{turn_index}"
                    events.append(MemoryEvent(
                        kind=EventKind.MESSAGE,
                        scope=scope,
                        content=str(turn.get("content", "")),
                        event_id=event_id,
                        timestamp=base_time + timedelta(microseconds=turn_index),
                        role=str(turn.get("role", "user")),
                        extensions={
                            "benchmark": self.benchmark_id,
                            "session_id": session_id,
                            "session_date": session_date,
                            "has_answer": bool(turn.get("has_answer", False)),
                        },
                    ))
            question_type = str(row["question_type"])
            yield BenchmarkCase(
                case_id=question_id,
                scope=scope,
                history=tuple(events),
                query=str(row["question"]),
                reference=row.get("answer"),
                metadata={
                    "question": row["question"],
                    "question_type": question_type,
                    "question_date": row.get("question_date"),
                    "answer_session_ids": list(row.get("answer_session_ids", [])),
                    "abstention": "_abs" in question_id,
                    "top_k": self.top_k,
                    "split": "test",
                },
            )

    def native_score(self, predictions: list[Prediction]) -> Mapping[str, Any]:
        fractional_recalls = []
        official: dict[str, list[float]] = defaultdict(list)
        for prediction in predictions:
            metadata = prediction.metadata.get("case_metadata", {})
            expected = {str(value) for value in metadata.get("answer_session_ids", [])}
            if not expected or metadata.get("abstention"):
                continue
            ranked = _ranked_retrieved_sessions(prediction)
            fractional_recalls.append(len(expected & set(ranked)) / len(expected))
            for k in RETRIEVAL_K_VALUES:
                recall_any, recall_all, ndcg = _retrieval_metrics(ranked, expected, k)
                official[f"session_recall_any@{k}"].append(recall_any)
                official[f"session_recall_all@{k}"].append(recall_all)
                official[f"session_ndcg@{k}"].append(ndcg)
        retrieval_values = {}
        for k in RETRIEVAL_K_VALUES:
            for name in (
                f"session_recall_any@{k}",
                f"session_recall_all@{k}",
                f"session_ndcg@{k}",
            ):
                values = official.get(name, [])
                retrieval_values[name] = mean(values) if values else None
        return {
            "session_recall": mean(fractional_recalls) if fractional_recalls else None,
            "retrieval_question_count": len(fractional_recalls),
            **retrieval_values,
            "judge_accuracy": None,
            "judge_question_count": 0,
            "judge_task_accuracy": None,
            "judge_task_question_count": 0,
            "judge_abstention_accuracy": None,
            "judge_abstention_question_count": 0,
        }

    async def native_score_async(self, predictions: list[Prediction]) -> Mapping[str, Any]:
        result = dict(self.native_score(predictions))
        if self.judge is None:
            return result
        labels = []
        task_labels = []
        abstention_labels = []
        by_type: dict[str, list[bool]] = defaultdict(list)
        for prediction in predictions:
            value = self.judge(prediction)
            if inspect.isawaitable(value):
                value = await value
            label = _judge_label(value)
            labels.append(label)
            metadata = prediction.metadata.get("case_metadata", {})
            question_type = str(metadata.get("question_type", "unknown"))
            by_type[question_type].append(label)
            if metadata.get("abstention"):
                abstention_labels.append(label)
            else:
                task_labels.append(label)
        result["judge_accuracy"] = mean(labels) if labels else 0.0
        result["judge_question_count"] = len(labels)
        result["judge_task_accuracy"] = mean(task_labels) if task_labels else None
        result["judge_task_question_count"] = len(task_labels)
        result["judge_abstention_accuracy"] = mean(abstention_labels) if abstention_labels else None
        result["judge_abstention_question_count"] = len(abstention_labels)
        result.update({f"judge_accuracy.{key}": mean(values) for key, values in sorted(by_type.items())})
        return result

    def describe(self) -> Mapping[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "path": str(self.path),
            "sha256": self.file_sha256(self.path),
            "record_count": len(self._data),
            "top_k": self.top_k,
            "judge_enabled": self.judge is not None,
            "official_metric": "LongMemEval LLM answer judge plus session Recall-any/Recall-all/NDCG at k",
        }
