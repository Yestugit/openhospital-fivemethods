from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.core.evaluation import BenchmarkCase, Prediction
from mematlas.core.models import EventKind, MemoryEvent, Scope
from mematlas.core.contributions import applied_sources


def _parse_session_time(value: str) -> datetime:
    normalized = " ".join(value.strip().split())
    for pattern in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
        try:
            return datetime.strptime(normalized, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _stemmer():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as exc:
        raise RuntimeError("LoCoMo official F1 requires the 'nltk' package") from exc
    return PorterStemmer()


def _normalize(value: Any) -> str:
    text = str(value).replace(",", "").lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: Any, reference: Any) -> float:
    stem = _stemmer().stem
    predicted = [stem(token) for token in _normalize(prediction).split()]
    expected = [stem(token) for token in _normalize(reference).split()]
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = Counter(predicted) & Counter(expected)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def _locomo_score(prediction: Any, reference: Any, category: int) -> float:
    output = str(prediction)
    answer = str(reference)
    if category == 1:
        predicted_parts = [item.strip() for item in output.split(",")]
        answer_parts = [item.strip() for item in answer.split(",")]
        if not answer_parts:
            return 0.0
        return mean(max(_token_f1(candidate, expected) for candidate in predicted_parts) for expected in answer_parts)
    if category in {2, 3, 4}:
        if category == 3:
            answer = answer.split(";", 1)[0].strip()
        return _token_f1(output, answer)
    if category == 5:
        lowered = output.lower()
        return float("no information available" in lowered or "not mentioned" in lowered)
    raise ValueError(f"Unsupported LoCoMo category: {category}")


def _retrieved_source_ids(prediction: Prediction) -> set[str]:
    result: set[str] = set()
    for item in applied_sources(prediction.applied_memory):
        result.update(str(value) for value in item.get("source_event_ids", []))
        metadata = item.get("metadata") or {}
        for key in ("dia_id", "event_id", "source_id"):
            if metadata.get(key):
                result.add(str(metadata[key]))
        for key in ("source_event_ids", "mematlas_event_ids", "event_ids"):
            values = metadata.get(key) or []
            if isinstance(values, str):
                values = [values]
            result.update(str(value) for value in values)
    return result


class LoCoMoBenchmark(BenchmarkAdapter):
    benchmark_id = "locomo"

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "text",
        max_conversations: int | None = None,
        max_questions_per_conversation: int | None = None,
        top_k: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if mode not in {"text", "image_text"}:
            raise ValueError("LoCoMo mode must be 'text' or 'image_text'")
        self.mode = mode
        self.max_conversations = max_conversations
        self.max_questions_per_conversation = max_questions_per_conversation
        self.top_k = top_k
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("LoCoMo annotation file must contain a list")
        self._data = payload[:max_conversations] if max_conversations is not None else payload

    def _history(self, sample: Mapping[str, Any], scope: Scope) -> tuple[MemoryEvent, ...]:
        conversation = sample["conversation"]
        events: list[MemoryEvent] = []
        speaker_a = str(conversation.get("speaker_a", ""))
        session_numbers = sorted(
            int(match.group(1))
            for key in conversation
            if (match := re.fullmatch(r"session_(\d+)", key))
        )
        for number in session_numbers:
            turns = conversation.get(f"session_{number}") or []
            base_time = _parse_session_time(str(conversation.get(f"session_{number}_date_time", "")))
            for turn_index, turn in enumerate(turns):
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text", ""))
                if speaker:
                    text = f"{speaker}: {text}"
                content = [{"type": "text", "text": text}] if text else []
                image_file = turn.get("img_file")
                caption = turn.get("blip_caption")
                if image_file and self.mode == "image_text":
                    image_path = Path(str(image_file))
                    if not image_path.is_absolute():
                        image_path = self.path.parent / image_path
                    content.append({"type": "image", "uri": str(image_path), "caption": caption or ""})
                elif image_file and caption:
                    content.append({"type": "text", "text": f"[Shared image: {caption}]"})
                dia_id = str(turn.get("dia_id") or f"D{number}:{turn_index + 1}")
                events.append(MemoryEvent(
                    kind=EventKind.MESSAGE,
                    scope=scope,
                    content=content,
                    event_id=dia_id,
                    timestamp=base_time + timedelta(microseconds=turn_index),
                    role="user" if speaker == speaker_a else "assistant",
                    extensions={
                        "speaker": speaker,
                        "benchmark": self.benchmark_id,
                        "session_id": f"D{number}",
                        "dia_id": dia_id,
                        "session_date_time": conversation.get(f"session_{number}_date_time"),
                    },
                ))
        return tuple(events)

    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]:
        if split not in {"test", "all"}:
            return
        for sample_index, sample in enumerate(self._data):
            sample_id = str(sample.get("sample_id", sample_index))
            scope = {"user_id": f"locomo:{sample_id}", "agent_id": "locomo-reader"}
            history = self._history(sample, scope)
            questions = sample.get("qa", [])
            if self.max_questions_per_conversation is not None:
                questions = questions[:self.max_questions_per_conversation]
            for question_index, question in enumerate(questions):
                category = int(question["category"])
                yield BenchmarkCase(
                    case_id=f"{sample_id}:{question_index}",
                    scope=scope,
                    history=history,
                    query=str(question["question"]),
                    reference=question.get("answer"),
                    metadata={
                        "sample_id": sample_id,
                        "question_index": question_index,
                        "category": category,
                        "evidence": list(question.get("evidence", [])),
                        "top_k": self.top_k,
                        "split": "test",
                    },
                )

    def native_score(self, predictions: list[Prediction]) -> Mapping[str, Any]:
        by_category: dict[int, list[float]] = defaultdict(list)
        retrieval_recalls: list[float] = []
        scores = []
        for prediction in predictions:
            metadata = prediction.metadata.get("case_metadata", {})
            category = int(metadata["category"])
            value = _locomo_score(prediction.output, prediction.reference, category)
            scores.append(value)
            by_category[category].append(value)
            evidence = {
                str(item).replace("(", "").replace(")", "")
                for item in metadata.get("evidence", []) if item
            }
            if evidence:
                retrieved = _retrieved_source_ids(prediction)
                retrieval_recalls.append(len(evidence & retrieved) / len(evidence))
        result: dict[str, Any] = {
            "qa_f1": mean(scores) if scores else 0.0,
            "question_count": len(scores),
            "retrieval_recall": mean(retrieval_recalls) if retrieval_recalls else None,
            "retrieval_question_count": len(retrieval_recalls),
        }
        result.update({f"qa_f1.category_{key}": mean(values) for key, values in sorted(by_category.items())})
        return result

    def describe(self) -> Mapping[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "path": str(self.path),
            "sha256": self.file_sha256(self.path),
            "mode": self.mode,
            "conversation_count": len(self._data),
            "max_questions_per_conversation": self.max_questions_per_conversation,
            "official_metric": "LoCoMo category-aware token F1",
        }
