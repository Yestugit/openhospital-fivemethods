from __future__ import annotations

from datetime import date
import math
import random
from typing import Any, Mapping

from mematlas.core.plugin import MethodServices
from mematlas.methods.core_utils import lexical_score


class MemoryBankCore:
    """Forgetting-aware retrieval plus episodic summary and personality memory."""

    def __init__(self, services: MethodServices, *, seed: int | None = None):
        self.services = services
        self.random = random.Random(seed)

    @staticmethod
    def _days_between(left: str, right: str) -> int:
        try:
            return max(0, (date.fromisoformat(right) - date.fromisoformat(left)).days)
        except ValueError:
            return 0

    @staticmethod
    def retention(elapsed: int, strength: float, formula: str = "source") -> float:
        """Preserve cf61c419's operator precedence; offer the stated curve explicitly."""
        if formula == "source":
            return math.exp(-max(0, elapsed) / 5 * max(1, strength))
        if formula == "strength_denominator":
            return math.exp(-max(0, elapsed) / (5 * max(1, strength)))
        raise ValueError("Unknown MemoryBank forgetting formula")

    def forget(self, user: dict[str, Any], current_date: str, formula: str = "source") -> None:
        for day, records in list(user.get("history", {}).items()):
            retained = []
            for record in records:
                elapsed = self._days_between(str(record.get("last_recall_date", day)), current_date)
                probability = self.retention(elapsed, float(record.get("memory_strength", 1)), formula)
                if self.random.random() <= probability:
                    retained.append(record)
            if retained:
                user["history"][day] = retained
            else:
                del user["history"][day]
                user.get("summary", {}).pop(day, None)

    def search(
        self, memory: Mapping[str, Any], *, user_id: str, query: str,
        top_k: int, current_date: str, enable_forgetting: bool,
        semantic_scores: Mapping[str, float] | None = None, read_only: bool = False,
    ) -> list[dict[str, Any]]:
        # Forgetting is a loader/index-initialization operation, not a score multiplier.
        user = memory.get(user_id, {})
        ranked = []
        for day, records in user.get("history", {}).items():
            for record in records:
                text = f"User: {record.get('query', '')}\nAssistant: {record.get('response', '')}"
                score = (semantic_scores.get(str(record.get("memory_id")), float("-inf"))
                         if semantic_scores is not None else lexical_score(query, text))
                ranked.append((score, day, record, text))
        for day, summary in user.get("summary", {}).items():
            if isinstance(summary, Mapping) and summary.get("content"):
                text = f"The summary of the conversation on {day} is: {summary['content']}"
                score = (semantic_scores.get(str(summary.get("memory_id")), float("-inf"))
                         if semantic_scores is not None else lexical_score(query, text))
                ranked.append((score, day, summary, text))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:top_k]
        # Upstream returns recalled documents in source/date order.
        selected.sort(key=lambda item: str(item[2].get("memory_id", "")))
        results = []
        for score, day, record, text in selected:
            if not read_only and enable_forgetting and "query" in record:
                record["memory_strength"] = int(record.get("memory_strength", 1)) + 1
                record["last_recall_date"] = current_date
            results.append({"text": text, "score": score,
                            "memory_id": record.get("memory_id"), "date": day})
        return results

    async def summarize(self, memory: dict[str, Any], user_id: str, language: str = "en") -> None:
        user = memory.get(user_id)
        if not user:
            return
        all_records = [
            (day, record)
            for day, records in user.get("history", {}).items()
            for record in records
        ]
        if self.services.llm is None:
            summary = " ".join(str(record.get("query", "")) for _, record in all_records[-20:])
            user["overall_history"] = summary
            user["overall_personality"] = ""
            return
        async def generate(prompt: str) -> str:
            result = await self.services.llm.complete([
                {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                {"role": "system", "content": "Sure, I will do my best to assist you."},
                {"role": "user", "content": f"Language: {language}. {prompt}"},
            ], temperature=0.7, max_tokens=400, top_p=1.0,
               frequency_penalty=0.4, presence_penalty=0.2)
            return str(result)

        for day, records in user.get("history", {}).items():
            dialogue = "\n".join(f"{user_id}: {r.get('query', '')}\nAI: {r.get('response', '')}" for r in records)
            if not user.setdefault("summary", {}).get(day):
                summary = await generate("Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n" + dialogue)
                user["summary"][day] = {"content": summary}
            if not user.setdefault("personality", {}).get(day):
                user["personality"][day] = await generate(f"Based on the following dialogue, please summarize {user_id}'s personality traits and emotions, and devise response strategies based on your speculation. Dialogue content:\n" + dialogue)
        history = "\n".join(f"At {day}, the events are {value.get('content', '') if isinstance(value, Mapping) else value}" for day, value in user["summary"].items())
        traits = "\n".join(f"At {day}, the analysis shows {value}" for day, value in user["personality"].items())
        user["overall_history"] = await generate("Please provide a highly concise summary of the following event, capturing the essential key information as succinctly as possible. Summarize the event:\n" + history)
        user["overall_personality"] = await generate("The following are the user's exhibited personality traits and emotions throughout multiple dialogues, along with appropriate response strategies for the current situation:\n" + traits + "\nPlease provide a highly concise and general summary of the user's personality and the most appropriate response strategy for the AI lover.")
