"""Small trajectory benchmark for the external-feedback ReasoningBank variant.

The upstream method is evaluated from agent trajectories, an external success
label, and task-similarity retrieval.  Conversation QA datasets do not carry
those inputs, so this adapter provides a deterministic, three-train/three-test
fixture with WebArena/SWE-Bench-shaped task and action records.  It is useful
for integration checks and does not claim to replace the upstream benchmarks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.core.evaluation import BenchmarkCase, Prediction
from mematlas.core.models import MemoryEpisode, MemoryEvent


def _event(episode_id: str, event_id: str, kind: str, text: str, *, role: str | None = None,
           scope: Mapping[str, str]) -> MemoryEvent:
    return MemoryEvent(kind, text, role=role, event_id=event_id, episode_id=episode_id,
                       timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), scope=dict(scope))


def _episode(case_id: str, task: str, *, success: bool, scope: Mapping[str, str],
             steps: tuple[tuple[str, str], ...], feedback: str) -> MemoryEpisode:
    events = [_event(case_id, f"{case_id}:{index}", kind, text,
                     role="user" if kind == "message" else "assistant", scope=scope)
              for index, (kind, text) in enumerate(steps)]
    return MemoryEpisode(episode_id=case_id, task=task, events=events, success=success,
                         feedback=feedback, scope=dict(scope))


class ReasoningBankTrajectoryBenchmark(BenchmarkAdapter):
    """A compact, reproducible trajectory benchmark.

    Test cases have no answer-model reference.  Native scoring checks whether
    retrieval returned the expected episode group, preserved its success label,
    and retained source event provenance.  This makes the score meaningful for
    the method's native contract instead of treating generated prose as QA.
    """

    benchmark_id = "reasoningbank_trajectory"

    def __init__(self, *, scope: Mapping[str, str] | None = None,
                 include_failure: bool = True) -> None:
        self.scope = dict(scope or {"user_id": "reasoningbank-trajectory"})
        self.include_failure = bool(include_failure)
        self._cases = self._build_cases()

    def _build_cases(self) -> tuple[BenchmarkCase, ...]:
        login = _episode(
            "train-login-success",
            "Sign in to the admin console and verify the account status",
            success=True,
            scope=self.scope,
            steps=(
                ("message", "Open the admin console for the requested account."),
                ("action", "Fill the username and one-time code fields."),
                ("observation", "The account dashboard is visible."),
                ("action", "Check the account status panel."),
                ("tool_result", "The status is active."),
            ),
            feedback="The dashboard showed an active account after verification.",
        )
        search = _episode(
            "train-search-failure",
            "Find the user's saved report in the dashboard",
            success=False,
            scope=self.scope,
            steps=(
                ("message", "Locate the saved quarterly report."),
                ("action", "Search the reports page for the quarterly report."),
                ("observation", "The reports page returns no matching item."),
                ("action", "Try the archive filter."),
                ("tool_result", "The report is still unavailable."),
            ),
            feedback="The report could not be located in the dashboard or archive.",
        )
        checkout = _episode(
            "train-checkout-success",
            "Complete checkout and confirm the order number",
            success=True,
            scope=self.scope,
            steps=(
                ("message", "Submit the prepared cart."),
                ("action", "Select the saved shipping address and confirm payment."),
                ("observation", "The payment confirmation page is displayed."),
                ("tool_result", "Order number 8421 was created."),
            ),
            feedback="Checkout completed and the order confirmation was displayed.",
        )
        training = [
            BenchmarkCase(case_id=episode.episode_id, query=episode.task or "", scope=self.scope,
                          episode=episode, metadata={"split": "train", "trajectory_source": "synthetic"})
            for episode in (login, search, checkout)
        ]
        tests = [
            BenchmarkCase(
                case_id="test-login-retrieval",
                query="Verify whether the admin account is active after signing in",
                scope=self.scope,
                reference=None,
                metadata={"split": "test", "expected_source_episode_ids": [login.episode_id],
                          "expected_success": True, "top_k": 1},
            ),
            BenchmarkCase(
                case_id="test-checkout-retrieval",
                query="Confirm the order created after payment during checkout",
                scope=self.scope,
                reference=None,
                metadata={"split": "test", "expected_source_episode_ids": [checkout.episode_id],
                          "expected_success": True, "top_k": 1},
            ),
        ]
        if self.include_failure:
            tests.append(BenchmarkCase(
                case_id="test-search-failure-retrieval",
                query="What should be done when the saved dashboard report cannot be found?",
                scope=self.scope,
                reference=None,
                metadata={"split": "test", "expected_source_episode_ids": [search.episode_id],
                          "expected_success": False, "top_k": 2},
            ))
        return tuple(training + tests)

    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]:
        return (case for case in self._cases
                if split == "all" or case.metadata.get("split", "test") == split)

    @staticmethod
    def _items(prediction: Prediction) -> list[Mapping[str, Any]]:
        items: list[Mapping[str, Any]] = []
        for contribution in prediction.applied_memory:
            items.extend(contribution.items)
        return items

    def native_score(self, predictions: list[Prediction]) -> Mapping[str, Any]:
        total = len(predictions)
        if not total:
            return {"reasoningbank.trajectory_recall": 0.0,
                    "reasoningbank.success_label_accuracy": 0.0,
                    "reasoningbank.source_provenance": 0.0,
                    "reasoningbank.evaluated_cases": 0}
        recalled = labels = provenance = 0
        for prediction in predictions:
            metadata = prediction.metadata.get("case_metadata", {})
            expected = {str(value) for value in metadata.get("expected_source_episode_ids", ())}
            items = self._items(prediction)
            source_ids = {str(item.get("metadata", {}).get("source_episode_id")) for item in items}
            hit = bool(expected & source_ids)
            recalled += int(hit)
            expected_success = metadata.get("expected_success")
            labels += int(hit and any(item.get("metadata", {}).get("source_success") == expected_success
                                      for item in items))
            provenance += int(hit and any(item.get("source_event_ids") for item in items))
        return {
            "reasoningbank.trajectory_recall": recalled / total,
            "reasoningbank.success_label_accuracy": labels / total,
            "reasoningbank.source_provenance": provenance / total,
            "reasoningbank.evaluated_cases": total,
        }

    def describe(self) -> Mapping[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "kind": "native_trajectory",
            "train_cases": sum(case.metadata.get("split") == "train" for case in self._cases),
            "test_cases": sum(case.metadata.get("split") == "test" for case in self._cases),
            "source": "synthetic WebArena/SWE-Bench-shaped trajectories",
            "scoring": "episode-group retrieval, external success label, and event provenance",
            "upstream_equivalents": ["WebArena", "SWE-Bench"],
        }


__all__ = ["ReasoningBankTrajectoryBenchmark"]
