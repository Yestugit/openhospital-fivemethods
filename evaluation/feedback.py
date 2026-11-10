"""Feedback delivery timing belongs to evaluation, alongside pending episodes."""
from dataclasses import dataclass
import inspect

from mematlas.core.episodes import EpisodeResult
from mematlas.core.models import MemoryEpisode
from mematlas.core.telemetry import usage_context
from mematlas.evaluation.validation import submit_episode


@dataclass(frozen=True)
class FeedbackSchedule:
    timing: str = "immediate"
    delay: int = 0

    def __post_init__(self):
        if self.timing not in {"immediate", "batch", "delayed"} or type(self.delay) is not int or self.delay < 0:
            raise ValueError("Invalid feedback schedule")
        if self.timing == "delayed" and self.delay < 1:
            raise ValueError("Delayed feedback requires a positive delay")

    async def accept(self, experience, episode: MemoryEpisode, resolve_feedback):
        memory, history = experience.core_memory, experience.history
        async with memory._lock:
            memory._writable()
            episode = memory._complete_episode(episode, None)
            identity = memory._history.identity(episode.episode_id, episode.scope)
            previous = history.results.get(identity)
            if previous and previous.learning == "submitted":
                return
            if identity in history.pending_feedback:
                if memory._history.episode(identity) != episode:
                    raise ValueError("Pending feedback episode cannot be replaced")
            else:
                await memory._setup()
                memory._history.record_episode(episode)
                memory._history.persist()
                history.exposures += 1
                history.pending_feedback[identity] = {"due": history.exposures + self.delay}
                history.record_result(EpisodeResult(identity, case_id=episode.episode_id,
                    metadata={"scope": episode.scope}))
        if self.timing in {"immediate", "delayed"}:
            await self.deliver_due(experience, resolve_feedback)

    async def deliver_due(self, experience, resolve_feedback, *, batch_end=False):
        memory, history = experience.core_memory, experience.history
        memory._writable()
        for identity, row in list(history.pending_feedback.items()):
            due = self.timing == "immediate" or self.timing == "batch" and batch_end or self.timing == "delayed" and row["due"] <= history.exposures
            if not due:
                continue
            episode = memory._history.episode(identity)
            with usage_context(case_id=episode.episode_id):
                completed = resolve_feedback(episode)
                if inspect.isawaitable(completed):
                    completed = await completed
                if not isinstance(completed, MemoryEpisode) or completed.episode_id != episode.episode_id or completed.scope != episode.scope:
                    raise ValueError("Feedback must preserve the original episode identity and scope")
                try:
                    result = await submit_episode(memory, completed)
                except BaseException as exc:
                    history.record_result(EpisodeResult(identity, case_id=episode.episode_id,
                        feedback="valid", learning="unknown" if memory.state.value == "failed" else "failed",
                        metadata={"scope": episode.scope}, error={"type": type(exc).__name__, "message": str(exc)}))
                    raise
            history.record_result(result)
            history.pending_feedback.pop(identity, None)
