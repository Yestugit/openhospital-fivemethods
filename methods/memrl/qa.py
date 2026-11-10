"""MemRL's finite-training boundary; answering reuses the complete DefaultQA."""
from dataclasses import replace

from mematlas.evaluation.qa import DefaultQA


class MemRLQA(DefaultQA):
    @staticmethod
    def retrieval_extensions(metadata):
        extensions = DefaultQA.retrieval_extensions(metadata)
        if metadata.get("task_type"):
            extensions["memrl"] = {**extensions.get("memrl", {}), "task_type": metadata["task_type"]}
        return extensions

    @staticmethod
    def prepare_episode(case, episode, *, last_for_scope):
        metadata = getattr(case, "metadata", {})
        if not last_for_scope and not metadata.get("task_type"):
            return episode
        options = dict(episode.extensions.get("memrl", {}))
        if metadata.get("task_type"):
            options.setdefault("task_type", metadata["task_type"])
            options.setdefault("task_id", episode.episode_id)
        if last_for_scope:
            options["end_of_training"] = True
        return replace(episode, extensions={**episode.extensions,
            "memrl": options})
