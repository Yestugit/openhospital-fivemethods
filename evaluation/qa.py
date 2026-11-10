"""Complete default QA, also the base for methods/<method>/qa.py adapters.

The experiment calls ``await qa(case, memory)``. This file contains the whole
single-question path: retrieval, prompt construction, model call and trajectory.
"""
from __future__ import annotations

from importlib import import_module

from mematlas.core.contributions import contribution_text
from mematlas.core.models import MemoryEpisode, MemoryEvent, content_text
from mematlas.providers.instrumentation import MeteredService
from mematlas.core.telemetry import active_usage


class DefaultQA:
    """Execute ``(case, memory) -> MemoryEpisode`` without reading reference answers.

    Override only necessary steps. prepare_episode() also receives imported
    histories and training trajectories produced by a user Agent; the experiment
    remains responsible for submitting them to Memory.add().
    """
    requires_answer_model = True
    # Multi-step QA may retain its complete episode for native scoring. Ordinary
    # QA keeps the existing compact Prediction format and v3 recovery contract.
    include_episode_in_prediction = False

    @classmethod
    def requires_answer_model_for(cls, config, benchmark=None) -> bool:
        """Allow method QA adapters to opt out for native non-QA benchmarks."""
        return bool(cls.requires_answer_model)

    def __init__(self, config, services):
        self.config, self.services = config, services
        provider = config.answer_model if config is not None else None
        self.llm = services.object("answer_llm")
        self._owned_llm = None
        if self.llm is None and provider is not None:
            from mematlas.providers import OpenAICompatibleLLM
            self.llm = OpenAICompatibleLLM(model=provider.model, base_url=provider.base_url,
                api_key_env=provider.api_key_env, default_options=dict(provider.options))
            self._owned_llm = self.llm
        self.model = provider.model if provider else getattr(self.llm, "model", None)
        self.temperature = provider.options.get("temperature", 0.0) if provider else 0.0

    async def __call__(self, case, memory) -> MemoryEpisode:
        contribution = await self.retrieve(case, memory)
        messages = self.messages(case, contribution)
        if self.llm is None:
            raise ValueError("Default QA requires experiment.answer_model or answer_llm")
        active = active_usage()
        provider = (MeteredService(self.llm, active[0], "llm")
                    if active is not None and not isinstance(self.llm, MeteredService) else self.llm)
        output = await provider.complete(messages, model=self.model, temperature=self.temperature)
        return MemoryEpisode(task=content_text(case.query), scope=case.scope, output=output,
            events=[MemoryEvent("message", case.query, role="user"),
                    MemoryEvent("message", output, role="assistant")], applied_memory=[contribution])

    async def retrieve(self, case, memory):
        return await memory.search(case.query, scope=case.scope,
            top_k=case.metadata.get("top_k"), extensions=self.retrieval_extensions(case.metadata))

    @staticmethod
    def retrieval_extensions(metadata):
        """Prepare method input for QA and other scenes using the same method."""
        return dict(metadata.get("memory_extensions", {}))

    def messages(self, case, contribution):
        question_date = case.metadata.get("question_date")
        temporal = (f"Question date: {question_date}\n"
                    "Interpret relative dates in the question using this date.\n\n") if question_date else ""
        prompt = (
            "Answer the question using the retrieved memory. Give only the answer needed by the question. "
            'If the memory does not contain the answer, reply exactly "No information available."\n\n'
            f"{temporal}Retrieved memory:\n{contribution_text(contribution)}\n\nQuestion: {content_text(case.query)}"
        )
        return [{"role": "system", "content": "You are a precise long-term-memory QA reader."},
                {"role": "user", "content": prompt}]

    @staticmethod
    def prepare_episode(case, episode, *, last_for_scope):
        """Pure data preparation, shared by preflight and learning submission."""
        return episode

    @staticmethod
    def collect_training_episode(case):
        """Opt in when this QA must execute a task to obtain learning experience.

        Default text QA consumes supplied histories/episodes as before. Multi-step
        adapters may opt in for raw tasks; the runner still owns learn/save.
        """
        return False

    @classmethod
    def validate_cases(cls, config, train, tests):
        """Check this QA's inputs before allocation; custom Agents own theirs."""

    async def close(self):
        if self._owned_llm is not None and hasattr(self._owned_llm, "_client"):
            await self._owned_llm._client.close()


def _qa_type(plugin):
    declaration = plugin.qa_factory
    if declaration is None:
        return DefaultQA
    if not isinstance(declaration, str) or ":" not in declaration:
        raise ValueError("MethodPlugin.qa_factory must be a module:class import path")
    module, name = declaration.split(":", 1)
    factory = getattr(import_module(module), name)
    if not isinstance(factory, type) or not issubclass(factory, DefaultQA):
        raise TypeError("Method QA must subclass DefaultQA")
    return factory
