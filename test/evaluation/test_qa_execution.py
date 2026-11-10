"""Complete QA, plugin selection and shared learning/testing execution."""
from dataclasses import replace

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, Memory, MemoryContribution, MemoryEpisode, MemoryEvent
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.plugin import MethodServices
from mematlas.core.registry import default_registry
from mematlas.evaluation.qa import DefaultQA, _qa_type
from mematlas.methods.memrl.qa import MemRLQA
from mematlas.providers import CallableLLM
from mematlas.test.evaluation.support import FixtureMethod, FixturePlugin, config_for, registry_for


class PreparedQA(DefaultQA):
    @staticmethod
    def prepare_episode(case, episode, *, last_for_scope):
        data = episode.extensions.get("fixture", {})
        return replace(episode, extensions={**episode.extensions, "fixture": {
            **data, "prepared": data.get("prepared", 0) + 1, "last": last_for_scope}})

    def messages(self, case, contribution):
        messages = super().messages(case, contribution)
        messages[0]["content"] += " Method-specific reader."
        return messages


class RequiredInputQA(PreparedQA):
    @classmethod
    def validate_cases(cls, config, train, tests):
        if any("required_context" not in case.metadata for case in tests):
            raise ValueError("Method QA requires case.metadata.required_context")


class NoModelQA(DefaultQA):
    requires_answer_model = False

    async def __call__(self, case, memory):
        used = await self.retrieve(case, memory)
        return MemoryEpisode(output="answer", applied_memory=[used])


def plugin_for(qa=PreparedQA):
    plugin = FixturePlugin()
    plugin.qa_factory = f"{qa.__module__}:{qa.__qualname__}"
    plugin.descriptor = replace(plugin.descriptor, input_schemas={"add": {
        "type": "object", "required": ["fixture"], "properties": {"fixture": {
            "type": "object", "required": ["prepared"], "properties": {
                "prepared": {"type": "integer", "minimum": 1}}}}}})
    return plugin


def dataset():
    scope = {"user_id": "alice"}
    return InMemoryBenchmark("method-qa", [
        *[BenchmarkCase(f"train-{i}", f"task-{i}", scope=scope,
            episode=MemoryEpisode(f"episode-{i}", events=[MemoryEvent("message", f"fact-{i}")],
                output="stored training output", extensions={"fixture": {"retained": True}}),
            metadata={"split": "train"}) for i in range(2)],
        BenchmarkCase("test", "question", scope=scope, reference="answer",
            history=(MemoryEvent("message", "historical fact"),)),
    ])


def experiment(tmp_path, protocol="streaming_memory"):
    return ExperimentConfig.from_dict({"methods": [config_for(tmp_path / "memory")],
        "benchmark": dataset(), "protocol": protocol,
        "checkpoints": (0, 1, 2) if protocol == "experience_learning_curve" else (),
        "output_dir": str(tmp_path / "results")})


@pytest.mark.asyncio
async def test_complete_default_qa_returns_actual_trajectory_and_sources(tmp_path):
    class ContextMethod(FixtureMethod):
        async def search(self, query, *, level_ref=None, top_k, scope, extensions):
            self.request = query, top_k, scope, extensions
            return MemoryContribution(items=[{"content": "unused candidate"}, {"content": "answer"}],
                context=[{"content": "formatted answer", "source_item_indices": [1]}])

    class Plugin(FixturePlugin):
        def create_method(self, config, services):
            method = ContextMethod()
            self.instances.append(method)
            return method

    prompts, plugin = [], Plugin()
    def answer(messages, **options):
        prompts.append(messages)
        return "answer"

    case = BenchmarkCase("question", "query", scope={"user_id": "alice"}, reference="secret reference",
        metadata={"top_k": 2, "memory_extensions": {"fixture": {"hint": "input"}}})
    qa = DefaultQA(None, MethodServices(objects={"answer_llm": CallableLLM(answer)}))
    async with Memory(config_for(tmp_path), registry=registry_for(plugin)) as memory:
        async with memory.read_only():
            episode = await qa(case, memory)
        assert plugin.instances[0].request[:3] == ("query", 2, case.scope)
        assert plugin.instances[0].request[3]["fixture"] == {"hint": "input"}
        assert not plugin.instances[0].episodes
    assert episode.scope == case.scope and episode.task == "query" and episode.output == "answer"
    assert [(event.role, event.content) for event in episode.events] == [("user", "query"), ("assistant", "answer")]
    assert episode.applied_memory[0].context[0]["source_item_indices"] == [1]
    assert "formatted answer" in prompts[0][-1]["content"]
    assert "unused candidate" not in prompts[0][-1]["content"]
    assert "secret reference" not in str(prompts)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["streaming_memory", "experience_train_test", "experience_learning_curve"])
@pytest.mark.parametrize("external_agent", [False, True])
async def test_method_qa_prepares_history_stored_training_and_agent_episodes(tmp_path, protocol, external_agent):
    plugin, calls, prompts = plugin_for(), [], []
    runner = ExperimentRunner(registry=registry_for(plugin))
    config = experiment(tmp_path, protocol)
    def answer(messages, **options):
        prompts.append(messages)
        return "answer"
    answer.config = {}
    async def agent(case, memory):
        calls.append(case.case_id)
        used = await memory.search(case.query, scope=case.scope)
        return MemoryEpisode(output="answer", applied_memory=[used],
            events=[MemoryEvent("message", "executed action")], extensions={"fixture": {"retained": True}})
    agent.config = {}
    services = MethodServices(objects={"answer_llm": CallableLLM(answer)})
    assert runner.check(config, agent=agent if external_agent else None, services=services)["ready"]
    assert not plugin.instances
    result = (await runner.evaluate(config, agent=agent if external_agent else None, services=services))[0]
    assert not result.failures, result.failures
    assert result.metrics["common.exact_match"] == 1
    assert result.predictions[0].applied_memory
    learned = plugin.instances[0].episodes
    assert [e["extensions"]["fixture"]["prepared"] for e in learned] == [1] * len(learned)
    assert [e["extensions"]["fixture"]["last"] for e in learned] == (
        [True] if protocol == "streaming_memory" else [False, True])
    if protocol != "streaming_memory":
        assert [e["output"] for e in learned] == ["answer" if external_agent else "stored training output"] * 2
        assert all(e["extensions"]["fixture"]["retained"] for e in learned)
    if external_agent:
        assert not prompts
    else:
        assert not calls
        assert len(prompts) == (3 if protocol == "experience_learning_curve" else 1)
        assert all("Method-specific reader." in p[0]["content"] for p in prompts)
    assert runner.results(tmp_path / "results")[0].to_dict() == result.to_dict()


def test_method_qa_input_checks_run_before_memory_allocation(tmp_path):
    plugin = plugin_for(RequiredInputQA)
    runner = ExperimentRunner(registry=registry_for(plugin))
    services = MethodServices(objects={"answer_llm": CallableLLM(lambda *a, **k: "answer")})
    report = runner.check(experiment(tmp_path), services=services)
    assert not report["ready"] and "required_context" in str(report)
    assert not plugin.instances
    assert not (tmp_path / "memory").exists()


@pytest.mark.asyncio
async def test_external_agent_uses_own_inputs_and_keeps_method_learning_preparation(tmp_path):
    plugin = plugin_for(RequiredInputQA)
    runner = ExperimentRunner(registry=registry_for(plugin))
    async def agent(case, memory):
        return MemoryEpisode(output="answer")
    result = (await runner.evaluate(experiment(tmp_path), agent=agent))[0]
    assert not result.failures, result.failures
    assert plugin.instances[0].episodes[0]["extensions"]["fixture"]["prepared"] == 1


@pytest.mark.asyncio
async def test_method_qa_can_run_without_answer_model_when_not_needed(tmp_path):
    plugin = FixturePlugin()
    plugin.qa_factory = f"{__name__}:NoModelQA"
    runner = ExperimentRunner(registry=registry_for(plugin))
    config = experiment(tmp_path)
    assert runner.check(config)["ready"]
    result = (await runner.evaluate(config))[0]
    assert not result.failures and result.metrics["common.exact_match"] == 1


def test_builtin_lazy_registry_resolves_method_qa_declaration():
    registry = default_registry()
    assert _qa_type(registry.get("memrl")) is MemRLQA
    assert _qa_type(registry.get("mem0")) is DefaultQA
    assert _qa_type(registry.get("memoryos")) is DefaultQA


@pytest.mark.asyncio
async def test_method_qa_change_rejects_cached_frozen_result(tmp_path, monkeypatch):
    plugin = plugin_for()
    runner = ExperimentRunner(registry=registry_for(plugin))
    config = experiment(tmp_path, "experience_train_test")
    services = MethodServices(objects={"answer_llm": CallableLLM(lambda *a, **k: "answer")})
    first = (await runner.evaluate(config, services=services))[0]
    assert not first.failures and first.run_metadata["recovery"]["supported"]
    resumed = (await runner.evaluate(replace(config, output_dir=str(tmp_path / "resumed")),
        services=services, resume_from=tmp_path / "results"))[0]
    assert not resumed.failures

    def changed_messages(self, case, contribution):
        return [{"role": "user", "content": "Changed reader"}]
    monkeypatch.setattr(PreparedQA, "messages", changed_messages)
    changed = (await runner.evaluate(replace(config, output_dir=str(tmp_path / "changed")),
        services=services, resume_from=tmp_path / "results"))[0]
    assert changed.failures and "contract" in str(changed.failures)
    assert not changed.predictions


@pytest.mark.asyncio
async def test_cached_training_episode_is_not_prepared_twice_on_retry(tmp_path):
    class InterruptedMethod(FixtureMethod):
        async def _learn(self, episode, **options):
            if fail["enabled"]:
                raise RuntimeError("interrupted learning")
            await super()._learn(episode, **options)

    class Plugin(FixturePlugin):
        qa_factory = f"{__name__}:PreparedQA"

        def create_method(self, config, services):
            method = InterruptedMethod()
            self.instances.append(method)
            return method

    fail, plugin = {"enabled": True}, Plugin()
    runner = ExperimentRunner(registry=registry_for(plugin))
    config = experiment(tmp_path, "experience_train_test")
    services = MethodServices(objects={"answer_llm": CallableLLM(lambda *a, **k: "answer")})
    first = (await runner.evaluate(config, services=services))[0]
    assert first.failures
    fail["enabled"] = False
    resumed = (await runner.evaluate(replace(config, output_dir=str(tmp_path / "resumed")),
        services=services, resume_from=tmp_path / "results"))[0]
    assert not resumed.failures, resumed.failures
    assert [e["extensions"]["fixture"]["prepared"] for e in plugin.instances[-1].episodes] == [1, 1]


@pytest.mark.parametrize("declaration", ["invalid", "mematlas.core.models:MemoryEpisode"])
def test_invalid_method_qa_declaration_is_rejected_by_preflight(tmp_path, declaration):
    plugin = FixturePlugin()
    plugin.qa_factory = declaration
    report = ExperimentRunner(registry=registry_for(plugin)).check(experiment(tmp_path))
    assert not report["ready"] and "qa" in str(report).lower()
    assert not plugin.instances
