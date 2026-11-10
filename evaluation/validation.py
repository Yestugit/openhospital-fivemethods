"""Evaluation preflight and normalization of imported learning units."""
from dataclasses import replace
import importlib.util
import os

from mematlas.core.capabilities import Operation
from mematlas.core.contracts import ProtocolRequirements, validate_episode, validate_scope
from mematlas.core.models import MemoryEpisode, MemoryUnit, content_modalities, content_text
from mematlas.core.provenance import json_sha256
from mematlas.core.scope import normalize_scope, scope_key
from mematlas.evaluation.qa import _qa_type


def memory_services(services):
    names = {"answer_llm", "native_judge"}
    return replace(services, objects={k: v for k, v in services.objects.items() if k not in names},
                   factory_specs={k: v for k, v in services.factory_specs.items() if k not in names})


async def submit_episode(memory, episode, *, scope=None):
    """Adapt one executed benchmark task to the public submission contract."""
    owner = scope if scope is not None else episode.scope
    if episode.events is None:
        episode = memory._complete_episode(episode, owner)
    validate_episode(memory.capabilities, episode)
    receipt = await memory.add(MemoryUnit.from_episode(episode), scope=owner)
    await memory.episode_end(receipt.episode_ref, scope=owner)
    result = memory._history.results[memory._history.identity(receipt.episode_ref,
        memory._scope(owner))]
    if result.learning != "submitted":
        raise ValueError("Training task ended without the feedback required by this method")
    return result


def history_episode(case):
    if case.episode is not None:
        return case.episode
    events = list(case.history)
    identities = {event.episode_id for event in events if event.episode_id is not None}
    if len(identities) > 1:
        raise ValueError("A history learning unit must contain one episode; supply case.episode explicitly")
    identity = next(iter(identities), "history:" + json_sha256({
        "scope": case.scope, "events": [event.event_id for event in events]}))
    return MemoryEpisode(identity, events=events, scope=case.scope)


def training_episode(case):
    episode = case.episode or MemoryEpisode("training:" + json_sha256({"case": case.case_id, "scope": case.scope}))
    return replace(episode, task=episode.task or content_text(case.query),
                   scope=case.scope if episode.scope is None else episode.scope,
                   events=list(case.history) if episode.events is None else episode.events)


def validate_learning(plugin, episode, scope):
    if not isinstance(episode, MemoryEpisode):
        raise TypeError("QA.prepare_episode must return MemoryEpisode")
    if episode.scope is not None and episode.scope != scope:
        raise ValueError("Episode scope differs from its benchmark case")
    for event in episode.events or ():
        if event.scope is not None and event.scope != scope:
            raise ValueError("Event scope differs from its benchmark case")
        if event.episode_id is not None and event.episode_id != episode.episode_id:
            raise ValueError("Event belongs to a different episode")
    validate_episode(plugin.descriptor.capabilities, episode)
    errors = plugin.validate_input("add", episode.extensions)
    if errors:
        raise ValueError("; ".join(errors))


def validate_cases(config, plugin, train, tests, agent):
    if len({c.case_id for c in tests}) != len(tests) or len({c.case_id for c in train}) != len(train):
        raise ValueError("Case IDs must be unique within each split")
    capabilities = plugin.descriptor.capabilities
    for case in (*train, *tests):
        validate_scope(capabilities, normalize_scope(case.scope))
    for case in tests:
        errors = plugin.validate_input("search", case.metadata.get("memory_extensions", {}))
        if errors:
            raise ValueError("; ".join(errors))
    operations = {Operation.SEARCH}
    learning = config.protocol != "streaming_memory"
    if learning:
        operations.update({Operation.ADD, Operation.EPISODE_END, Operation.SAVE, Operation.LOAD})
    elif any(case.episode is not None or case.history for case in tests):
        operations.update({Operation.ADD, Operation.EPISODE_END})
    ProtocolRequirements(operations=frozenset(operations), frozen=learning,
        scope_count=len({scope_key(c.scope) for c in (*train, *tests)}) or 1,
        modalities=frozenset(m for case in tests for m in content_modalities(case.query))).validate(plugin.descriptor)
    qa = _qa_type(plugin)
    if agent is None:
        qa.validate_cases(config, train, tests)
    last = {scope_key(case.scope): case.case_id for case in train}
    if learning and agent is None:
        for case in train:
            if qa.collect_training_episode(case):
                continue  # Validate the actual episode after QA executes it.
            episode = qa.prepare_episode(case, training_episode(case),
                last_for_scope=case.case_id == last[scope_key(case.scope)])
            validate_learning(plugin, episode, case.scope)
    elif not learning:
        for case in tests:
            if case.episode is not None or case.history:
                episode = qa.prepare_episode(case, history_episode(case), last_for_scope=True)
                validate_learning(plugin, episode, case.scope)


def check_models(config, agent, services, resolve, benchmark, registry):
    from mematlas.benchmarks.factory import _scoring_needs_model
    for method in config.methods:
        supplied = resolve(services, method)
        qa_type = _qa_type(registry.get(method.method.method_id))
        needs_answer = agent is None and qa_type.requires_answer_model_for(config, benchmark)
        if not (needs_answer or _scoring_needs_model(config.benchmark, benchmark, supplied)):
            continue
        if supplied.object("answer_llm") is not None:
            continue
        provider = config.answer_model
        if provider is None or not provider.model:
            raise ValueError("Provide experiment.answer_model, or agent= for a custom Agent; native judges also need a model")
        if not os.getenv(provider.api_key_env):
            raise ValueError(f"Missing answer model credential: {provider.api_key_env}")
        if importlib.util.find_spec("openai") is None:
            raise ValueError("Install mematlas[openai] to use the configured answer model")
