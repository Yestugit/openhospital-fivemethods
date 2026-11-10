import json
from dataclasses import replace
import pytest
from mematlas import ExperimentConfig, ExperimentRunner, Memory, MemoryContribution, MemoryEpisode, MemoryEvent
from mematlas.core.evaluation import BenchmarkCase, Prediction
from mematlas.core.models import content_text
from mematlas.benchmarks import JsonDatasetBenchmark, LoCoMoBenchmark, LongMemEvalBenchmark
from mematlas.test.evaluation.support import FixturePlugin, config_for, registry_for


def write_json(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_custom_episodes_and_event_identity_are_stable(tmp_path):
    path=write_json(tmp_path/"cases.json", [{"case_id":"train", "split":"train", "scope":{"user_id":"alice"},
        "history":[{"role":"user","text":"8443"}], "query":"port",
        "episode":{"task":"port", "success":True, "extensions":{"fixture":{"end":True}}}}])
    benchmark=JsonDatasetBenchmark(path)
    first,second=tuple(benchmark.cases("train"))[0],tuple(benchmark.cases("train"))[0]
    assert first.history == second.history
    assert first.episode == second.episode
    assert first.episode.success is True and first.episode.events == list(first.history)
    assert first.episode.extensions["fixture"]["end"] is True


def test_native_datasets_use_current_content_and_applied_sources(tmp_path):
    path=write_json(tmp_path/"locomo.json",[{"sample_id":"s", "conversation":{
        "speaker_a":"A", "speaker_b":"B", "session_1":[{"speaker":"A","text":"8443","dia_id":"D1:1","img_file":"photo.png"}]},
        "qa":[{"question":"port", "answer":"8443","category":2,"evidence":["D1:1"]}]}])
    dataset=LoCoMoBenchmark(path,mode="image_text")
    case=next(iter(dataset.cases()))
    assert case.query == "port" and case.scope["user_id"] == "locomo:s"
    assert case.history[0].content[1]["type"] == "image"
    assert case.history[0].extensions["dia_id"] == "D1:1"
    used=MemoryContribution(items=[{"content":"8443","source_event_ids":["D1:1"]},
                                  {"content":"unused","source_event_ids":["D1:2"]}],
        context=[{"content":"8443", "source_item_indices":[0]}])
    prediction=Prediction(case.case_id,"8443",case.reference,(used,),metadata={"case_metadata":case.metadata})
    assert dataset.native_score([prediction])["retrieval_recall"] == 1
    assert dataset.native_score([replace(prediction,applied_memory=())])["retrieval_recall"] == 0
    path=write_json(tmp_path/"long.json",[{"question_id":"q","question":"port","answer":"8443",
        "question_type":"single-session-user","haystack_sessions":[[{"role":"user","content":"8443"}]],
        "haystack_session_ids":["session-1"],"answer_session_ids":["session-1"]}])
    dataset=LongMemEvalBenchmark(path)
    case=next(iter(dataset.cases()))
    assert case.history[0].content == "8443"
    used=MemoryContribution(items=[{"content":"8443","source_event_ids":["session-1:turn:0"]}])
    prediction=Prediction(case.case_id,"8443",case.reference,(used,),metadata={"case_metadata":case.metadata})
    assert dataset.native_score([prediction])["session_recall"] == 1


@pytest.mark.asyncio
async def test_native_benchmark_runs_and_scores_with_shared_history(tmp_path):
    path=write_json(tmp_path/"locomo.json",[{"sample_id":"s", "conversation":{
        "speaker_a":"A", "session_1":[{"speaker":"A","text":"8443","dia_id":"D1:1"}]},
        "qa":[{"question":"port", "answer":"8443","category":2,"evidence":["D1:1"]},
              {"question":"port again", "answer":"8443","category":2,"evidence":["D1:1"]}]}])
    plugin=FixturePlugin()
    runner=ExperimentRunner(registry=registry_for(plugin))
    experiment=ExperimentConfig.from_dict({"methods":[config_for(tmp_path/"memory")],
        "benchmark":{"kind":"locomo","path":str(path)},"output_dir":str(tmp_path/"report")})
    async def answer(case, memory):
        used = await memory.search(case.query, scope=case.scope)
        return MemoryEpisode(output="8443" if "8443" in content_text(used.items[0]["content"]) else "unavailable",
                             applied_memory=[used])
    assert runner.check(experiment,agent=answer)["ready"]
    result=(await runner.evaluate(experiment,agent=answer))[0]
    assert not result.failures
    assert result.metrics["native.qa_f1"] == 1 and result.metrics["native.retrieval_recall"] == 1
    assert len(plugin.instances[0].episodes) == 1
    assert (tmp_path/"report/results.json").exists()
    from mematlas.core.provenance import git_revision
    import mematlas
    from pathlib import Path
    assert result.run_metadata["framework"]["git_revision"] == git_revision(Path(mematlas.__file__).parent)


@pytest.mark.asyncio
async def test_complete_training_episode_preserves_real_feedback(tmp_path):
    path=write_json(tmp_path/"custom.json",[
        {"case_id":"train","split":"train","scope":{"user_id":"alice"}, "query":"port",
         "history":[{"role":"user","text":"8443"}],"episode":{"task":"port","success":False}},
        {"case_id":"test","scope":{"user_id":"alice"},"query":"port","reference":"8443"}])
    plugin=FixturePlugin(required_feedback=True)
    runner=ExperimentRunner(registry=registry_for(plugin))
    config=ExperimentConfig.from_dict({"methods":[config_for(tmp_path/"memory")],
        "benchmark":{"kind":"custom","path":str(path)},"protocol":"experience_train_test"})
    async def answer(case,memory):
        used = await memory.search(case.query, scope=case.scope)
        return MemoryEpisode(output=content_text(used.items[0]["content"]), applied_memory=[used])
    from mematlas import MethodServices
    from mematlas.providers import CallableLLM
    supplied = MethodServices(objects={"answer_llm": CallableLLM(lambda messages, **options: "8443")})
    assert runner.check(config,services=supplied)["ready"]
    result=(await runner.evaluate(config,services=supplied))[0]
    assert not result.failures and plugin.instances[0].episodes[0]["success"] is False
    rows=json.loads(path.read_text()); rows[0].pop("episode"); write_json(path,rows)
    report=runner.check(config,services=supplied)
    assert not report["ready"] and "success" in str(report["methods"][0]["errors"])


@pytest.mark.asyncio
async def test_agent_auto_episode_id_and_existing_learning_are_preserved(tmp_path):
    from mematlas.benchmarks.datasets import InMemoryBenchmark
    plugin = FixturePlugin()
    runner = ExperimentRunner(registry=registry_for(plugin))
    benchmark = InMemoryBenchmark("agent", [
        BenchmarkCase("train", "question", scope={"user_id": "alice"}, metadata={"split": "train"}),
        BenchmarkCase("test", "question", scope={"user_id": "alice"})])
    calls = []
    async def agent(case, memory):
        calls.append((case.case_id, memory.is_read_only))
        used = await memory.search(case.query, scope=case.scope)
        return MemoryEpisode(events=[MemoryEvent("message", "answer", role="assistant")],
                             output="answer", applied_memory=[used])
    experiment = ExperimentConfig.from_dict({"methods": [config_for(tmp_path / "memory")],
        "benchmark": benchmark, "protocol": "experience_learning_curve", "checkpoints": (0, 1),
        "output_dir": str(tmp_path / "results")})
    result = (await runner.evaluate(experiment, agent=agent))[0]
    assert not result.failures
    assert len(plugin.instances[0].episodes) == 1
    assert plugin.instances[0].episodes[0]["episode_id"] != "train"
    assert calls == [("test", True), ("train", False), ("test", True)]
    assert result.predictions[0].applied_memory


def test_cli_runs_custom_benchmark_through_v2_runner(tmp_path,monkeypatch,capsys):
    import importlib
    from mematlas.providers import CallableLLM
    cli=importlib.import_module("mematlas.cli.main")
    monkeypatch.setenv("EVALUATION_TEST_KEY", "local-test")
    monkeypatch.setattr("mematlas.providers.OpenAICompatibleLLM", lambda **kwargs: CallableLLM(lambda messages, **options: "8443"))
    monkeypatch.setattr(cli,"default_registry",registry_for)
    data=write_json(tmp_path/"cases.json",[{"case_id":"one","history":[{"role":"user","text":"8443"}],
        "query":"port","reference":"8443"}])
    config=write_json(tmp_path/"experiment.json", {"methods": [config_for(tmp_path/"memory")],
        "benchmark": {"kind": "custom", "path": str(data)}, "output_dir": str(tmp_path/"report"),
        "answer_model": {"model": "local-test", "api_key_env": "EVALUATION_TEST_KEY"}})
    assert cli.main(["benchmark","custom",str(data)]) == 0
    assert json.loads(capsys.readouterr().out)["case_count"] == 1
    assert cli.main(["check",str(config),"--experiment"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"]
    assert cli.main(["run","--config",str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["failures"] == 0
    assert cli.main(["results",str(tmp_path/"report")]) == 0
    assert json.loads(capsys.readouterr().out)[0]["metrics"]["common.exact_match"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method",["mem0","memoryos"])
async def test_public_benchmark_runs_with_real_local_method_storage(tmp_path,method):
    from mematlas.core.contributions import contribution_text
    params={"implementation_mode":"offline"}
    supplied = None
    if method=="mem0":
        from mematlas.test.mem0.native_support import require_native, services
        require_native()
        params = {"infer": False, "embedding_dims": 16}
        supplied = services()
    path=write_json(tmp_path/"cases.json",[{"case_id":"one","query":"Helios port","reference":"8443",
        "history":[{"role":"user","text":"Remember the Helios configuration"},
                   {"role":"assistant","text":"Helios port is 8443"}]}])
    experiment=ExperimentConfig.from_dict({"methods":[{"method":{"method_id":method,"params":params},
        "runtime":{"artifact_dir":str(tmp_path/"memory")}}],
        "benchmark":{"kind":"custom","path":str(path)},"output_dir":str(tmp_path/"report")})
    async def answer(case,memory):
        used = await memory.search(case.query, scope=case.scope)
        output="8443" if "8443" in contribution_text(used) else "unavailable"
        return MemoryEpisode(output=output, applied_memory=[used])
    result=(await ExperimentRunner().evaluate(experiment,agent=answer,services=supplied))[0]
    assert not result.failures
    assert result.metrics["common.exact_match"] == 1
    assert result.predictions[0].applied_memory
