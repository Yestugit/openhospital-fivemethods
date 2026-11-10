"""Public hospital configuration, scene selection, progress and owned services."""
import asyncio
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import socket

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, BatchResult, MemoryEpisode
from mematlas.core.evaluation import EvaluationResult
from mematlas.integrations.openhospital import OpenHospitalServices
from mematlas.integrations.openhospital.config import SERVICE_URLS
from mematlas.integrations.openhospital.execution import HospitalExecution
from mematlas.integrations.openhospital.metrics import aggregate_openhospital_evaluations
from mematlas.test.evaluation.support import FixturePlugin, config_for, registry_for
from mematlas.test.evaluation.test_openhospital_adaptation import REPORT, STATE, FINAL


def experiment_for(path, *, condition="repeated", advanced=1, **settings):
    departments = ["心内科", "呼吸内科"] if condition == "novel_departments" else ["心内科"]
    return ExperimentConfig(
        methods=(MemAtlasConfig.from_dict(config_for(path / "memory")),),
        benchmark={"kind": "openhospital", "condition": condition, "departments": departments,
                   "evaluation_departments": ["心内科"], "cases_per_block": 2, "blocks": 2,
                   "heldout_per_department": 1, "advanced_per_department": advanced, **settings},
        protocol="experience_learning_curve", seed=0, output_dir=str(path / "run"))


@pytest.mark.parametrize("condition,unique", [("repeated", 2), ("novel_same_department", 4), ("novel_departments", 4)])
def test_preflight_selects_scene_and_counts_without_services_or_memory(tmp_path, condition, unique):
    plugin = FixturePlugin(required_feedback=True)
    config = experiment_for(tmp_path, condition=condition)
    report = ExperimentRunner(registry=registry_for(plugin)).check(config)
    assert report["ready"], report
    assert report["train_unique_count"] == unique
    assert report["train_exposure_count"] == 4
    assert report["primary_test_count"] == report["advanced_test_count"] == 1
    assert report["checkpoints"] == [0, 2, 4]
    assert not plugin.instances and not (tmp_path / "run").exists()
    assert not (tmp_path / "memory").exists()


def test_custom_checkpoints_and_invalid_options_are_checked_before_execution(tmp_path):
    runner = ExperimentRunner(registry=registry_for())
    config = replace(experiment_for(tmp_path, advanced=0), checkpoints=(0, 1, 4))
    report = runner.check(config)
    assert report["ready"] and report["checkpoints"] == [0, 1, 4]
    for options in ({"smoke_train": 1}, {"feedback_generation": "on_delivery"},
                    {"feedback_every": 2}, {"cases_per_block": True}):
        failed = runner.check(experiment_for(tmp_path, **options))
        assert not failed["ready"] and failed["errors"]
    assert not (tmp_path / "run").exists()


def test_hospital_file_paths_resolve_relative_to_experiment_configuration(tmp_path):
    config = experiment_for(tmp_path)
    values = config.to_dict()
    values["benchmark"].update(doctor_config="doctor.yaml", hospital_root="hospital", world_dir="world")
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    loaded = ExperimentConfig.load(path)
    for key, name in (("doctor_config", "doctor.yaml"), ("hospital_root", "hospital"), ("world_dir", "world")):
        assert loaded.benchmark[key] == str((tmp_path / name).resolve())


@pytest.mark.asyncio
async def test_recorded_dataset_path_survives_a_different_working_directory(tmp_path, monkeypatch):
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{"case_id": "one", "query": "question", "reference": "answer"}]), encoding="utf-8")
    config = replace(experiment_for(tmp_path), benchmark={"kind": "custom", "path": "cases.json"},
                     protocol="streaming_memory")
    monkeypatch.chdir(tmp_path)
    runner = ExperimentRunner(registry=registry_for())
    async def agent(case, memory):
        return MemoryEpisode(output="answer")
    result = (await runner.evaluate(config, agent=agent))[0]
    assert not result.failures
    loaded = ExperimentConfig.load(config.output_dir)
    assert loaded.benchmark["path"] == str(dataset)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert runner.check(loaded, agent=agent)["ready"]


def install_scene_execution(monkeypatch, *, interrupt=False):
    import mematlas.integrations.openhospital.application as application
    control = {"training": [], "testing": [], "interrupt": interrupt}

    def health(url, **kwargs):
        split = next(key for key, value in SERVICE_URLS.items() if url.startswith(value))
        return BytesIO(json.dumps({"dataset": split, "world_id": split, "contract": {"dataset": split}}).encode())

    monkeypatch.setattr(application, "urlopen", health)
    monkeypatch.setattr(application, "load_doctor_runtime", lambda root: (object, None))

    async def train(self, ids, bridge, index):
        batches = self.plan.batches()
        offset = sum(map(len, batches[:index - 1]))
        bridge.configure_training_batch(ids, batch_index=index, end_of_training=index == len(batches))
        for n, patient in enumerate(ids, 1):
            occurrence = bridge._occurrence(patient)
            status = bridge.history.result(occurrence, scope=bridge.scope)
            if status.learning == "submitted" or status.episode_id in bridge.history.pending_feedback:
                continue
            if control["interrupt"] and control["training"]:
                control["interrupt"] = False
                raise RuntimeError("interrupted after a completed diagnosis")
            await bridge.learn_case(patient_id=patient, reflection={"summary": "clinical feedback"},
                                    case_state=STATE, evaluation_report=REPORT, final_result=FINAL)
            control["training"].append(occurrence)
            if self.on_case_completed:
                await self.on_case_completed(offset + n)
        await bridge.deliver_feedback(batch_end=offset + len(ids) in self.plan.feedback_boundaries)
        return BatchResult(tuple(bridge.history.result(bridge._occurrence(patient), scope=bridge.scope) for patient in ids))

    async def test(self, ids, bridge, checkpoint, split):
        assert bridge.core_memory.is_read_only
        control["testing"].append((checkpoint, split))
        rows = [{"patient_id": patient, "department": bridge._department(patient),
                 "report": REPORT, "final_result": FINAL, "applied_memory": []} for patient in ids]
        return {**aggregate_openhospital_evaluations(rows), "case_reports": rows}

    monkeypatch.setattr(HospitalExecution, "train_batch", train)
    monkeypatch.setattr(HospitalExecution, "test_batch", test)
    return control


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["repeated", "novel_same_department", "novel_departments"])
async def test_public_runner_runs_all_streams_without_user_adapter(tmp_path, monkeypatch, condition):
    control = install_scene_execution(monkeypatch)
    plugin = FixturePlugin(required_feedback=True)
    runner = ExperimentRunner(registry=registry_for(plugin))
    config = experiment_for(tmp_path, condition=condition)
    result = (await runner.evaluate(config))[0]
    assert not result.failures, result.failures
    assert len(control["training"]) == len(set(control["training"])) == 4
    assert control["testing"] == [(0, "primary_test"), (2, "primary_test"),
                                   (4, "primary_test"), (4, "advanced_test")]
    assert result.run_metadata["progress"]["completed_exposures"] == 4
    assert result.run_metadata["progress"]["learned_exposures"] == 4
    assert runner.results(config.output_dir)[0].to_dict() == result.to_dict()
    report = (Path(config.output_dir) / "report.md").read_text(encoding="utf-8")
    assert "overall_clinical_score" in report and "Learning Progress" in report
    saved = ExperimentConfig.load(config.output_dir)
    assert saved.benchmark["condition"] == condition
    assert Path(saved.benchmark["hospital_root"]).is_absolute()


@pytest.mark.asyncio
async def test_resume_reads_recorded_configuration_and_does_not_repeat_training(tmp_path, monkeypatch):
    control = install_scene_execution(monkeypatch, interrupt=True)
    plugin = FixturePlugin(required_feedback=True)
    runner = ExperimentRunner(registry=registry_for(plugin))
    original = experiment_for(tmp_path)
    failed = (await runner.evaluate(original))[0]
    assert failed.failures and len(control["training"]) == 1
    restored = replace(ExperimentConfig.load(original.output_dir), output_dir=str(tmp_path / "resumed"))
    result = (await runner.evaluate(restored, resume_from=original.output_dir))[0]
    assert not result.failures, result.failures
    assert len(control["training"]) == len(set(control["training"])) == 4
    assert result.run_metadata["progress"]["completed_exposures"] == 4
    assert result.run_metadata["progress"]["learned_exposures"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("timing", ["immediate", "batch", "delayed"])
async def test_feedback_timing_preserves_completed_and_pending_counts(tmp_path, monkeypatch, timing):
    install_scene_execution(monkeypatch)
    plugin = FixturePlugin(required_feedback=True)
    config = experiment_for(tmp_path, feedback_timing=timing, feedback_delay=1 if timing == "delayed" else 0)
    result = (await ExperimentRunner(registry=registry_for(plugin)).evaluate(config))[0]
    assert not result.failures, result.failures
    progress = result.run_metadata["progress"]
    assert progress["completed_exposures"] == 4
    assert progress["learned_exposures"] == (3 if timing == "delayed" else 4)
    assert progress["pending_feedback"] == (1 if timing == "delayed" else 0)


@pytest.mark.asyncio
async def test_new_scene_is_registered_without_runner_changes(tmp_path, monkeypatch):
    from mematlas.evaluation.scenes import SCENE_FACTORIES

    class Scene:
        def __init__(self, config, *, registry):
            pass
        def _check_experiment(self, config):
            return {"primary_test_count": 1}
        async def _evaluate_experiment(self, runner, config, method, directory, source, services, evaluator):
            return EvaluationResult(method.method.method_id, "new_scene", (), {"score": 1}, {})

    monkeypatch.setitem(SCENE_FACTORIES, "new_scene", Scene)
    config = replace(experiment_for(tmp_path), benchmark={"kind": "new_scene"})
    runner = ExperimentRunner(registry=registry_for())
    assert runner.check(config)["ready"]
    result = (await runner.evaluate(config))[0]
    assert result.benchmark_id == "new_scene" and result.metrics == {"score": 1}


@pytest.mark.asyncio
async def test_service_context_starts_only_required_services_and_closes_processes(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("psycopg")
    for split in SERVICE_URLS:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            monkeypatch.setitem(SERVICE_URLS, split, f"http://127.0.0.1:{sock.getsockname()[1]}")
    config = experiment_for(tmp_path, advanced=0, world_dir=str(tmp_path / "world"))
    async with OpenHospitalServices(config) as hospital:
        processes = list(hospital._processes)
        assert len(processes) == 2 and all(p.returncode is None for p in processes)
        assert (tmp_path / "world/train/world.sqlite3").exists()
        assert not (tmp_path / "world/advanced_test").exists()
    assert all(p.returncode is not None for p in processes)


@pytest.mark.asyncio
async def test_occupied_port_does_not_attach_to_or_stop_another_service(tmp_path, monkeypatch):
    config = experiment_for(tmp_path, advanced=0, world_dir=str(tmp_path / "world"))
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen()
        port = existing.getsockname()[1]
        monkeypatch.setitem(SERVICE_URLS, "train", f"http://127.0.0.1:{port}")
        hospital = OpenHospitalServices(config)
        with pytest.raises(RuntimeError, match="occupied"):
            async with hospital:
                pytest.fail("Service ownership conflict was accepted")
        assert not hospital._processes
        assert existing.fileno() != -1
