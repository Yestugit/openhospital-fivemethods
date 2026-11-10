"""Prepare the hospital scene behind the common ExperimentRunner entry."""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4
from urllib.request import urlopen
import asyncio

from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import file_sha256, json_sha256
from mematlas.core.contracts import ProtocolRequirements
from mematlas.core.capabilities import Operation
from .config import HospitalOptions, OPENHOSPITAL_ROOT, SERVICE_URLS
from .planning import build_plan
from .bridge import OpenHospitalMemoryIntegration
from .execution import HospitalExecution
from .rsi import _HospitalEvaluation


def audited_source_repair(prior, current, suite_root):
    """Accept explicit hash-audited hotfixes, never changed clinical contracts.

    Keep the original logical recovery contract so existing checkpoint hashes
    remain valid. Actual source hashes and their audit paths remain separate.
    """
    source_fields = {"framework_sources", "scene_sources"}
    old = {k:v for k,v in prior.items() if k not in source_fields}
    new = {k:v for k,v in current.items() if k not in source_fields}
    if old != new:
        return False
    suite_root = Path(suite_root).resolve()
    try:
        suite = json.loads((suite_root / "suite.json").read_text(encoding="utf-8"))
        transitions = {}
        for reference in suite.get("source_repairs", []):
            path = Path(reference).resolve()
            if not path.is_relative_to(suite_root / "repairs"):
                return False
            audit = json.loads(path.read_text(encoding="utf-8"))
            for name, change in audit["changes"].items():
                name = name.replace("\\", "/")
                transitions.setdefault(name, []).append((change["before"], change["after"]))
        signature = {k.replace("\\", "/"):v for k,v in suite["source_signature"].items()}
        for field in source_fields:
            previous, actual = prior.get(field, {}), current.get(field, {})
            if previous.keys() != actual.keys():
                return False
            for name, initial in previous.items():
                key = name.replace("\\", "/")
                if field == "scene_sources":
                    key = "integrations/openhospital_runtime/" + key
                expected = initial
                # A method may have started after earlier repairs. Follow only
                # exact hash transitions reachable from that method's baseline.
                for before, after in transitions.get(key, []):
                    if expected == before:
                        expected = after
                if expected != actual[name] or signature.get(key) != expected:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_doctor_runtime(hospital_root=OPENHOSPITAL_ROOT):
    sdk_root = Path(hospital_root) / "hospital-agent-sdk"
    example_root = Path(hospital_root) / "hospital_agent_example"
    for source in (sdk_root, example_root):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    from dotenv import load_dotenv
    load_dotenv(Path(hospital_root) / ".env.local", override=False)
    from agent import MyDoctorAgent
    from hospital_agent_sdk import runtime_config_from_project_config
    return MyDoctorAgent, runtime_config_from_project_config


def _doctor_config(path: str | Path, output_dir: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Doctor config must be a mapping")
    config["output_dir"] = str(output_dir / "official_sdk")
    config["rsi_strict_outputs"] = True
    return config


def _runtime_config(
    project_config: Mapping[str, Any],
    *,
    patient_ids: Sequence[str],
    service_url: str,
    mode: str,
    team_id: str,
    hospital_root=OPENHOSPITAL_ROOT,
):
    config = copy.deepcopy(dict(project_config))
    selection = {
        "selection": "forward",
        "patient_count": len(patient_ids),
        "patient_ids": list(patient_ids),
    }
    config[mode] = selection
    _, runtime_config_from_project_config = load_doctor_runtime(hospital_root)
    runtime = runtime_config_from_project_config(config, {
        "service_base_url": service_url,
        "local_test": mode == "test",
        "team_id": team_id,
    })
    if mode == "train":
        runtime.train.patient_ids = list(patient_ids)
        runtime.train.patient_count = len(patient_ids)
        runtime.train.selection = "forward"
    else:
        runtime.test.patient_ids = list(patient_ids)
        runtime.test.patient_count = len(patient_ids)
        runtime.test.selection = "forward"
    return config, runtime


class OpenHospitalScene:
    """Pure preflight followed by per-method live hospital preparation."""
    path_options = ("path", "hospital_root", "doctor_config", "world_dir")

    def __init__(self, config, *, registry):
        self.options = HospitalOptions.from_experiment(config)
        self.config = replace(config, benchmark={**dict(config.benchmark),
            "hospital_root": str(self.options.root), "path": str(self.options.roster),
            "doctor_config": str(self.options.doctor),
            "world_dir": str(Path(self.options.world_dir).expanduser().resolve())})
        self.registry = registry
        self.prepared = None

    def _check_experiment(self, config):
        from dotenv import load_dotenv
        load_dotenv(self.options.root / ".env.local", override=False)
        if config.protocol != "experience_learning_curve" or len(config.methods) != 1:
            raise ValueError("OpenHospital requires experience_learning_curve and one Memory configuration")
        if config.train_limit is not None or config.test_limit is not None or config.train_benchmark is not None:
            raise ValueError("Set hospital case counts in benchmark; train_limit/test_limit/train_benchmark are unused")
        if not self.options.doctor.is_file():
            raise ValueError(f"Doctor configuration not found: {self.options.doctor}")
        if not os.environ.get("MODEL_API_KEY"):
            raise ValueError("Set MODEL_API_KEY in the environment or hospital .env.local for the doctor and hospital models")
        _doctor_config(self.options.doctor, Path(config.output_dir or '.'))
        plugin = self.registry.get(config.methods[0].method.method_id)
        descriptor = plugin.descriptor
        ProtocolRequirements(operations=frozenset({Operation.SEARCH, Operation.ADD, Operation.EPISODE_END, Operation.SAVE, Operation.LOAD}),
                             episode_fields=frozenset({"task", "trajectory", "success", "reward"}),
                             frozen=True, recovery=True).validate(descriptor)
        from mematlas.evaluation.qa import _qa_type
        adapter = _qa_type(plugin)
        errors = plugin.validate_input("search", adapter.retrieval_extensions({"task_type": "clinical_reasoning"}))
        if errors:
            raise ValueError("Hospital retrieval input: " + "; ".join(errors))
        from mematlas.core.models import MemoryEpisode, MemoryEvent
        from mematlas.core.evaluation import BenchmarkCase
        sample = MemoryEpisode(task="Clinical preflight", scope=config.methods[0].scope,
            events=[MemoryEvent("action", "Clinical action")],
            extensions={"openhospital": {"patient_id": "preflight", "department": self.options.departments[0]}})
        sample = adapter.prepare_episode(BenchmarkCase("preflight", sample.task), sample, last_for_scope=False)
        for event in sample.events:
            if descriptor.capabilities.accepted_events and event.kind not in descriptor.capabilities.accepted_events:
                raise ValueError(f"Hospital learning adapter produced unsupported event kind: {event.kind.value}")
        self.prepared = build_plan(self.options, checkpoints=config.checkpoints, seed=config.seed or 0)
        return self.prepared.summary()

    async def _evaluate_experiment(self, runner, config, method, directory, source, services, evaluator):
        output = Path(directory).parent.parent
        source_root = Path(source).parent.parent if source else None
        prior = json.loads((source_root / "execution.json").read_text(encoding="utf-8")) if source_root else None
        if source_root and any(output.iterdir()):
            # ExperimentRunner has already written experiment.json for recovery.
            unexpected = {p.name for p in output.iterdir()} - {"experiment.json", "runs"}
            if unexpected:
                raise FileExistsError("Resume output must be a new run directory")
        team_id = prior["team_id"] if prior else "rsi_" + uuid4().hex
        contracts = {}
        for split in self.options.splits:
            def health(split=split):
                with urlopen(self.options.service_urls[split] + "/health", timeout=10) as response:
                    return json.loads(response.read().decode("utf-8"))
            info = await asyncio.to_thread(health)
            if info.get("dataset") != split or not info.get("world_id") or not info.get("contract"):
                raise ValueError(f"Hospital service {split} has an invalid dataset or world contract")
            contracts[split] = {"world_id": info["world_id"], "contract": info["contract"]}
        root = self.options.root
        agent_class, _ = load_doctor_runtime(root)
        doctor = _doctor_config(self.options.doctor, output)
        sources = {}
        for folder in (root / "hospital_agent_example/agent", root / "hospital-agent-sdk/hospital_agent_sdk"):
            sources.update({str(p.relative_to(root)): file_sha256(p) for p in folder.glob("*.py")})
        contract = {"team_id": team_id, "services": contracts,
            "doctor_model": {"model": os.environ.get("DOCTOR_MODEL", "qwen3.5-flash"),
                             "base_url": os.environ.get("DOCTOR_MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")},
            "doctor_config": file_sha256(self.options.doctor), "scene_sources": sources,
            "feedback": {"timing": self.options.feedback_timing, "delay": self.options.feedback_delay,
                         "every": self.options.feedback_every}}
        from mematlas.evaluation.execution import _agent_identity
        from mematlas.evaluation.qa import _qa_type
        adapter = _qa_type(self.registry.get(method.method.method_id))
        contract["method_adapter"] = {name: _agent_identity(getattr(adapter, name))
                                      for name in ("retrieve", "retrieval_extensions", "prepare_episode")}
        local_embedding = method.metadata.get("local_embedding")
        if local_embedding:
            from mematlas.providers.local_qwen import model_spec
            if model_spec(local_embedding["model_dir"]) != local_embedding:
                raise ValueError("Local embedding deployment differs from the frozen experiment")
            with urlopen(method.providers["embedding"]["base_url"].removesuffix("/v1") + "/health", timeout=10) as response:
                embedding_health = json.loads(response.read().decode("utf-8"))
            if embedding_health.get("fingerprint") != local_embedding["fingerprint"]:
                raise ValueError("Local embedding service weights differ from the experiment")
            contract["local_embedding"] = {key: value for key, value in embedding_health.items()
                if key not in {"exact_cache_hits", "inferred_texts", "inferred_tokens", "saved_tokens"}}
        framework = Path(__file__).resolve().parents[2]
        contract["framework_sources"] = {
            str(p.relative_to(framework)): file_sha256(p)
            for folder in ("core", "runtime", "evaluation", "integrations/openhospital", f"methods/{method.method.method_id}")
            for p in (framework / folder).rglob("*")
            if p.is_file() and (p.suffix == ".py" or p.name == "method.yaml")}
        contract = json.loads(json.dumps(contract))
        if prior and prior["contract"] != contract:
            if not audited_source_repair(prior["contract"], contract, source_root.parents[2]):
                raise ValueError("Resume conditions changed: doctor, hospital world, source or feedback schedule")
            atomic_write_json(output / "source-compatibility.json", {
                "actual_framework_sources":contract["framework_sources"],
                "original_framework_sources":prior["contract"]["framework_sources"],
                "actual_scene_sources":contract["scene_sources"],
                "original_scene_sources":prior["contract"]["scene_sources"],
                "audit_suite":str(source_root.parents[2] / "suite.json")})
            contract = prior["contract"]
        manifest = self.prepared.freeze(output)
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        if source_root:
            previous = json.loads((source_root / "frozen_cohort.json").read_text(encoding="utf-8"))
            if previous != frozen:
                raise ValueError("Resume requires the original hospital cases, schedule and data")
            import shutil
            for folder in ("encounters", "test_cases", "test_execution", "case_usage"):
                if (source_root / folder).exists():
                    shutil.copytree(source_root / folder, output / folder, dirs_exist_ok=True)
        memory = OpenHospitalMemoryIntegration.from_config(method, registry=self.registry, services=services,
            top_k=self.options.memory_top_k,
            department=self.options.departments[0] if len(self.options.departments) == 1 else "shared_doctor",
            patient_departments=self.prepared.labels, feedback_timing=self.options.feedback_timing,
            feedback_delay=self.options.feedback_delay, journal_dir=output / "encounters")
        try:
            atomic_write_json(output / "execution.json", {"team_id": team_id, "contract": contract})
            execution = HospitalExecution(plan=self.prepared.plan, output_dir=output, source_dir=source_root,
                agent_class=agent_class, doctor_config=doctor,
                runtime_builder=partial(_runtime_config, hospital_root=root), service_urls=self.options.service_urls, team_id=team_id)
            adapter = _HospitalEvaluation(self.prepared.plan, memory, execution,
                execution_metadata={"case_resume": True, "protocol_fingerprint": json_sha256(frozen),
                                    "snapshot_keep_recent": self.options.snapshot_keep_recent,
                                    "snapshot_keep_every": self.options.snapshot_keep_every,
                                    "resume_contract": contract})
            return await adapter._evaluate_experiment(runner, config, method, directory, source, services, evaluator)
        finally:
            await memory.close()
