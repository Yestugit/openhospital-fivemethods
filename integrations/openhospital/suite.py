"""Mem0 RSI orchestration using the public experiment runner and service owner."""
import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, RuntimeConfig
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import git_revision, json_sha256
from mematlas.core.telemetry import UsageLedger
from mematlas.core.config import MethodConfig
from .config import HospitalOptions, OPENHOSPITAL_ROOT
from .planning import build_plan
from .services import OpenHospitalServices, terminate_owned_process
from .monitor import print_status

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/openhospital-mem0-rsi.py"
CONDITIONS = ("repeated", "novel_same_department", "novel_departments")
DEPARTMENTS = ("血液科", "呼吸内科", "消化内科", "内分泌科", "神经内科")
REFERENCE = ROOT / ".mematlas/oph-rsi/20261028-173625-929/awm"
MAAS = "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1"


def load_environment():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env.local", override=False)
    load_dotenv(OPENHOSPITAL_ROOT / ".env.local", override=False)
    os.environ.setdefault("DOCTOR_MODEL", "qwen3.5-flash")
    os.environ.setdefault("DOCTOR_MODEL_BASE_URL", MAAS)
    os.environ.setdefault("HOSPITAL_PATIENT_MODEL", "qwen3.5-flash")
    os.environ.setdefault("HOSPITAL_MODEL_BASE_URL", MAAS)
    os.environ.setdefault("LOCAL_EMBEDDING_API_KEY", "local-no-api-charge")
    # Try the existing account if no separate embedding credential was supplied.
    if not os.environ.get("EMBEDDING_API_KEY"):
        candidate = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MODEL_API_KEY")
        if candidate:
            os.environ["EMBEDDING_API_KEY"] = candidate


def model_environment():
    return {key: os.environ[key] for key in ("DOCTOR_MODEL", "DOCTOR_MODEL_BASE_URL",
            "HOSPITAL_PATIENT_MODEL", "HOSPITAL_MODEL_BASE_URL")}


def make_experiment(run, condition, *, smoke=False, embedding="local", model_dir=None):
    run = Path(run).resolve()
    cross = condition == "novel_departments"
    blocks = 5 if cross or not smoke else 11 if condition == "repeated" else 1
    benchmark = {"kind": "openhospital", "condition": condition,
        "departments": DEPARTMENTS if cross else DEPARTMENTS[:1],
        "evaluation_departments": DEPARTMENTS[:1] if smoke else DEPARTMENTS,
        "cases_per_block": 1 if smoke else 50, "blocks": blocks,
        "training_only": not cross, "heldout_per_department": (1 if smoke else 10) if cross else 0,
        "advanced_per_department": (1 if smoke else 10) if cross else 0,
        "world_dir": str(run / "world"), "doctor_config": str(ROOT / "configs/openhospital/mem0-doctor.yaml"),
        "feedback_timing": "immediate", "snapshot_keep_recent": 2, "snapshot_keep_every": 50,
        "memory_top_k": 5}
    metadata = {}
    if embedding == "local":
        from mematlas.providers.local_qwen import model_spec, BASE_URL, MODEL_ID
        metadata["local_embedding"] = model_spec(model_dir) if model_dir else model_spec()
        embedding_provider = {"model": MODEL_ID, "dimensions": 1024,
                              "base_url": BASE_URL, "api_key_env": "LOCAL_EMBEDDING_API_KEY"}
    else:
        embedding_provider = {"model": "text-embedding-v4", "dimensions": 1024,
                "base_url": os.environ.get("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                "api_key_env": "EMBEDDING_API_KEY"}
    memory = MemAtlasConfig(method=MethodConfig("mem0", {"implementation_mode": "standard", "algorithm_profile": "additive",
        "infer": True, "write_policy": "episode", "hybrid_search": True, "rerank": False, "top_k": 5}), metadata=metadata, providers={
        "llm": {"model": "qwen3.5-flash", "base_url": os.environ.get("MEMORY_MODEL_BASE_URL", MAAS),
                "api_key_env": "MODEL_API_KEY", "options": {"max_tokens": 8192,
                "extra_body": {"enable_thinking": False}, "prompt_cache": True,
                "response_journal": str(run / condition / "memory-responses.jsonl")}},
        "embedding": embedding_provider},
        runtime=RuntimeConfig(artifact_dir=str(run / condition / "attempt-001/memory")),
        scope={"user_id": f"openhospital:{condition}", "agent_id": "doctor"})
    return ExperimentConfig(methods=(memory,), benchmark=benchmark,
        protocol="experience_learning_curve", seed=0,
        checkpoints=(0, 5) if smoke and cross else (),
        output_dir=str(run / condition / "attempt-001/evaluation"))


def make_pilot(run, *, model_dir=None):
    config = make_experiment(run, "repeated", model_dir=model_dir)
    benchmark = {**config.benchmark, "training_only": False, "evaluation_departments": DEPARTMENTS[:1],
        "heldout_per_department": 2, "advanced_per_department": 0, "pilot_training_count": 20}
    return replace(config, benchmark=benchmark, checkpoints=(20,))


def validate_pilot(manifest):
    job = manifest["jobs"]["repeated"]
    progress = read_json(Path(job["output"]) / "progress.json", {})
    results = read_json(Path(job["output"]) / "results.json", [{}])
    learned = results[0].get("run_metadata", {}).get("progress", {}).get("learned_exposures")
    return {"ready": progress.get("training_completed") == 20 and progress.get("test_completed") == 2
        and learned == 20 and job["attempt"] == 2 and manifest.get("resume_boundary") == 10,
        "training_encounters": progress.get("training_completed"), "heldout_encounters": progress.get("test_completed"),
        "learned": learned, "resume_verified": job["attempt"] == 2 and learned == 20}


def audit_schedule(config, reference=REFERENCE):
    prepared = build_plan(HospitalOptions.from_experiment(config), checkpoints=config.checkpoints, seed=0)
    plan = prepared.plan
    condition = config.benchmark["condition"]
    path = Path(reference) / condition / "frozen_cohort.json"
    if not path.is_file():
        raise ValueError(f"AWM roster reference missing: {path}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    current = {"train": list(plan.train_ids), "primary_test": list(plan.primary_test_ids),
               "advanced_test": list(plan.advanced_test_ids)}
    if [list(batch) for batch in plan.batches()] != previous["training_batches"]:
        raise ValueError(f"AWM training order mismatch: {condition}")
    if current != previous["splits"]:
        raise ValueError(f"AWM patient roster mismatch: {condition}")
    for name, digest in previous.get("source_hashes", {}).items():
        from mematlas.core.provenance import file_sha256
        parts = Path(name).parts
        relative = Path(*parts[parts.index("openhospital_runtime") + 1:]) if "openhospital_runtime" in parts else None
        source = HospitalOptions.from_experiment(config).root / relative if relative else Path(name)
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(f"AWM dataset source changed: {source}")
    return {"matched": True, "current_dataset_hashes_verified": len(previous.get("source_hashes", {})),
            "reference": str(path), "training_batches_sha256": json_sha256(plan.batches()),
            **prepared.summary()}


def check_suite(configs, *, smoke=False, reference=REFERENCE):
    runner = ExperimentRunner()
    report = {}
    for condition, config in configs.items():
        if config.methods[0].metadata.get("local_embedding"):
            import importlib.util
            missing = [name for name in ("torch", "transformers", "safetensors")
                       if importlib.util.find_spec(name) is None]
            if missing:
                raise ValueError("Local embedding dependencies missing: " + ", ".join(missing))
        check = runner.check(config)
        if not smoke and not config.benchmark.get("pilot_training_count"):
            check["awm_schedule"] = audit_schedule(config, reference)
        report[condition] = check
    return {"ready": all(item["ready"] for item in report.values()), "conditions": report}


def brief_check(report):
    return {**report, "conditions": {condition: {key: value for key, value in item.items()
            if key != "methods"} for condition, item in report["conditions"].items()}}


async def check_providers(config):
    from mematlas.providers import OpenAICompatibleEmbedder, OpenAICompatibleLLM
    ledger = embedding_ledger(config)
    embedding, llm = config.methods[0].providers["embedding"], config.methods[0].providers["llm"]
    embedder = OpenAICompatibleEmbedder(**{key: embedding[key] for key in
        ("model", "base_url", "api_key_env", "dimensions")}, usage=ledger)
    from uuid import uuid4
    journal = ROOT / ".mematlas/provider-checks" / f"{uuid4().hex}.jsonl"
    model = OpenAICompatibleLLM(model=llm["model"], base_url=llm["base_url"],
        api_key_env=llm["api_key_env"], default_options={**llm["options"], "response_journal": str(journal)}, usage=ledger)
    try:
        vectors = await embedder.embed(["Clinical memory connectivity check."])
        if len(vectors) != 1 or len(vectors[0]) != 1024 or not any(vectors[0]):
            raise ValueError("Embedding response must contain one nonzero 1024-dimensional vector")
        await model.complete([{"role": "user", "content": "Reply OK."}], max_tokens=16)
        return {"ready": True, "embedding_model": embedding["model"],
                "embedding_location": "local" if config.methods[0].metadata.get("local_embedding") else "remote",
                "embedding_dimensions": 1024, "response_journal": str(journal), "usage": ledger.summary()}
    except Exception as exc:
        # Do not serialize provider error bodies or credentials.
        return {"ready": False, "error_type": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None), "usage": ledger.summary()}
    finally:
        await embedder._client.close()
        await model._client.close()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


async def worker(args):
    config = ExperimentConfig.load(args.config)
    from mematlas.core.plugin import MethodServices
    services = MethodServices(usage=embedding_ledger(config))
    results = await ExperimentRunner().evaluate(config, resume_from=args.source, services=services)
    failures = [failure for result in results for failure in result.failures]
    if any(failure.get("error_type") == "LLMRequestPaused" for failure in failures):
        return 76
    if any(failure.get("error_type") in {"HospitalPaused", "DoctorOutputPaused"} for failure in failures):
        return 75
    return 1 if failures else 0


def embedding_ledger(config):
    if config.methods[0].metadata.get("local_embedding"):
        return UsageLedger(model_pricing={config.methods[0].providers["embedding"]["model"]: {
            "input_cost_per_million": 0.0, "output_cost_per_million": 0.0}})
    return UsageLedger()


def validate_stop_targets(configs, targets):
    if not isinstance(targets, dict) or set(targets) - set(configs):
        raise ValueError("Stop targets must name existing conditions")
    for condition, target in targets.items():
        if (not isinstance(target, dict) or set(target) != {"training", "tests"}
                or any(type(value) is not int or value < 0 for value in target.values())):
            raise ValueError("Each stop target requires nonnegative integer training/tests counts")
        config = configs[condition]
        plan = build_plan(HospitalOptions.from_experiment(config), checkpoints=config.checkpoints,
                          seed=config.seed or 0).plan
        total, boundaries = 0, set()
        for batch in plan.batches():
            total += len(batch)
            boundaries.add(total)
        train = target["training"]
        tests = len(plan.primary_test_ids) * sum(point <= train for point in plan.effective_checkpoints())
        if train == plan.training_count:
            tests += len(plan.advanced_test_ids)
        if train not in boundaries or target["tests"] != tests:
            raise ValueError(f"Stop target for {condition} must follow a complete training batch and its tests")
    return targets


def next_attempt(run, condition, job):
    config = ExperimentConfig.load(job["config"])
    source = None
    # A service/startup failure may occur before any recovery state exists.
    # Fall back to the prior attempt only if this attempt executed no cases.
    from mematlas.evaluation.artifacts import recovery_checkpoint
    for candidate in (job["output"], job.get("source")):
        if not candidate:
            continue
        root = Path(candidate)
        memory_run = root / "runs/001-mem0"
        if any((memory_run / "snapshots").glob("learned_*.json")):
            recovery_checkpoint(memory_run)
            source = candidate
            break
        if any(any((root / folder).glob("*.json")) for folder in ("encounters", "test_cases", "test_execution")):
            raise ValueError("Executed hospital records have no recovery snapshot; preserve this attempt for diagnosis")
    attempt = job["attempt"] + 1
    base = Path(run) / condition / f"attempt-{attempt:03d}"
    memory = replace(config.methods[0], runtime=replace(config.methods[0].runtime, artifact_dir=str(base / "memory")))
    config = replace(config, methods=(memory,), output_dir=str(base / "evaluation"))
    job.update(attempt=attempt, output=config.output_dir, source=source, status="pending")
    atomic_write_json(base / "config.json", config.to_dict())
    job["config"] = str(base / "config.json")


def queue_api_recovery(job, now):
    """Bound automatic restarts at one durable position, reset after progress."""
    from mematlas.evaluation.artifacts import recovery_checkpoint
    position, _, _ = recovery_checkpoint(Path(job["output"]) / "runs/001-mem0")
    count = job.get("api_recoveries", 0) if job.get("api_recovery_position") == position else 0
    job["api_recovery_position"] = position
    if count >= 3:
        return False
    job["api_recoveries"] = count + 1
    job["retry_after"] = now + min(30 * 2 ** count, 120)
    job["status"] = "retrying"
    return True


async def run_suite(args):
    if args.command == "resume":
        manifest = read_json(Path(args.run_dir) / "suite.json")
        if not manifest:
            raise ValueError("Resume requires an existing suite.json")
        config = ExperimentConfig.load(next(iter(manifest["jobs"].values()))["config"])
        spec = config.methods[0].metadata.get("local_embedding")
    else:
        from mematlas.providers.local_qwen import model_spec
        spec = (model_spec(args.model_dir) if args.model_dir else model_spec()) if args.embedding == "local" else None
    async with AsyncExitStack() as stack:
        if spec:
            from mematlas.providers.local_qwen import LocalEmbeddingService
            service = await stack.enter_async_context(LocalEmbeddingService(spec,
                log_path=ROOT / ".mematlas/local-embedding/service.log", device=args.embedding_device))
            print("Local embedding ready: " + json.dumps(service.health), flush=True)
        result = await _run_suite(args)
        if args.run_dir:
            manifest = read_json(Path(args.run_dir) / "suite.json", {})
            if manifest.get("pilot"):
                from .pilot_report import write_report
                write_report(args.run_dir)
        return result


async def _run_suite(args):
    resuming = args.command == "resume"
    run = Path(args.run_dir).resolve() if args.run_dir else ROOT / ".mematlas/oph-rsi/mem0" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else "-seed0"))
    manifest_path = run / "suite.json"
    if resuming:
        manifest = read_json(manifest_path)
        if not manifest:
            raise ValueError("Resume requires an existing suite.json")
        if manifest["model_environment"] != model_environment():
            raise ValueError("Doctor or hospital model settings changed; restore the recorded environment")
        # Refuse concurrent writers, including a suite whose monitor was closed.
        import psutil
        owner = manifest.get("owner")
        if owner and psutil.pid_exists(owner["pid"]):
            try:
                if psutil.Process(owner["pid"]).create_time() == owner["created"]:
                    raise ValueError("This suite is still running; request stop and wait before resume")
            except psutil.NoSuchProcess:
                pass
        for condition, job in manifest["jobs"].items():
            if job["status"] != "completed":
                next_attempt(run, condition, job)
        configs = {condition: ExperimentConfig.load(job["config"]) for condition, job in manifest["jobs"].items()}
    else:
        if run.exists():
            raise FileExistsError("Choose a new run directory or use resume")
        configs = ({"repeated": make_pilot(run, model_dir=args.model_dir)} if args.pilot else
            {condition: make_experiment(run, condition, smoke=args.smoke, embedding=args.embedding, model_dir=args.model_dir) for condition in CONDITIONS})
        report = check_suite(configs, smoke=args.smoke, reference=args.reference)
        print(json.dumps(brief_check(report), ensure_ascii=True, indent=2))
        if not report["ready"]:
            return 1
        manifest = {"schema_version": 1, "method_id": "mem0", "pilot": args.pilot, "smoke": args.smoke, "git_revision": git_revision(ROOT),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model_environment": model_environment(), "preflight": report, "jobs": {}}
        for condition, config in configs.items():
            path = run / condition / "attempt-001/config.json"
            manifest["jobs"][condition] = {"attempt": 1, "config": str(path), "output": config.output_dir,
                                           "source": None, "status": "pending"}
    targets = read_json(args.stop_targets) if args.stop_targets else manifest.get("stop_targets", {})
    test_concurrency = getattr(args, "test_concurrency", None) or manifest.get("test_concurrency", 1)
    if type(test_concurrency) is not int or not 1 <= test_concurrency <= 16:
        raise ValueError("Test concurrency must be between 1 and 16")
    manifest["test_concurrency"] = test_concurrency
    manifest["stop_targets"] = validate_stop_targets(configs, targets)
    if all(job["status"] == "completed" for job in manifest["jobs"].values()):
        print("All conditions already completed.")
        return 0
    providers = await check_providers(next(iter(configs.values())))
    print("Provider preflight: " + json.dumps(providers))
    if not providers["ready"]:
        # No hospital service or paid encounter is started on provider failure.
        if run.exists():
            atomic_write_json(run / "provider-check.json", providers)
        return 1
    if not resuming:
        # Reserve the directory atomically if another launcher passed preflight
        # with the same run name while this one was checking providers.
        run.mkdir(parents=True, exist_ok=False)
    for condition, config in configs.items():
        if not resuming:
            atomic_write_json(manifest["jobs"][condition]["config"], config.to_dict())
    import psutil
    owner = psutil.Process()
    manifest["owner"] = {"pid": owner.pid, "created": owner.create_time()}
    manifest["provider_check"] = providers
    if next(iter(configs.values())).methods[0].metadata.get("local_embedding"):
        with urlopen("http://127.0.0.1:8766/health", timeout=5) as response:
            manifest["local_embedding_runtime"] = json.loads(response.read().decode("utf-8"))
    manifest["status"] = "running"
    stop_file = run / "STOP"
    stop_file.unlink(missing_ok=True)
    atomic_write_json(manifest_path, manifest)
    processes = {}
    logs = []

    async def start(condition, *, pause=False):
        job = manifest["jobs"][condition]
        log = (Path(job["config"]).parent / "worker.log").open("ab")
        logs.append(log)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
               "OPH_TEST_CONCURRENCY": str(manifest["test_concurrency"]),
               "OPH_STOP_FILE": str(stop_file), "OPH_PAUSE_AFTER": "10" if manifest.get("pilot") and job["attempt"] == 1 else "1" if pause else "0",
               "OPH_STOP_TRAIN_AFTER": str(manifest["stop_targets"].get(condition, {}).get("training", 0)),
               "OPH_STOP_TEST_AFTER": str(manifest["stop_targets"].get(condition, {}).get("tests", 0))}
        command = [sys.executable, "-X", "utf8", str(ENTRY), "_worker", "--config", job["config"]]
        if job["source"]:
            command += ["--source", job["source"]]
        processes[condition] = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env,
            stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        job["status"] = "running"
        job["pid"] = processes[condition].pid
        try:
            job["created"] = psutil.Process(job["pid"]).create_time()
        except psutil.NoSuchProcess:
            job.pop("created", None)

    try:
        # The transfer configuration owns all three required service splits once.
        async with OpenHospitalServices(configs.get("novel_departments", next(iter(configs.values())))):
            for condition, job in manifest["jobs"].items():
                if job["status"] != "completed":
                    await start(condition, pause=manifest["smoke"] and not resuming and condition == "repeated")
            last_print = 0
            while processes or any(job["status"] == "retrying" for job in manifest["jobs"].values()):
                for condition, process in list(processes.items()):
                    if process.returncode is None:
                        continue
                    job = manifest["jobs"][condition]
                    job["status"] = "completed" if process.returncode == 0 else "paused" if process.returncode in {75, 76} else "failed"
                    job["exit_code"] = process.returncode
                    del processes[condition]
                    # A temporary API outage must not invalidate completed cases.
                    # Back off without blocking the other conditions or the monitor.
                    if process.returncode == 76 and not stop_file.exists():
                        queue_api_recovery(job, asyncio.get_running_loop().time())
                    # The smoke deliberately verifies restart without repeating a case.
                    if process.returncode == 75 and (manifest["smoke"] or manifest.get("pilot")) and job["attempt"] == 1 and not stop_file.exists():
                        next_attempt(run, condition, job)
                        await start(condition)
                        manifest["smoke_resume_attempted"] = True
                        manifest["resume_boundary"] = 10 if manifest.get("pilot") else 1
                for condition, job in manifest["jobs"].items():
                    if job["status"] == "retrying":
                        if stop_file.exists():
                            job["status"] = "paused"
                        elif asyncio.get_running_loop().time() >= job["retry_after"]:
                            next_attempt(run, condition, job)
                            job.pop("retry_after", None)
                            await start(condition)
                atomic_write_json(manifest_path, manifest)
                now = asyncio.get_running_loop().time()
                if now - last_print >= 10:
                    print_status(manifest)
                    last_print = now
                await asyncio.sleep(0.5)
    except BaseException:
        for condition in processes:
            manifest["jobs"][condition]["status"] = "interrupted"
        raise
    finally:
        for process in processes.values():
            await terminate_owned_process(process)
        for log in logs:
            log.close()
        if next(iter(configs.values())).methods[0].metadata.get("local_embedding"):
            try:
                with urlopen("http://127.0.0.1:8766/health", timeout=5) as response:
                    manifest["local_embedding_final"] = json.loads(response.read().decode("utf-8"))
            except OSError:
                pass
        manifest.pop("owner", None)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "completed" if all(job["status"] == "completed" for job in manifest["jobs"].values()) else "incomplete"
        if manifest.get("pilot") and manifest["status"] == "completed":
            manifest["pilot_validation"] = validate_pilot(manifest)
            if not manifest["pilot_validation"]["ready"]:
                manifest["status"] = "incomplete"
        if manifest["smoke"] and manifest["status"] == "completed":
            manifest["smoke_validation"] = validate_smoke(manifest)
            manifest["smoke_resume_verified"] = manifest["smoke_validation"]["resume_verified"]
            if not manifest["smoke_validation"]["ready"]:
                manifest["status"] = "incomplete"
        atomic_write_json(manifest_path, manifest)
        print_status(manifest)
    print(f"Suite: {run}")
    return 0 if manifest["status"] == "completed" else 1


def validate_smoke(manifest):
    training, testing, learned = 0, 0, {}
    for condition, job in manifest["jobs"].items():
        root = Path(job["output"])
        training += len(list((root / "encounters").glob("*.json")))
        testing += len(list((root / "test_cases").glob("*.json")))
        results = read_json(root / "results.json", [{}])
        learned[condition] = results[0].get("run_metadata", {}).get("progress", {}).get("learned_exposures")
    repeated = manifest["jobs"]["repeated"]
    live = read_json(Path(repeated["output"]) / "progress.json", {})
    migrated = live.get("memory", {}).get("memories", 0) > 0
    resumed = repeated["attempt"] > 1 and learned.get("repeated") == 11
    return {"ready": training == 17 and testing == 3 and migrated and resumed
                and learned == {"repeated": 11, "novel_same_department": 1, "novel_departments": 5},
            "training_encounters": training, "heldout_encounters": testing, "learned": learned,
            "native_migration_verified": migrated, "resume_verified": resumed}



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "resume", "watch", "status", "monitor", "report", "stop", "_worker"))
    parser.add_argument("--run-dir")
    parser.add_argument("--port", type=int, default=8767, help="Read-only browser monitor port")
    parser.add_argument("--monitor", dest="live_monitor", action="store_true",
                        help="Run a live English dashboard and open it in the default browser (run/resume)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="20 training cases, restart after 10, then 2 frozen tests")
    parser.add_argument("--model-dir", help="Read-only existing Qwen3 embedding deployment directory")
    parser.add_argument("--env-file", help="Read credentials from an existing environment file")
    parser.add_argument("--live", action="store_true", help="Check embedding and memory LLM endpoints")
    parser.add_argument("--embedding", choices=("local", "api"), default="local",
                        help="New experiments use local Qwen3-Embedding-0.6B by default")
    parser.add_argument("--embedding-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--reference", default=str(REFERENCE))
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--stop-targets", help="JSON operational stop targets, persisted for later resumes")
    parser.add_argument("--test-concurrency", type=int, choices=range(1, 17), metavar="1..16",
                        help="Concurrent frozen test patients; training stays sequential (resume persists setting)")
    args = parser.parse_args(argv)
    if args.live_monitor and (args.command not in {"run", "resume"} or not args.run_dir):
        parser.error("--monitor requires run/resume with an explicit --run-dir")
    load_environment()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if args.pilot and (args.smoke or args.embedding != "local"):
        parser.error("--pilot requires local embeddings and cannot be combined with --smoke")
    try:
        if args.command == "_worker":
            return asyncio.run(worker(args))
        if args.command == "check":
            run = Path(args.run_dir or ROOT / ".mematlas/oph-rsi/mem0/preflight")
            configs = ({"repeated": make_pilot(run, model_dir=args.model_dir)} if args.pilot else
            {condition: make_experiment(run, condition, smoke=args.smoke, embedding=args.embedding, model_dir=args.model_dir) for condition in CONDITIONS})
            report = check_suite(configs, smoke=args.smoke, reference=args.reference)
            if args.live:
                async def live_check():
                    async with AsyncExitStack() as stack:
                        spec = configs["repeated"].methods[0].metadata.get("local_embedding")
                        if spec:
                            from mematlas.providers.local_qwen import LocalEmbeddingService
                            await stack.enter_async_context(LocalEmbeddingService(spec,
                                log_path=ROOT / ".mematlas/local-embedding/service.log", device=args.embedding_device))
                        return await check_providers(configs["repeated"])
                report["providers"] = asyncio.run(live_check())
                report["ready"] = report["ready"] and report["providers"]["ready"]
            print(json.dumps(brief_check(report), ensure_ascii=True, indent=2))
            return 0 if report["ready"] else 1
        if args.command in {"watch", "status", "monitor", "report", "stop", "resume"} and not args.run_dir:
            parser.error("--run-dir is required")
        if args.command == "report":
            from .pilot_report import write_report
            print(json.dumps(write_report(args.run_dir), ensure_ascii=False, indent=2))
            return 0
        if args.command == "monitor":
            from .monitor import serve
            serve(args.run_dir, args.port)
            return 0
        if args.command == "stop":
            run = Path(args.run_dir)
            if not (run / "suite.json").is_file():
                raise ValueError("No suite.json in the requested directory")
            (run / "STOP").touch()
            print("Stop requested. Workers will finish the active case and save recovery state.")
            return 0
        if args.command == "status":
            manifest = read_json(Path(args.run_dir) / "suite.json")
            if not manifest:
                raise ValueError("No readable suite.json")
            print_status(manifest)
            return 0
        if args.command == "watch":
            import time
            while True:
                manifest = read_json(Path(args.run_dir) / "suite.json")
                if not manifest:
                    raise ValueError("No readable suite.json in the requested directory")
                print_status(manifest)
                if manifest.get("status") != "running":
                    return 0
                time.sleep(10)
        if args.live_monitor:
            from .monitor import background_monitor
            import webbrowser
            with background_monitor(args.run_dir, args.port) as url:
                print(f"Live dashboard: {url} (read-only; refreshes every 3 seconds)", flush=True)
                try:
                    webbrowser.open(url)
                except webbrowser.Error:
                    pass
                return asyncio.run(run_suite(args))
        return asyncio.run(run_suite(args))
    except KeyboardInterrupt:
        print("Interrupted. Use resume with the recorded suite directory.")
        return 130
    except Exception as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1
