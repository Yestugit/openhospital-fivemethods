"""Serial suite owner. All clinical execution uses the public ExperimentRunner.

An OS lock protects the whole suite, including service startup and recovery.
Each worker is disposable; a restart loads only a validated committed snapshot.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import file_sha256, json_sha256
from mematlas.core.activity import summarize, rows, activity
from mematlas.core.recovery import error_info
from .config import HospitalOptions
from .planning import build_plan
from .services import OpenHospitalServices, terminate_owned_process
from .suite import load_environment, model_environment, embedding_ledger

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/openhospital-memory-efficiency.py"
METHODS = ("memos", "mem0", "memmachine", "emem", "memora")
SCHEMA = 1


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class SuiteLock:
    def __init__(self, directory):
        self.path = Path(directory) / "suite.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError("This suite already has an active owner") from exc
        return self

    def __exit__(self, *exc):
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.stream.close()


def make_config(run, method, settings, attempt=1):
    source = importlib.util.spec_from_file_location("adapter_config", ROOT / "scripts/memory-adapter-config.py")
    module = importlib.util.module_from_spec(source)
    source.loader.exec_module(module)
    data = module.configuration(method, root=ROOT, neo4j_uri=f"bolt://127.0.0.1:{settings['bolt_port']}" if method == "memos" else None)
    base = Path(run) / method / f"attempt-{attempt:03d}"
    data["runtime"]["artifact_dir"] = str(base / "memory")
    data["scope"] = {"user_id": settings["identity"] + ":" + method, "agent_id": "doctor"}
    from mematlas.providers.local_qwen import model_spec
    data["metadata"] = {"local_embedding": model_spec(settings["model_dir"])}
    data["providers"]["embedding"]["base_url"] = f"http://127.0.0.1:{settings['embedding_port']}/v1"
    data["providers"]["llm"]["options"].update({"extra_body": {"enable_thinking": False}, "prompt_cache": True,
        "response_journal": str(Path(run) / method / "memory-responses.jsonl")})
    if method == "mem0":
        data["method"]["params"].update(algorithm_profile="additive", infer=True)
    if method == "memos":
        data["method"]["params"]["neo4j_password_env"] = "RSI_NEO4J_PASSWORD"
        # Reopen this suite's exclusive namespace before restoring the last
        # committed graph. A new namespace would leave orphan graph copies in
        # the same physical database and inflate measured storage on recovery.
        data["method"]["params"]["storage_path"] = str(Path(run) / "memos/native-store")
    return ExperimentConfig(methods=(MemAtlasConfig.from_dict(data),),
        benchmark={"kind": "openhospital", "condition": "novel_same_department", "departments": ["血液科"],
            "evaluation_departments": ["血液科"], "cases_per_block": 1 if settings["smoke"] else 20,
            "blocks": 2 if settings["smoke"] else 5, "training_only": True,
            "heldout_per_department": 0, "advanced_per_department": 0, "memory_top_k": None,
            "doctor_config": str(ROOT / "configs/openhospital/efficiency-doctor.yaml"),
            "world_dir": str(Path(run) / method / "world"), "service_port": settings["hospital_port"],
            "feedback_timing": "immediate", "snapshot_keep_recent": 2, "snapshot_keep_every": 20},
        protocol="experience_learning_curve", seed=0, checkpoints=(), output_dir=str(base / "evaluation"))


def source_signature():
    return {str(p.relative_to(ROOT)): file_sha256(p)
            for folder in ("core", "runtime", "evaluation", "providers", "methods", "integrations/openhospital", "integrations/openhospital_runtime/hospital_agent_example", "integrations/openhospital_runtime/hospital-agent-sdk", "configs/openhospital", "scripts")
            for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml"}}


def preflight(run, settings):
    result, cohorts = {}, []
    embedding_python = Path(settings["embedding_python"])
    if not embedding_python.is_file():
        raise ValueError("Dedicated embedding interpreter is missing: " + str(embedding_python))
    import socket
    ports = [settings["embedding_port"], settings["hospital_port"], settings["bolt_port"], settings["neo4j_http_port"]]
    if len(set(ports)) != len(ports):
        raise ValueError("Service ports must be distinct")
    for port in ports:
        if not 1024 <= port <= 65535:
            raise ValueError("Invalid owned service port")
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Owned service port {port} is occupied") from exc
    os.environ.setdefault("RSI_NEO4J_PASSWORD", "preflight-placeholder-not-used")
    for method in METHODS:
        config = make_config(run, method, settings)
        report = ExperimentRunner().check(config)
        result[method] = report
        plan = build_plan(HospitalOptions.from_experiment(config), checkpoints=(), seed=0)
        cohorts.append(list(plan.plan.train_ids))
    if any(cohort != cohorts[0] for cohort in cohorts) or len(set(cohorts[0])) != len(cohorts[0]):
        raise ValueError("Five-method cohort mismatch or repeated case")
    return {"ready": all(row["ready"] for row in result.values()), "methods": result, "hardware": hardware_inventory(),
            "case_ids": cohorts[0], "case_ids_sha256": json_sha256(cohorts[0])}


def hardware_inventory():
    import platform
    import psutil
    return {"platform":platform.platform(), "cpu":platform.processor(), "logical_cpus":psutil.cpu_count(),
        "physical_cpus":psutil.cpu_count(logical=False), "ram_bytes":psutil.virtual_memory().total,
        "runner_python":sys.version, "runner_executable":sys.executable}


def run_root(job, method):
    return Path(job["output"]) / "runs" / f"001-{method}"


def recovery_position(job, method):
    from mematlas.evaluation.artifacts import recovery_checkpoint
    journal_without_snapshot = False
    for candidate in (job["output"], job.get("source")):
        if not candidate:
            continue
        root = Path(candidate) / "runs" / f"001-{method}"
        if any((root / "snapshots").glob("learned_*.json")):
            position, snapshot, payload = recovery_checkpoint(root)
            return position, candidate
        if any((Path(candidate) / "encounters").glob("*.json")):
            journal_without_snapshot = True
    if journal_without_snapshot:
        raise ValueError("Clinical journal exists without a committed recovery snapshot")
    return 0, None


def next_attempt(run, manifest, method):
    job = manifest["jobs"][method]
    position, source = recovery_position(job, method)
    previous = ExperimentConfig.load(job["config"])
    job["attempt"] += 1
    base = Path(run) / method / f"attempt-{job['attempt']:03d}"
    memory = replace(previous.methods[0], runtime=replace(previous.methods[0].runtime, artifact_dir=str(base / "memory")))
    config = replace(previous, methods=(memory,), output_dir=str(base / "evaluation"))
    if job.get("failure"):
        job.setdefault("failure_history", []).append({"attempt": job["attempt"]-1, **job["failure"]})
    job.update(config=str(Path(config.output_dir).parent / "config.json"), output=config.output_dir,
               source=source, position=position, status="pending", failure={})
    atomic_write_json(job["config"], config.to_dict())


def terminal_status(run, manifest=None):
    from .dashboard import collect, render, Console
    try:
        Console().show(render(collect(run, manifest)))
    except Exception as exc:
        # Progress presentation is an observer, never the clinical worker's
        # lifetime owner. A locked log or closed console must not kill a case.
        warning = {"type":type(exc).__name__, "message":str(exc),
            "observed_at":datetime.now(timezone.utc).isoformat(), "clinical_worker_continues":True}
        try:
            atomic_write_json(Path(run) / "observer-warning.json", warning)
            print(f"Progress temporarily unavailable ({type(exc).__name__}); experiment continues.", file=sys.stderr, flush=True)
        except Exception:
            pass


def cleanup_completed_checkpoints(run, manifest, method):
    """Remove only recovery copies after durable completion and result validation.

    A durable per-file receipt makes interrupted cleanup idempotent. Native live
    databases, encounters, model responses, timing/token ledgers and metrics stay.
    """
    from .efficiency_metrics import case_measurements
    import re
    run = Path(run).resolve()
    job = manifest["jobs"][method]
    total, rounds = (2, 2) if manifest["settings"]["smoke"] else (100, 5)
    if job["status"] != "completed" or job.get("position") != total:
        raise ValueError("Checkpoint cleanup requires a completed method")
    method_root = run / method
    root = run_root(job, method)
    results = read(Path(job["output"]) / "results.json", [])
    curve = read(root / "curve.json", [])
    if (len(results) != 1 or results[0].get("method_id") != method or results[0].get("failures")
            or len(results[0].get("predictions", [])) != total
            or {p["batch_index"] for p in curve if "batch_index" in p} != set(range(1, rounds+1))):
        raise ValueError("Complete clinical results must be retained before checkpoint cleanup")
    points = case_measurements(method_root, total)
    if [p["case_position"] for p in points] != list(range(total+1)):
        raise ValueError("Complete per-case measurements required before checkpoint cleanup")
    for index in range(rounds+1):
        if read(method_root / "activity" / f"measurement-{index}.json") is None:
            raise ValueError("Complete round measurements required before checkpoint cleanup")
    receipt_path = method_root / "checkpoint-cleanup.json"
    receipt = read(receipt_path)
    if receipt is None:
        candidates = []
        # Explicit recovery-only directories, never a whole attempt or database.
        for attempt in method_root.glob("attempt-*"):
            if not re.fullmatch(r"attempt-\d+", attempt.name):
                continue
            if attempt.is_symlink() or (hasattr(attempt,"is_junction") and attempt.is_junction()):
                raise ValueError("Recovery cleanup cannot traverse a linked attempt")
            folders = [attempt / "evaluation/runs" / f"001-{method}" / "snapshots"]
            memory_root = attempt / "memory"
            for parent, dirs, _ in os.walk(memory_root, followlinks=False):
                dirs[:] = [d for d in dirs if not Path(parent,d).is_symlink()
                           and not (hasattr(Path(parent,d), "is_junction") and Path(parent,d).is_junction())]
                for name in list(dirs):
                    if name in {"snapshots", "state-snapshots", "adapter_snapshots"}:
                        folders.append(Path(parent,name))
                        dirs.remove(name)
            for folder in folders:
                if folder.is_symlink() or (hasattr(folder,"is_junction") and folder.is_junction()):
                    raise ValueError("Recovery cleanup cannot traverse a linked directory")
                if not folder.resolve().is_relative_to(method_root.resolve()):
                    raise ValueError("Recovery directory escaped the method directory")
                for parent, dirs, names in os.walk(folder, followlinks=False):
                    dirs[:] = [d for d in dirs if not Path(parent,d).is_symlink()
                               and not (hasattr(Path(parent,d), "is_junction") and Path(parent,d).is_junction())]
                    for name in names:
                        path = Path(parent,name)
                        if path.is_symlink() or not path.resolve().is_relative_to(method_root.resolve()):
                            raise ValueError("Recovery file escaped the method directory")
                        candidates.append({"path":str(path.relative_to(run)), "bytes":path.stat().st_size,
                                           "sha256":file_sha256(path)})
        receipt = {"schema":1, "method":method, "validated_cases":total, "validated_rounds":rounds,
                   "created_at":datetime.now(timezone.utc).isoformat(), "status":"pending",
                   "files":candidates, "deleted_bytes":0}
        atomic_write_json(receipt_path, receipt)
    if receipt["status"] == "completed":
        return receipt
    for row in receipt["files"]:
        path = run / row["path"]
        if not path.resolve().is_relative_to(method_root.resolve()) or path.is_symlink():
            raise ValueError("Invalid recovery cleanup receipt path")
        if path.exists():
            if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
                raise ValueError("Recovery file changed after cleanup validation")
            path.unlink()
    receipt.update(status="completed", deleted_bytes=sum(p["bytes"] for p in receipt["files"]),
                   completed_at=datetime.now(timezone.utc).isoformat())
    atomic_write_json(receipt_path, receipt)
    return receipt


def finalize_completed_method(run, manifest, method, *, manifest_path=None, write_report=True):
    # Cleanup is post-run housekeeping: errors keep snapshots and never repeat
    # completed clinical work or block the remaining methods.
    try:
        if write_report:
            report(run)
        receipt = cleanup_completed_checkpoints(run, manifest, method)
        manifest["jobs"][method]["checkpoint_cleanup"] = {
            "status":receipt["status"], "deleted_bytes":receipt["deleted_bytes"]}
    except Exception as exc:
        manifest["jobs"][method]["checkpoint_cleanup"] = {"status":"pending", "error":str(exc)}
    atomic_write_json(manifest_path or Path(run) / "suite.json", manifest)


async def worker(args):
    from mematlas.core.plugin import MethodServices
    config = ExperimentConfig.load(args.config)
    try:
        ledger = embedding_ledger(config)
        # Failed attempts may have no valid snapshot. Their durable request
        # costs still belong to this method, even when state falls back further.
        method_root = Path(args.config).parent.parent
        for journal in sorted(method_root.glob("attempt-*/evaluation/runs/*/usage.jsonl")):
            ledger.import_journal(journal)
        results = await ExperimentRunner().evaluate(config, resume_from=args.source,
            services=MethodServices(usage=ledger))
        failures = [failure for result in results for failure in result.failures]
        atomic_write_json(Path(args.config).parent / "worker-result.json", {"failures": failures})
        return 76 if any(r.get("recoverable") for r in failures) else 75 if any(r.get("requested_stop") for r in failures) else 1 if failures else 0
    except Exception as exc:
        atomic_write_json(Path(args.config).parent / "worker-result.json", {"failures": [error_info(exc)]})
        return 76 if error_info(exc)["recoverable"] else 1


async def neo4j(action, run, settings):
    command = [str(ROOT / ".venv-memos/Scripts/python.exe"), str(ROOT / "scripts/local-memos-neo4j.py"), action,
        "--services-dir", str(Path(run) / "memos/neo4j"), "--bolt-port", str(settings["bolt_port"]),
        "--http-port", str(settings["neo4j_http_port"])]
    log_path = Path(run) / "memos/neo4j-owner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if await process.wait():
            raise RuntimeError("Owned Neo4j " + action + " failed; see " + str(log_path))


async def recovery_wait(run, method, job, manifest, stop_file):
    """Automatic recovery delay is active run cost; manual pauses are excluded."""
    previous = os.environ.get("MEMATLAS_ACTIVITY_DIR")
    os.environ["MEMATLAS_ACTIVITY_DIR"] = str(Path(run) / method / "activity")
    try:
        with activity("round", "automatic_recovery"), activity("recovery_wait", "automatic_recovery_backoff"):
            while time.time() < job["retry_at"] and not stop_file.exists():
                terminal_status(run, manifest)
                await asyncio.sleep(min(5, max(0, job["retry_at"]-time.time())))
    finally:
        if previous is None:
            os.environ.pop("MEMATLAS_ACTIVITY_DIR", None)
        else:
            os.environ["MEMATLAS_ACTIVITY_DIR"] = previous


async def execute(args):
    run = Path(args.run_dir).resolve()
    selected = getattr(args, "method", None) if args.command == "_method-owner" else None
    if selected and selected not in {"emem", "memora"}:
        raise ValueError("Only the final two methods can have isolated controllers")
    path = run / selected / "controller-suite.json" if selected else run / "suite.json"
    resuming = args.command in {"resume", "_method-owner"}
    if not resuming and path.exists():
        raise FileExistsError("Suite already exists; use resume or a new directory")
    with SuiteLock(run / selected if selected else run):
        if resuming:
            manifest = read(path)
            if not manifest or manifest.get("schema") != SCHEMA:
                raise ValueError("Unsupported or missing suite manifest; old experiments are not migrated")
            if manifest["source_signature"] != source_signature() or manifest["model_environment"] != model_environment():
                raise ValueError("Sources or model configuration changed; resume refused")
            settings = manifest["settings"]
        else:
            settings = settings_from_args(args)
            check = preflight(run, settings)
            atomic_write_json(run / "preflight.json", check)
            if not check["ready"]:
                raise ValueError("Preflight failed; see preflight.json")
            manifest = {"schema": SCHEMA, "settings": settings, "model_environment": model_environment(),
                "source_signature": source_signature(), "case_ids": check["case_ids"], "case_ids_sha256": check["case_ids_sha256"],
                "hardware":check["hardware"],
                "status": "pending", "created_at": datetime.now(timezone.utc).isoformat(), "jobs": {}}
            for method in METHODS:
                config = make_config(run, method, settings)
                job = {"attempt": 1, "status": "pending", "position": 0, "config": str(Path(config.output_dir).parent / "config.json"), "output": config.output_dir, "source": None}
                manifest["jobs"][method] = job
                atomic_write_json(job["config"], config.to_dict())
        if selected and settings.get("cpu_affinity"):
            import psutil
            psutil.Process().cpu_affinity(settings["cpu_affinity"])
        # Check surviving workers before touching any backend after owner death.
        import psutil
        for name, job in manifest["jobs"].items():
            if selected and name != selected:
                continue
            pid = job.get("pid")
            if pid and psutil.pid_exists(pid) and psutil.Process(pid).create_time() == job.get("created"):
                raise RuntimeError("An orphan suite worker is still active; stop its owner before resume")
        if manifest.get("failure"):
            manifest.setdefault("failure_history", []).append({"recorded_at":datetime.now(timezone.utc).isoformat(), **manifest.pop("failure")})
        manifest.update(status="running", owner_pid=os.getpid())
        stop_file = run / "STOP"
        if not selected:
            stop_file.unlink(missing_ok=True)
        atomic_write_json(path, manifest)
        try:
            for method in ((selected,) if selected else METHODS):
                job = manifest["jobs"][method]
                if stop_file.exists():
                    manifest["status"] = "PAUSED"
                    atomic_write_json(path, manifest)
                    return 75
                if job["status"] == "completed":
                    if job.get("checkpoint_cleanup", {}).get("status") != "completed":
                        finalize_completed_method(run, manifest, method, manifest_path=path, write_report=not selected)
                    continue
                if resuming and job["status"] != "pending":
                    next_attempt(run, manifest, method)
                while True:
                    config = ExperimentConfig.load(job["config"])
                    graph_started = False
                    try:
                        if method == "memos":
                            await neo4j("start", run, settings)
                            graph_started = True
                            credentials = read(run / "memos/neo4j/memos-neo4j-local.json")
                            os.environ["RSI_NEO4J_PASSWORD"] = credentials["password"]
                        from mematlas.providers.local_qwen import LocalEmbeddingService
                        async with AsyncExitStack() as stack:
                            started = time.perf_counter()
                            embedding = await stack.enter_async_context(LocalEmbeddingService(config.methods[0].metadata["local_embedding"],
                                log_path=Path(job["config"]).parent / "embedding.log", port=settings["embedding_port"],
                                python_executable=settings["embedding_python"], device=settings["embedding_device"]))
                            from mematlas.providers import OpenAICompatibleEmbedder
                            provider = config.methods[0].providers["embedding"]
                            warmup = OpenAICompatibleEmbedder(**{key:provider[key] for key in ("model","base_url","api_key_env","dimensions")})
                            try:
                                await warmup.embed(["Clinical memory warmup."])
                            finally:
                                await warmup._client.close()
                            await stack.enter_async_context(OpenHospitalServices(config))
                            activity_dir = run / method / "activity"
                            activity_dir.mkdir(parents=True, exist_ok=True)
                            atomic_write_json(Path(job["config"]).parent / "preparation.json", {
                                "startup_warmup_s": time.perf_counter()-started, "cache_reset": True,
                                "embedding_runtime": embedding.health})
                            options = {"method": method, "tokenizer_dir": settings["model_dir"],
                                "neo4j_data_dir": str(run / "memos/neo4j/neo4j-community-5.26.0/data")}
                            pause = settings["smoke"] and not job.get("smoke_pause_verified") and job["position"] < 1
                            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "OPH_TEST_CONCURRENCY": "1",
                                "OPH_STOP_FILE": str(stop_file), "OPH_PAUSE_AFTER": "1" if pause else "0",
                                "OPH_STOP_TRAIN_AFTER": "0", "OPH_STOP_TEST_AFTER": "0", "MEMATLAS_ACTIVITY_DIR": str(activity_dir),
                                "MEMATLAS_EFFICIENCY_OPTIONS": json.dumps(options), "MEMATLAS_ATTEMPT": str(job["attempt"])}
                            command = [sys.executable, "-X", "utf8", str(ENTRY), "_worker", "--run-dir", str(run), "--config", job["config"]]
                            if job.get("source"):
                                command += ["--source", job["source"]]
                            job["log"] = str(Path(job["config"]).parent / "worker.log")
                            with Path(job["log"]).open("ab") as log:
                                process = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                                job.update(pid=process.pid, created=psutil.Process(process.pid).create_time(), status="running")
                                atomic_write_json(path, manifest)
                                try:
                                    while process.returncode is None:
                                        terminal_status(run, manifest)
                                        await asyncio.sleep(5)
                                    code = await process.wait()
                                finally:
                                    if process.returncode is None:
                                        await terminate_owned_process(process)
                    finally:
                        if graph_started:
                            await neo4j("stop", run, settings)
                    job.pop("pid", None)
                    result = read(Path(job["config"]).parent / "worker-result.json", {})
                    job["failure"] = next(iter(result.get("failures", [])), {})
                    job["position"], _ = recovery_position(job, method)
                    if code == 0:
                        job["status"] = "completed"
                        atomic_write_json(path, manifest)
                        finalize_completed_method(run, manifest, method, manifest_path=path, write_report=not selected)
                        break
                    if code == 75 and pause and not stop_file.exists():
                        if job["position"] != 1:
                            raise RuntimeError("Smoke pause did not commit exactly one case")
                        job["smoke_pause_verified"] = True
                        next_attempt(run, manifest, method)
                        atomic_write_json(path, manifest)
                        continue
                    count = job.get("recoveries", 0) if job.get("recovery_position") == job["position"] else 0
                    if code == 76 and count < 3 and not stop_file.exists():
                        job.update(recoveries=count+1, recovery_position=job["position"], status="retrying", retry_at=time.time()+30*2**count)
                        atomic_write_json(path, manifest)
                        await recovery_wait(run, method, job, manifest, stop_file)
                        if not stop_file.exists():
                            next_attempt(run, manifest, method)
                            continue
                    job["status"] = "paused"
                    manifest["status"] = "PAUSED"
                    atomic_write_json(path, manifest)
                    return 75
            manifest["status"] = "completed"
            atomic_write_json(path, manifest)
            if not selected:
                report(run)
            return 0
        except BaseException as exc:
            manifest.update(status="PAUSED", failure=error_info(exc))
            atomic_write_json(path, manifest)
            raise


def report(run):
    import csv
    run = Path(run)
    manifest = read(run / "suite.json", {})
    table = []
    cases = []
    from .efficiency_metrics import request_inventory, request_summary
    all_requests = []
    attribution = {}
    for method in METHODS:
        job = manifest.get("jobs", {}).get(method, {})
        from .efficiency_metrics import case_measurements
        committed = job.get("position", 0)
        if job.get("output"):
            latest = read(run_root(job, method) / "latest_snapshot.json", {})
            committed = max(committed, latest.get("position", 0))
        cases.extend({**p, "method":method} for p in case_measurements(run / method, committed))
        requests, unassigned = request_inventory(run / method)
        all_requests.extend({"method":method, **r} for records in [*requests.values(), unassigned] for r in records)
        attribution[method] = {"unassigned_requests":len(unassigned), "rounds":{str(k):len(v) for k,v in requests.items()}}
        curve = read(run_root(job, method) / "curve.json", []) if job.get("output") else []
        cumulative_memory = cumulative_total = 0
        for point in curve:
            if "batch_index" not in point:
                continue
            index = point["batch_index"]
            measured = read(run / method / "activity" / f"measurement-{index}.json", {})
            cumulative_memory += measured.get("memory_method_wall_s", 0)
            cumulative_total += measured.get("round_total_s", 0)
            scalar = {k:v for k,v in measured.items() if isinstance(v, (int,float,bool,str)) or v is None}
            table.append({"method": method, "round": index, **scalar, "cumulative_memory_s": cumulative_memory,
                "cumulative_total_s": cumulative_total,
                **{k:v for k,v in point.items() if isinstance(v,(int,float,bool)) and k != "efficiency"},
                **request_summary(requests.get(index, []))})
    atomic_write_json(run / "request-attribution.json", attribution)
    atomic_write_json(run / "actual-requests.json", all_requests)
    atomic_write_json(run / "efficiency-rounds.json", table)
    case_table = [{k:v for k,v in row.items() if isinstance(v,(int,float,bool,str)) or v is None} for row in cases]
    atomic_write_json(run / "efficiency-cases.json", case_table)
    if case_table:
        fields = list(dict.fromkeys(key for row in case_table for key in row))
        with (run / "efficiency-cases.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(case_table)
    if table:
        fields = list(dict.fromkeys(key for row in table for key in row))
        with (run / "efficiency-rounds.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots = [("round", "memory_method_wall_s"), ("round", "round_total_s"), ("round", "cumulative_memory_s"), ("round", "cumulative_total_s"),
                 ("round", "memory_persistence_s"), ("round", "memory_s_per_1k_rsi_tokens"),
                 ("round", "diagnosis_accuracy"), ("round", "treatment_overall_score")]
        for x,y in plots:
            fig, ax = plt.subplots()
            for method in METHODS:
                points = [row for row in table if row["method"] == method and isinstance(row.get(y),(int,float))]
                if points:
                    ax.plot([row[x] for row in points],[row[y] for row in points], marker="o", label=method)
            ax.set(xlabel=x, ylabel=y)
            if ax.lines:
                ax.legend()
            fig.tight_layout()
            fig.savefig(run / f"{y}.png")
            plt.close(fig)
    if case_table:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for key in ("storage_bytes", "memory_text_tokens"):
            fig, ax = plt.subplots()
            for method in METHODS:
                values = [p for p in case_table if p["method"] == method]
                if values:
                    ax.plot([p["rsi_tokens"] for p in values],
                        [p[key]/1048576 if key == "storage_bytes" else p[key] for p in values],
                        marker=".", markersize=3, label=method)
            ax.set(xlabel="Cumulative accepted RSI tokens (one point per case)",
                ylabel="Memory storage (MiB)" if key == "storage_bytes" else "Retained memory text tokens")
            ax.legend()
            fig.tight_layout()
            fig.savefig(run / f"{key}.png")
            plt.close(fig)
    print(f"Report: {run / 'efficiency-rounds.json'}", flush=True)
    return table


def settings_from_args(args):
    return {"identity": "rsi_" + uuid4().hex, "smoke": args.command == "smoke" or args.smoke,
        "model_dir": str(Path(args.model_dir).resolve()), "embedding_python": str(Path(args.embedding_python).resolve()),
        "embedding_port": args.embedding_port, "embedding_device": args.embedding_device,
        "hospital_port": args.hospital_port, "bolt_port": args.bolt_port, "neo4j_http_port": args.neo4j_http_port}


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check","smoke","run","resume","status","watch","dashboard","stop","report","parallel","_method-owner","_worker"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-dir", default=str(ROOT / "models/Qwen3-Embedding-0.6B"))
    parser.add_argument("--embedding-python", default=str(ROOT / ".venv-embedding/Scripts/python.exe"))
    parser.add_argument("--embedding-device", choices=("auto","cpu","cuda"), default="auto")
    parser.add_argument("--embedding-port", type=int, default=18766)
    parser.add_argument("--hospital-port", type=int, default=18001)
    parser.add_argument("--bolt-port", type=int, default=18687)
    parser.add_argument("--neo4j-http-port", type=int, default=18474)
    parser.add_argument("--dashboard-port", type=int, default=18888, help="Read-only web dashboard, loopback only")
    parser.add_argument("--env-file")
    parser.add_argument("--smoke", action="store_true", help="Use smoke settings for check only")
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("emem","memora"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke and args.command != "check":
        parser.error("--smoke applies only to check; use the smoke command for a live pilot")
    load_environment()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    run = Path(args.run_dir)
    if args.command == "check":
        result = preflight(run, settings_from_args(args))
        atomic_write_json(run / "preflight.json", result)
        print(json.dumps({"ready":result["ready"], "cases":len(result["case_ids"]), "details":str(run / "preflight.json")}),flush=True)
        return 0 if result["ready"] else 1
    if args.command == "parallel" or (args.command == "resume" and read(run / "suite.json", {}).get("execution_mode") == "parallel_final_two"):
        from .parallel import execute_parallel
        return asyncio.run(execute_parallel(args))
    if args.command in {"smoke","run","resume","_method-owner"}:
        return asyncio.run(execute(args))
    if args.command == "_worker":
        return asyncio.run(worker(args))
    if args.command == "stop":
        if not (run / "suite.json").exists():
            raise ValueError("Unknown suite")
        (run / "STOP").write_text("Pause at the next complete case boundary\n",encoding="utf-8")
        print("Stop requested; waiting for a complete case boundary.")
    elif args.command == "report":
        report(run)
    elif args.command == "watch":
        from .dashboard import collect, render, Console
        console = Console()
        while True:
            state = collect(run)
            console.show(render(state))
            if state["status"] in {"completed","PAUSED"}:
                break
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                break
    elif args.command == "dashboard":
        from .dashboard import serve
        serve(run, args.dashboard_port)
    else:
        terminal_status(run)
    return 0
