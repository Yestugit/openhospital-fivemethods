"""Isolated final-two controllers; only this parent writes the suite manifest."""
import asyncio
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from mematlas import ExperimentConfig, ExperimentRunner
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import json_sha256
from mematlas.core.recovery import error_info
from .efficiency import (SuiteLock, METHODS, ENTRY, ROOT, read, source_signature,
                         model_environment, terminal_status, report)
from .planning import build_plan
from .config import HospitalOptions
from .services import terminate_owned_process

FINAL_METHODS = ("emem", "memora")
PORTS = {"emem": {"embedding_port":18767, "hospital_port":18002},
         "memora": {"embedding_port":18768, "hospital_port":18003}}


def prepare_controller(run, manifest, method, affinity):
    """Configure only an untouched pending method; never mutate a resume world."""
    job = manifest["jobs"][method]
    path = run / method / "controller-suite.json"
    if path.exists():
        previous = read(path)
        if (previous["settings"]["identity"] != manifest["settings"]["identity"]
                or previous["case_ids_sha256"] != manifest["case_ids_sha256"]):
            raise ValueError("Parallel controller identity or cohort mismatch")
        # Preserve completed/failed attempts and service identities on resume.
        previous.update(source_signature=manifest["source_signature"],
                        source_repairs=manifest.get("source_repairs", []))
        atomic_write_json(path, previous)
        return path
    if job["status"] != "pending" or job.get("source") or job.get("position"):
        raise ValueError("Parallel transition requires an untouched method")
    output = Path(job["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Parallel transition cannot change an existing hospital world")
    config = ExperimentConfig.load(job["config"])
    data = config.to_dict()
    ports = PORTS[method]
    data["benchmark"]["service_port"] = ports["hospital_port"]
    data["methods"][0]["providers"]["embedding"]["base_url"] = f"http://127.0.0.1:{ports['embedding_port']}/v1"
    isolated = ExperimentConfig.from_dict(data)
    plan = build_plan(HospitalOptions.from_experiment(isolated), checkpoints=(), seed=0)
    if json_sha256(list(plan.plan.train_ids)) != manifest["case_ids_sha256"]:
        raise ValueError("Parallel transition changed case order")
    checked = ExperimentRunner().check(isolated)
    if not checked["ready"]:
        raise ValueError(f"Parallel {method} preflight failed: {checked}")
    atomic_write_json(run / method / "config-before-parallel.json", config.to_dict())
    atomic_write_json(job["config"], isolated.to_dict())
    job["parallel_services"] = {**ports, "cpu_affinity":affinity}
    controller = copy.deepcopy(manifest)
    controller["settings"].update(ports)
    controller["settings"]["cpu_affinity"] = affinity
    controller["settings"]["parallel_method"] = method
    controller["status"] = "pending"
    atomic_write_json(path, controller)
    atomic_write_json(run / method / "parallel-preflight.json", checked)
    return path


async def execute_parallel(args):
    """Each method owns its interpreter, services, lock, backoff and state file."""
    import psutil
    run = Path(args.run_dir).resolve()
    path = run / "suite.json"
    with SuiteLock(run):
        manifest = read(path)
        if not manifest or manifest["source_signature"] != source_signature() or manifest["model_environment"] != model_environment():
            raise ValueError("Suite sources or model environment changed")
        if any(manifest["jobs"][m]["status"] != "completed" for m in METHODS[:3]):
            raise ValueError("First three methods must complete before parallel final two")
        if any(manifest["jobs"][m].get("checkpoint_cleanup", {}).get("status") != "completed" for m in METHODS[:3]):
            raise ValueError("First three checkpoint cleanups must complete before parallel launch")
        for job in manifest["jobs"].values():
            for key, created in (("pid","created"), ("controller_pid","controller_created")):
                pid = job.get(key)
                if pid and psutil.pid_exists(pid) and psutil.Process(pid).create_time() == job.get(created):
                    raise RuntimeError("A suite worker/controller is still alive; do not duplicate it")
        cpus = psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else []
        halves = [cpus[:len(cpus)//2],cpus[len(cpus)//2:]] if len(cpus)>1 else [cpus,cpus]
        states = {}
        for method, affinity in zip(FINAL_METHODS, halves):
            if manifest["jobs"][method]["status"] != "completed":
                states[method] = prepare_controller(run, manifest, method, affinity)
        stop = run / "STOP"
        stop.unlink(missing_ok=True)
        manifest.update(status="running", owner_pid=os.getpid(), execution_mode="parallel_final_two")
        manifest.setdefault("execution_changes", []).append({
            "at":datetime.now(timezone.utc).isoformat(), "mode":"parallel_final_two",
            "methods":list(FINAL_METHODS), "ports":PORTS,
            "cpu_sets":dict(zip(FINAL_METHODS,halves)),
            "note":"Independent state/services/cache; shared RAM/disk/network/API capacity. Timing belongs to each method; sums are not suite wall time."})
        atomic_write_json(path, manifest)
        processes, logs, exitcodes = {}, [], {}
        try:
            for method in states:
                job = manifest["jobs"][method]
                log = (run / method / "controller.log").open("ab")
                logs.append(log)
                command = [sys.executable,"-u","-X","utf8",str(ENTRY),"_method-owner",
                           "--run-dir",str(run),"--method",method]
                process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                processes[method] = process
                job.update(controller_pid=process.pid, controller_created=psutil.Process(process.pid).create_time())
                atomic_write_json(path, manifest)
            while processes:
                # A reviewed repair can restart only its stopped controller.
                # Running peers keep their process/state/service identities.
                for method in FINAL_METHODS:
                    request = run / method / "restart-request.json"
                    if method in processes or not request.exists():
                        continue
                    repair = read(request)
                    current = source_signature()
                    if not repair or repair.get("source_signature") != current:
                        raise ValueError("Restart request sources do not match the audited repair")
                    manifest.update(source_signature=current, source_repairs=repair["source_repairs"])
                    controller = read(states.get(method, run / method / "controller-suite.json"))
                    if not controller or controller["jobs"][method]["status"] == "completed":
                        raise ValueError("Cannot restart an unknown or completed controller")
                    controller.update(source_signature=current, source_repairs=repair["source_repairs"])
                    state_path = run / method / "controller-suite.json"
                    atomic_write_json(state_path, controller)
                    states[method] = state_path
                    log = (run / method / "controller.log").open("ab")
                    logs.append(log)
                    command = [sys.executable,"-u","-X","utf8",str(ENTRY),"_method-owner",
                               "--run-dir",str(run),"--method",method]
                    process = await asyncio.create_subprocess_exec(*command,stdout=log,stderr=log,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                    processes[method] = process
                    manifest["jobs"][method].update(controller_pid=process.pid,
                        controller_created=psutil.Process(process.pid).create_time())
                    request.rename(request.with_name("restart-accepted-"+str(process.pid)+".json"))
                for method, process in list(processes.items()):
                    state = read(states[method], {})
                    latest = state.get("jobs", {}).get(method)
                    if latest:
                        manifest["jobs"][method].update(latest)
                        if "pid" not in latest:
                            manifest["jobs"][method].pop("pid",None)
                    if process.returncode is not None:
                        exitcodes[method] = await process.wait()
                        # Final state may be committed between the first read
                        # and observing process exit. Re-read after wait.
                        final = read(states[method], {}).get("jobs", {}).get(method)
                        if final:
                            manifest["jobs"][method].update(final)
                            if "pid" not in final:
                                manifest["jobs"][method].pop("pid",None)
                        job = manifest["jobs"][method]
                        job.pop("controller_pid",None)
                        if job["status"] != "completed":
                            job["status"] = "paused"
                            if not job.get("failure"):
                                job["failure"] = {"message":f"Method controller exited {exitcodes[method]}; see controller.log"}
                        del processes[method]
                # A failed method does not cancel the other method's valid work.
                atomic_write_json(path, manifest)
                terminal_status(run, manifest)
                if processes:
                    await asyncio.sleep(5)
            manifest["status"] = "completed" if all(j["status"]=="completed" for j in manifest["jobs"].values()) else "PAUSED"
            atomic_write_json(path, manifest)
            report(run)
            return 0 if manifest["status"]=="completed" else 75
        except BaseException as exc:
            manifest.update(status="PAUSED", failure=error_info(exc))
            atomic_write_json(path, manifest)
            raise
        finally:
            for process in processes.values():
                if process.returncode is None:
                    await terminate_owned_process(process)
            for log in logs:
                log.close()
