"""Own the local service processes used by one hospital experiment."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.request import urlopen

from .config import HospitalOptions, SERVICE_URLS
from .planning import build_plan


class OpenHospitalServices:
    def __init__(self, experiment):
        self.experiment = experiment
        self.options = HospitalOptions.from_experiment(experiment)
        self._processes = []
        self._logs = []

    async def __aenter__(self):
        # Validate the plan before opening ports or creating service state.
        build_plan(self.options, checkpoints=self.experiment.checkpoints, seed=self.experiment.seed or 0)
        world = Path(self.options.world_dir).expanduser().resolve()
        world.mkdir(parents=True, exist_ok=True)
        try:
            for split in self.options.splits:
                port = int(self.options.service_urls[split].rsplit(":", 1)[1])
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", port))
                    except OSError as exc:
                        raise RuntimeError(f"Hospital port {port} is occupied; close the existing service first") from exc
                log_path = world / f"{split}-service.log"
                log = log_path.open("ab")
                self._logs.append(log)
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(self.options.root / "run_openhospital_local.py"),
                    "--hospital-root", str(self.options.root), "--dataset", split, "--port", str(port), "--world-dir", str(world),
                    stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self._processes.append(process)
                deadline = asyncio.get_running_loop().time() + 60
                while True:
                    if process.returncode is not None:
                        raise RuntimeError(f"Hospital {split} exited; see {log_path}")
                    try:
                        def health():
                            with urlopen(self.options.service_urls[split] + "/health", timeout=1) as response:
                                return json.loads(response.read().decode("utf-8"))
                        info = await asyncio.to_thread(health)
                        if info.get("dataset") == split and info.get("world_id") and info.get("contract"):
                            break
                    except (OSError, ValueError):
                        pass
                    if asyncio.get_running_loop().time() >= deadline:
                        raise RuntimeError(f"Hospital {split} did not become ready; see {log_path}")
                    await asyncio.sleep(0.2)
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *exc):
        for process in reversed(self._processes):
            await terminate_owned_process(process)
        self._processes.clear()
        for log in self._logs:
            log.close()
        self._logs.clear()


async def terminate_owned_process(process):
    """Stop only this owned process and its descendants, including on Windows."""
    if process.returncode is not None:
        return
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        process.kill()
        await process.wait()
    for child in descendants:
        try:
            child.wait(timeout=1)
        except psutil.TimeoutExpired:
            child.kill()
        except psutil.NoSuchProcess:
            pass
