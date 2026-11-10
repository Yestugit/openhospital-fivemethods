"""Run the local agent test service and evaluate the generated results."""

from __future__ import annotations

import json
import asyncio
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR.parents[1] / ".env.local", override=False)
LOCAL_SDK_DIR = REPO_DIR / "hospital-agent-sdk"
if LOCAL_SDK_DIR.exists() and str(LOCAL_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SDK_DIR))

from hospital_agent_sdk import build_service_clients, load_config, runtime_config_from_project_config
from hospital_agent_sdk.actions import DoctorActions
from hospital_agent_sdk.event_logger import EventLogger


TEST_URL = "http://127.0.0.1:7860/test"


def main() -> None:
    server = start_agent_server()
    try:
        wait_until_ready()
        test_summary = request_test_run()
        output_dir = Path(str(test_summary["result"]["output_dir"]))
        report = evaluate_test_output(output_dir)
    finally:
        stop_agent_server(server)

    print(json.dumps({"test": test_summary, "evaluation": report}, ensure_ascii=False, indent=2))


def start_agent_server() -> subprocess.Popen[str]:
    env = os.environ.copy()
    if LOCAL_SDK_DIR.exists():
        env["PYTHONPATH"] = "%s%s%s" % (
            LOCAL_SDK_DIR,
            os.pathsep,
            env.get("PYTHONPATH", ""),
        )
    return subprocess.Popen(
        [sys.executable, "-m", "agent.agent"],
        cwd=BASE_DIR,
        env=env,
        text=True,
    )


def wait_until_ready(timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    health_url = "http://127.0.0.1:7860/health"
    while time.time() < deadline:
        try:
            request_json("GET", health_url)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Agent service did not become ready on http://127.0.0.1:7860.")


def request_test_run() -> dict[str, Any]:
    return request_json("POST", TEST_URL)


def evaluate_test_output(output_dir: Path) -> dict[str, Any]:
    config = load_config(BASE_DIR / "config.yaml")
    runtime_config = runtime_config_from_project_config(config, {"local_test": True})
    clients = build_service_clients(
        base_url=runtime_config.service_base_url,
        token=runtime_config.service_token,
        model_api_key=runtime_config.model_api_key,
        team_id=runtime_config.team_id,
        mode="train",
    )
    actions = DoctorActions(
        patient_client=clients.patient_client,
        exam_client=clients.exam_client,
        evaluate_client=clients.evaluate_client,
        event_logger=EventLogger(output_dir=output_dir),
        team_id=runtime_config.team_id,
    )
    return asyncio.run(actions.batch_evaluation(output_dir))


def request_json(method: str, url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=b"" if method == "POST" else None,
        headers={"Accept": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON: {payload!r}")
    return payload


def stop_agent_server(server: subprocess.Popen[str]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
