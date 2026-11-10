"""Read-only experiment views: concise PowerShell screen and local web dashboard."""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import shutil
import sys
import time

from mematlas.core.activity import summarize

METHODS = ("memos", "mem0", "memmachine", "emem", "memora")
NAMES = dict(zip(METHODS, ("MemOS", "Mem0", "MemMachine", "EMem", "Memora")))


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def collect(run, manifest=None):
    """Read saved results and activity only; never inspect or mutate a live backend."""
    run = Path(run).resolve()
    manifest = manifest or read(run / "suite.json", {})
    replacement = read(run / "serial-replacement.json", {})
    if replacement.get("status") in {"waiting", "validating", "validation_paused", "deferred"}:
        # A retired run must not appear as completed progress for its replacement.
        manifest = {**manifest, "jobs": dict(manifest.get("jobs", {})), "execution_mode": "serial"}
        method = replacement["method"]
        manifest["jobs"][method] = {**replacement["job"],
            "status":"deferred" if replacement["status"] == "deferred" else "queued"}
    total = 2 if manifest.get("settings", {}).get("smoke") else 100
    rounds, size = (2, 1) if total == 2 else (5, 20)
    live = [m for m in METHODS if manifest.get("jobs", {}).get(m, {}).get("status") in {"running", "retrying"}]
    current = live[0] if live else next((m for m in METHODS if manifest.get("jobs", {}).get(m, {}).get("status") != "completed"), METHODS[-1])
    if not live and manifest.get("status") == "stopped":
        current = next((m for m in reversed(METHODS)
                        if manifest.get("jobs", {}).get(m, {}).get("status") == "completed"), current)
    methods, points, growth = [], [], []
    for method in METHODS:
        job = manifest.get("jobs", {}).get(method, {})
        output = Path(job.get("output", run / method / "pending"))
        progress = read(output / "progress.json", {})
        folder = run / method / "activity"
        measurements = [read(p) for p in folder.glob("measurement-*.json")]
        measurements = sorted((p for p in measurements if p), key=lambda p: p["round"])
        latest = measurements[-1] if measurements else {}
        from .efficiency_metrics import case_measurements
        checkpoint = read(output / "runs" / f"001-{method}" / "latest_snapshot.json", {})
        committed = max(job.get("position",0), checkpoint.get("position",0), progress.get("latest_snapshot",{}).get("position",0))
        samples = case_measurements(run / method, committed)
        if samples:
            latest = samples[-1]
        growth.extend({"method":method, **{k:p.get(k) for k in ("case_position","round","rsi_tokens","payload_tokens",
            "memory_text_tokens","storage_bytes","storage_growth_bytes","background_pending","phase")}} for p in samples)
        context = read(folder / "context.json", {})
        index = int(context.get("round", 1))
        completed = total if job.get("status") == "completed" else max(job.get("position", 0), progress.get("training_completed", 0))
        curve = read(run / method / "clinical-rounds.json") if job.get("status") == "completed" else None
        if curve is None:
            curve = read(output / "runs" / f"001-{method}" / "curve.json", [])
        by_round = {p["batch_index"]: p for p in curve if "batch_index" in p}
        for measurement in measurements:
            r = measurement["round"]
            if r:
                score = by_round.get(r, {})
                points.append({"method": method, "round": r,
                    **{k: measurement.get(k) for k in ("round_total_s", "memory_method_wall_s", "memory_persistence_s", "rsi_tokens", "payload_tokens", "memory_text_tokens", "storage_bytes", "memory_s_per_1k_rsi_tokens")},
                    **{k: score.get(k) for k in ("diagnosis_accuracy", "treatment_overall_score", "examination_precision", "evaluation_coverage")}})
        timing = summarize(folder, index) if method == current or method in live else {}
        method_time = summarize(folder)
        usage = progress.get("usage", {})
        row = {"id": method, "name": NAMES[method], "status": job.get("status", "pending"),
            "completed": completed, "total": total, "round": index, "rounds": rounds,
            "round_completed": max(0, min(size, completed - (index - 1) * size)), "round_size": size,
            "phase": progress.get("phase", "startup"), "patient": progress.get("patient_id"),
            "attempt": job.get("attempt", 1), "heartbeat": progress.get("updated_at"),
            "scores": progress.get("training_metrics", {}), "round_scores": progress.get("round_metrics", {}),
            "timing": timing, "totals": method_time, "measurement": {k:latest.get(k) for k in ("round", "case_position", "phase", "background_pending", "rsi_tokens", "payload_tokens", "memory_text_tokens", "storage_bytes")},
            "usage": {k:usage.get(k) for k in ("llm_calls", "embedding_calls", "failed_calls", "total_tokens", "known_total_tokens", "tracked_calls")},
            "json": progress.get("json_recovery", {}), "checkpoint": progress.get("latest_snapshot", {}).get("position"),
            "failure": job.get("failure", {}).get("message"),
            "retry_in_s": max(0, math.ceil(job.get("retry_at", 0) - time.time()))}
        methods.append(row)
    active = next(row for row in methods if row["id"] == current)
    # Aggregate cumulative time across all segments and methods; pauses have no round span.
    totals = {key: 0.0 for key in ("round_total_s", "memory_method_wall_s", "memory_persistence_s")}
    for method in methods:
        values = method["totals"]
        for key in totals:
            totals[key] += values.get(key, 0)
    return {"status": manifest.get("status", "initializing"), "run_name": run.name,
        "serial_replacement": {k:replacement.get(k) for k in ("method", "status", "waiting_for")},
        "updated_at": datetime.now(timezone.utc).isoformat(), "completed": sum(row["completed"] for row in methods),
        "total": total * 5, "active": active, "active_methods":[row for row in methods if row["id"] in live],
        "execution_mode":manifest.get("execution_mode", "serial"), "methods": methods, "rounds": points, "growth":growth, "totals": totals,
        "stop_requested": (run / "STOP").exists(), "failure": manifest.get("failure", {}).get("message")}


def duration(value):
    if value is None:
        return "--"
    seconds = max(0, int(value))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


def number(value, digits=0):
    return "--" if value is None else f"{value:,.{digits}f}"


def percent(value):
    return "--" if value is None else f"{value*100:.1f}%"


def render(state):
    a, total = state["active"], state["total"]
    n = state["completed"]
    filled = min(28, int(28*n/total))
    bar = "#"*filled + "-"*(28-filled)
    status = state["status"].upper()
    if state["stop_requested"] and status == "RUNNING":
        status = "STOPPING AT CASE BOUNDARY"
    local = datetime.fromisoformat(state["updated_at"]).astimezone().strftime("%H:%M:%S")
    lines = [f"OPENHOSPITAL | Hematology RSI | {status}",
        f"Overall [{bar}] {n}/{total} ({100*n/total:.1f}%)   Updated {local}",
        "Method       State          Cases     Diagnosis   Treatment   Coverage",
        "-"*76]
    for m in state["methods"]:
        mark = ">" if m["id"] == a["id"] and m["status"] != "completed" else " "
        s = m["scores"]
        coverage = s.get('evaluation_coverage') if s.get('evaluated_cases',0) else None
        lines.append(f"{mark}{m['name']:<11} {m['status']:<13} {str(m['completed'])+'/'+str(m['total']):<9} {percent(s.get('diagnosis_accuracy')):>9} {number(s.get('treatment_overall_score'),3):>11} {percent(coverage):>10}")
    lines.extend(["-"*76,
        f"NOW {a['name']} | Round {a['round']}/{a['rounds']} ({a['round_completed']}/{a['round_size']} done) | {a['patient'] or 'Preparing services'}",
        f"Stage {a['phase']} | Attempt {a['attempt']} | Saved checkpoint: case {a['checkpoint'] if a['checkpoint'] is not None else '--'}",
        "                                  THIS ROUND      ALL METHODS",
        f"Total elapsed                     {duration(a['timing'].get('round_total_s')):>10}       {duration(state['totals']['round_total_s']):>10}",
        f"MEMORY method time                {duration(a['timing'].get('memory_method_wall_s')):>10}       {duration(state['totals']['memory_method_wall_s']):>10}",
        f"Checkpoint / state protection     {duration(a['timing'].get('memory_persistence_s')):>10}       {duration(state['totals']['memory_persistence_s']):>10}",
        f"Memory foreground {duration(a['timing'].get('memory_foreground_wait_s'))} | Background {duration(a['timing'].get('memory_background_active_s'))}"])
    if state.get("execution_mode") == "parallel_final_two":
        lines.append("PARALLEL: " + " | ".join(f"{m['name']} {m['completed']}/{m['total']} {m['status']}" for m in state["methods"] if m['id'] in {"emem","memora"}))
    if state.get("serial_replacement", {}).get("status") in {"waiting", "validating", "validation_paused"}:
        lines.append("SERIAL: Memora first; fresh EMem starts only after Memora completes and exits.")
    elif state.get("serial_replacement", {}).get("status") == "deferred":
        lines.append("EMem DEFERRED by user. Stop after Memora; no automatic EMem start.")
    measured = a["measurement"]
    r, position = measured.get("round"), measured.get("case_position")
    if position is not None:
        lines.append(f"Memory size / RSI tokens: after case {position}/{a['total']} | Background pending: {measured.get('background_pending','--')}")
    else:
        label = "empty baseline" if r == 0 else f"legacy end of round {r}" if r is not None else "not measured yet"
        lines.append(f"Memory size / RSI tokens: {label} | new runs sample EVERY CASE")
    lines.extend([f"RSI tokens {number(measured.get('rsi_tokens'))} | Payload {number(measured.get('payload_tokens'))} | Memory text {number(measured.get('memory_text_tokens'))}",
        f"Storage {number(measured['storage_bytes']/1048576,2) if measured.get('storage_bytes') is not None else '--'} MiB | Round diagnosis {percent(a['round_scores'].get('diagnosis_accuracy'))} | Treatment {number(a['round_scores'].get('treatment_overall_score'),3)}",
        f"LLM {number(a['usage'].get('llm_calls'))} | Embedding {number(a['usage'].get('embedding_calls'))} | Failed requests {number(a['usage'].get('failed_calls'))} | JSON repairs {a['json'].get('repairs',0)} / fallback {a['json'].get('fallbacks',0)}"])
    reason = a["failure"] or state.get("failure")
    if reason:
        lines.append(f"Attention: {reason} | Retry in {a['retry_in_s']}s")
    lines.extend(["Treatment: 0-1, higher is better. -- = not available. Time = HH:MM:SS.",
        "Ctrl+C closes this viewer only; the experiment continues in background."])
    return "\n".join(lines)


class Console:
    """Repaint a TTY in place; redirected logs remain plain append-only text."""
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.live = self.stream.isatty()
        if self.live and sys.platform == "win32":
            import ctypes
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                self.live = bool(ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 4))
            else:
                self.live = False

    def show(self, text):
        if self.live:
            width, height = shutil.get_terminal_size((100, 35))
            text = "\n".join(line[:max(1,width-1)] for line in text.splitlines()[:max(1,height-1)])
            self.stream.write("\x1b[H\x1b[2J" + text + "\n")
        else:
            self.stream.write(text + "\n\n")
        self.stream.flush()


def serve(run, port):
    """Serve a single dashboard on loopback; no file browsing or control endpoints."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Lock
    run = Path(run).resolve()
    if not (run / "suite.json").is_file():
        raise ValueError("Unknown suite; start the experiment first")
    html = (Path(__file__).with_name("dashboard.html")).read_bytes()
    lock, cached = Lock(), {"at":0, "data":None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/":
                data, kind = html, "text/html; charset=utf-8"
            elif self.path.split("?")[0] == "/api/status":
                with lock:
                    if time.monotonic()-cached["at"] > 3:
                        try:
                            cached.update(at=time.monotonic(), data=collect(run))
                        except Exception as exc:
                            if cached["data"] is None:
                                self.send_error(503, "Progress temporarily unavailable; experiment is independent of this viewer")
                                return
                            cached["data"]["observer_warning"] = f"Showing last valid metrics: {type(exc).__name__}: {exc}"
                            cached["at"] = time.monotonic()
                    data = json.dumps(cached["data"],ensure_ascii=False).encode("utf-8")
                kind = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionError):
                pass  # Closing a viewer does not affect the experiment.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Read-only dashboard: http://127.0.0.1:{port} | {run.name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
