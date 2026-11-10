"""Small atomic status files; English terminal presentation is owned by the suite."""
from datetime import datetime, timezone
import os
import json
from pathlib import Path
from time import monotonic

from mematlas.core.files import atomic_write_json


class HospitalPaused(RuntimeError):
    """A requested stop reached a durable case boundary."""


class HospitalProgress:
    def __init__(self, output, plan):
        self.path = Path(output) / "progress.json"
        self.plan = plan
        self.started = monotonic()
        self.state = {}
        self.training = {}
        self.tests = {}
        self.last_test = {}
        self.bridge = None
        self.checkpoints = {}
        self.case_identity = None
        self.case_started = None
        self.case_started_at = None

    def write(self, **changes):
        self.state.update(changes)
        identity = tuple(self.state.get(key) for key in ("phase", "patient_id", "position", "checkpoint"))
        if self.state.get("patient_id") and identity != self.case_identity:
            self.case_identity, self.case_started = identity, monotonic()
            self.case_started_at = datetime.now(timezone.utc).isoformat()
        bridge = self.bridge
        memory = {}
        if bridge is not None:
            for instance in getattr(bridge.core_memory._method, "instances", {}).values():
                for key, value in (("short_term_pairs", len(instance.short_term_memory.get_all())),
                                   ("mid_term_pages", len(instance.mid_term_pages)),
                                   ("user_knowledge", len(instance.user_knowledge)),
                                   ("assistant_knowledge", len(instance.assistant_knowledge))):
                    memory[key] = memory.get(key, 0) + value
        if bridge is not None:
            native = getattr(bridge.core_memory._method, "memory", None)
            if native is not None and hasattr(native, "memories"):
                memory["memories"] = len(native.memories)
        usage = bridge.core_memory.services.usage.summary() if bridge else {}
        breakdown = {}
        if bridge:
            ledger = bridge.core_memory.services.usage
            for label, phases, kind in (("Doctor", "doctor", "llm"), ("Memory LLM", "memory", "llm"),
                    ("Patient simulation", "patient", "llm"), ("Examination simulation", "examination", "llm"),
                    ("Scoring", "judge", "llm"), ("Embedding", None, "embedding")):
                row = ledger.summary(phases, resource_kind=kind)
                breakdown[label] = {key: row[key] for key in ("tracked_calls", "prompt_tokens", "completion_tokens",
                    "failed_calls", "token_coverage", "untracked_calls", "known_cached_prompt_tokens")}
        times = [row["response_time_s"] for row in sorted(self.training.values(), key=lambda row: row.get("position", 0))
                 if row.get("response_time_s") is not None]
        recoveries = [row.get("json_recovery", {}) for row in (*self.training.values(), *self.tests.values())]
        latest = {}
        if bridge:
            path = self.path.parent / "runs" / f"001-{bridge.core_memory.method_id}" / "latest_snapshot.json"
            if path.is_file():
                latest = json.loads(path.read_text(encoding="utf-8"))
        current_batch = int(self.state.get("batch", 1))
        boundaries = self.plan.batches()
        lower = sum(len(batch) for batch in boundaries[:current_batch-1])
        upper = lower + (len(boundaries[current_batch-1]) if current_batch <= len(boundaries) else 0)
        round_metrics = self.metrics(row for row in self.training.values() if lower < row.get("position",0) <= upper)
        atomic_write_json(self.path, {**self.state, "round_metrics": round_metrics, "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": monotonic() - self.started, "training_total": self.plan.training_count,
            "training_completed": len(self.training), "test_completed": len(self.tests),
            "test_total": len(self.plan.primary_test_ids) * len(self.plan.effective_checkpoints()) + len(self.plan.advanced_test_ids),
            "training_metrics": self.metrics(self.training.values()), "test_metrics": self.last_test,
            "memory": memory, "usage": usage, "latest_snapshot": latest,
            "worker_pid": os.getpid(), "case_started_at": self.case_started_at,
            "case_elapsed_s": monotonic() - self.case_started if self.case_started is not None else None,
            "usage_breakdown": breakdown, "test_checkpoints": list(self.checkpoints.values()),
            "timing": {"training_samples": len(times), "mean_training_s": sum(times) / len(times) if times else None,
                "mean_last_5_training_s": sum(times[-5:]) / len(times[-5:]) if times else None},
            "json_recovery": {"repairs": sum(row.get("repairs", 0) for row in recoveries),
                "fallbacks": sum(row.get("fallbacks", 0) for row in recoveries)},
            "learning": bridge.history.counts() if bridge else {}})

    def record_checkpoint(self, learned, split, result):
        self.checkpoints[f"{learned}-{split}"] = {"checkpoint": learned, "split": split,
            **{key: value for key, value in result.items() if key not in {"case_reports", "resource_metrics", "window_usage"}}}
        self.last_test = max(self.checkpoints.values(), key=lambda row: (row["checkpoint"], row["split"] == "advanced_test"))

    @staticmethod
    def metrics(rows):
        from .metrics import aggregate_openhospital_evaluations
        return aggregate_openhospital_evaluations(rows)

    def check_stop(self, position):
        stop = os.environ.get("OPH_STOP_FILE")
        pause_after = int(os.environ.get("OPH_PAUSE_AFTER", "0"))
        if stop and Path(stop).exists() or pause_after and position >= pause_after:
            self.write(status="paused")
            raise HospitalPaused("Stopped at a durable case boundary; resume the suite to continue")

    def check_limits(self):
        """Operational limits apply after complete batches/checkpoints are sealed."""
        train = int(os.environ.get("OPH_STOP_TRAIN_AFTER", "0"))
        tests = int(os.environ.get("OPH_STOP_TEST_AFTER", "0"))
        if train and len(self.training) >= train and len(self.tests) >= tests:
            reason = f"Stop target reached: training {len(self.training)}/{train}, held-out tests {len(self.tests)}/{tests}"
            self.write(status="paused", pause_reason=reason)
            raise HospitalPaused(reason)
