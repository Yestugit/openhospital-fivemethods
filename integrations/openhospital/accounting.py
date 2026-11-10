"""Import hospital provider attempts and doctor calls into the experiment ledger."""
from __future__ import annotations

import json
from typing import Mapping
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from mematlas.core.telemetry import UsageLedger, UsageRecord, usage_scope, usage_context, current_usage_context


class HospitalUsageCollector:
    def __init__(self, service_url: str, team_id: str, ledger: UsageLedger):
        self.service_url, self.team_id, self.ledger = service_url.rstrip("/"), team_id, ledger
        self._ids: list[str] = []
        self.world_id = None
        self._rows = []

    def sync(self) -> None:
        with urlopen(f"{self.service_url}/usage?{urlencode({'team_id': self.team_id})}", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("usage_schema_version") != 2 or not isinstance(payload.get("usage_records"), list):
            raise ValueError("Hospital service requires detailed usage_schema_version=2; aggregate counters cannot prove complete costs")
        rows = payload["usage_records"]
        world_id = payload.get("world_id")
        if self.world_id is not None and world_id != self.world_id:
            raise ValueError("Hospital accounting world changed")
        if rows[:len(self._rows)] != self._rows:
            raise ValueError("Hospital usage history changed or reset")
        ids = [row["record_id"] for row in rows]
        if ids[:len(self._ids)] != self._ids or len(set(ids)) != len(ids):
            raise ValueError("Hospital usage history reset or contains duplicate calls")
        known_ids = {row.record_id for row in self.ledger.all()}
        for row in rows[len(self._ids):]:
            if row["record_id"] in known_ids:
                continue
            data = {**current_usage_context(), **row}
            self.ledger.add(UsageRecord(**data), imported=True)
        self._ids = ids
        self._rows = rows
        self.world_id = world_id


def doctor_usage_observer(ledger: UsageLedger):
    def observe(event: Mapping):
        # Service-side provider attempts are authoritative for proxy calls.
        if event["origin"] == "service_proxy":
            return
        usage = event.get("usage")
        known = isinstance(usage, Mapping) and all(isinstance(usage.get(key), int)
            and not isinstance(usage[key], bool) and usage[key] >= 0 for key in ("prompt_tokens", "completion_tokens"))
        details = usage.get("prompt_tokens_details") if isinstance(usage, Mapping) else None
        metadata = {"attempt": event["attempt"], "origin": "doctor_direct",
            **{key:event[key] for key in ("request_id", "elapsed_s", "http_status") if key in event}}
        if isinstance(details, Mapping):
            metadata["prompt_tokens_details"] = dict(details)
        with usage_scope(ledger, "doctor"), usage_context(operation="doctor_model_call",
            case_id=current_usage_context().get("case_id") or event.get("case_id")):
            ledger.add(UsageRecord(
                provider=urlparse(event["base_url"]).netloc, model=event["model"],
                prompt_tokens=usage["prompt_tokens"] if known else 0,
                completion_tokens=usage["completion_tokens"] if known else 0,
                usage_known=known, status=event["status"], metadata=metadata,
            ))
    return observe


def hospital_window_metrics(ledger: UsageLedger, cursor: int) -> dict:
    doctor = ledger.summary("doctor", since=cursor)
    service = ledger.summary(("patient", "examination", "judge"), since=cursor)
    return {
        "doctor_llm_calls": doctor["tracked_calls"], "doctor_prompt_tokens": doctor["prompt_tokens"],
        "doctor_completion_tokens": doctor["completion_tokens"], "doctor_total_tokens": doctor["total_tokens"],
        "doctor_cost": doctor["cost"], "service_requests": service["tracked_calls"],
        "service_input_tokens": service["prompt_tokens"], "service_output_tokens": service["completion_tokens"],
        "service_total_tokens": service["total_tokens"], "service_cost": service["cost"],
        "service_cost_scope": "patient_examination_judge_excludes_doctor_proxy",
        "usage_coverage": "complete" if doctor["token_coverage"] == service["token_coverage"] == "complete" else "partial",
    }
