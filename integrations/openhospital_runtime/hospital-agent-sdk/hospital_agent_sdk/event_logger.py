"""JSONL event logger for OpenHospital doctor-agent runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


class EventLogger:
    """Write debug events and final case results to JSONL files."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.output_dir / "events.jsonl"
        self.final_results_path = self.output_dir / "final_results.jsonl"
        self.evaluation_results_path = self.output_dir / "evaluation_results.jsonl"
        self._event_index = 0
        self._latest_final_results_by_patient: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def write_event(
        self,
        event_type: str,
        agent_id: str,
        patient_id: str,
        status: str = "success",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._event_index += 1
        event_payload = dict(payload or {})
        event_payload.setdefault("patient_id", patient_id)
        event = {
            "event_index": self._event_index,
            "timestamp": self._timestamp(),
            "event_type": event_type,
            "agent_id": agent_id,
            "patient_id": patient_id,
            "status": status,
            "payload": event_payload,
        }
        self._append_jsonl(self.events_path, event)
        return event

    def write_final_result(self, result: Dict[str, Any]) -> None:
        patient_id = str(result.get("patient_id") or "").strip()
        record = dict(result)
        if patient_id:
            self._latest_final_results_by_patient[patient_id] = record
        self._append_jsonl(self.final_results_path, record)

    def latest_final_result(self, patient_id: str) -> Dict[str, Any]:
        resolved_patient_id = str(patient_id or "").strip()
        if resolved_patient_id in self._latest_final_results_by_patient:
            return dict(self._latest_final_results_by_patient[resolved_patient_id])

        latest: Dict[str, Any] | None = None
        if self.final_results_path.exists():
            with self.final_results_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and str(row.get("patient_id") or "").strip() == resolved_patient_id:
                        latest = row
        if latest is None:
            raise ValueError("No final_result has been recorded for patient_id=%s." % resolved_patient_id)
        self._latest_final_results_by_patient[resolved_patient_id] = latest
        return dict(latest)

    def write_evaluation_result(self, patient_id: str, report: Dict[str, Any]) -> None:
        record = {
            "timestamp": self._timestamp(),
            "patient_id": patient_id,
            "report": report,
        }
        self._append_jsonl(self.evaluation_results_path, record)

    @staticmethod
    def _append_jsonl(path: Path, data: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
