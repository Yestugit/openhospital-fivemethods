"""Observable encounter inputs and ordered method-facing events."""
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from mematlas.core.models import EventKind, MemoryEvent, Scope
from .context import bounded_text


def text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def task_description(department: str, state: Mapping[str, Any]) -> str:
    # Only information available to the doctor; final diagnoses/feedback are excluded.
    patient_turns = [str(t.get("text", t.get("content", ""))) for t in state.get("chat_history", ())
                     if str(t.get("from", t.get("role", "user"))).lower() in {"patient", "user", "human"}]
    return f"诊疗任务（{department}）\n患者描述：" + bounded_text("\n".join(patient_turns), 6000) + (
        "\n检查信息：" + bounded_text(text(state["examination_results"]), 4000) if state.get("examination_results") else "")


def encounter_events(scope: Scope, occurrence: str, patient: str, department: str,
                     state: Mapping[str, Any], result: Mapping[str, Any], feedback: Mapping[str, Any]):
    rows = list(state.get("timeline", ()))
    if not rows:
        # Agents supplying only chat history require an explicitly marked reconstruction.
        rows = [{"kind": "message", "role": "assistant" if t.get("from", t.get("role")) in {"doctor", "assistant"} else "user",
                 "text": t.get("text", t.get("content", ""))} for t in state.get("chat_history", ())]
        if state.get("examination_results"):
            rows.append({"kind": "tool_result", "text": text(state["examination_results"])})
    if result and not any(row.get("text") == text(result) for row in rows):
        rows.append({"kind": "action", "role": "assistant", "text": text(result)})
    if feedback:
        rows.append({"kind": "feedback", "role": "assistant", "text": text(feedback)})
    # Bridges persist this timestamp in the encounter journal before learning.
    base = state.get("encounter_timestamp")
    base = datetime.fromisoformat(base) if isinstance(base, str) else base
    base = base or datetime(1970, 1, 1, tzinfo=timezone.utc)
    return tuple(MemoryEvent(
        kind=EventKind(row["kind"]), scope=scope,
        content=str(row.get("text", "")),
        event_id=f"{occurrence}:step:{i}", role=row.get("role"), episode_id=occurrence,
        timestamp=row.get("timestamp") or base + timedelta(microseconds=i),
        extensions={"patient_id": patient, "department": department, "episode_id": occurrence,
                  "timeline_order": i, "reconstructed": not bool(state.get("timeline"))},
    ) for i, row in enumerate(rows))
