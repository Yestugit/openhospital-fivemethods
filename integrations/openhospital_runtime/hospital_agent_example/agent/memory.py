"""Small Markdown memory for evaluation reflections.

The baseline stores one reflection field per patient. Future prompts read the
latest patient reflections as simple reference notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = BASE_DIR / "data" / "memory_data" / "memory.md"


class MarkdownMemory:
    def __init__(
        self,
        path: Union[str, Path],
        *,
        max_notes: int = 3,
        max_note_chars: int = 1200,
    ):
        self.path = Path(path)
        self.max_notes = int(max_notes)
        self.max_note_chars = int(max_note_chars)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Baseline Doctor Memory\n\n", encoding="utf-8")

    def load_notes(self, limit: Optional[int] = None) -> List[str]:
        """读取最近几条评估反思，新的在前。"""
        text = self.path.read_text(encoding="utf-8")
        notes = []
        for block in text.split("\n## "):
            block = block.strip()
            if block and not block.startswith("# Baseline"):
                notes.append(self._truncate("## " + block))
        notes.reverse()
        return notes[: int(limit or self.max_notes)]

    def append_case_reflection(
        self,
        *,
        patient_id: str,
        evaluation_reflection: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        """训练评估后，每个患者只保存一个 reflection 字段。"""
        case_block = self._case_block(patient_id, evaluation_reflection or {})
        self._upsert_case_block(patient_id, case_block)

    def _case_block(self, patient_id: str, reflection: Dict[str, Any]) -> str:
        reflection_value = reflection.get("reflection") if "reflection" in reflection else reflection
        if not isinstance(reflection_value, dict):
            reflection_value = {"summary": reflection_value}

        lines = [
            "## Case %s Reflection" % patient_id,
            "",
            "### Reflection",
            "",
        ]
        for label, key in [
            ("Profile", "profile"),
            ("Diagnosis", "diagnosis_reflection"),
            ("Examination", "examination_reflection"),
            ("Treatment", "treatment_reflection"),
            ("Future Strategy", "future_strategy"),
        ]:
            value = self._truncate(reflection_value.get(key))
            if value:
                lines.append("- **%s:** %s" % (label, value))

        extra_items = [
            (key, value)
            for key, value in reflection_value.items()
            if key
            not in {
                "profile",
                "diagnosis_reflection",
                "examination_reflection",
                "treatment_reflection",
                "future_strategy",
            }
        ]
        for key, value in extra_items:
            lines.append("- **%s:** %s" % (str(key), self._truncate(value)))

        return "\n".join(lines).rstrip() + "\n"

    def _upsert_case_block(self, patient_id: str, new_block: str) -> None:
        text = self.path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        header = parts[0].rstrip() if parts else "# Baseline Doctor Memory"
        case_blocks = []
        for raw_block in parts[1:]:
            block = "## " + raw_block.strip()
            if not self._is_patient_block(block, patient_id):
                case_blocks.append(block)
        body = "\n\n".join([*case_blocks, new_block]).strip()
        self.path.write_text("%s\n\n%s\n" % (header, body), encoding="utf-8")

    def _is_patient_block(self, block: str, patient_id: str) -> bool:
        first_line = block.splitlines()[0].strip() if block.strip() else ""
        return first_line in {
            "## Case %s" % patient_id,
            "## Case %s Reflection" % patient_id,
        }

    def _truncate(self, value: Any, max_chars: Optional[int] = None) -> str:
        text = str(value or "").strip()
        limit = int(max_chars or self.max_note_chars)
        if len(text) <= limit:
            return text
        return text[: limit - 18].rstrip() + "\n...[truncated]"


def build_memory(config: Dict[str, Any]) -> MarkdownMemory:
    memory_config = config.get("memory") if isinstance(config.get("memory"), dict) else {}
    if memory_config.get("provider") == "mematlas" or memory_config.get("method"):
        from mematlas.integrations.openhospital import OpenHospitalMemoryBridge

        method = memory_config.get("method")
        if isinstance(method, str):
            method = {"method_id": method, "params": memory_config.get("params", {})}
        if not isinstance(method, dict):
            raise ValueError("memory.method must be a method ID or MemAtlas method mapping")
        runtime = dict(memory_config.get("runtime", {}))
        runtime.setdefault("artifact_dir", str(BASE_DIR / "data" / "memory_data" / "mematlas"))
        return OpenHospitalMemoryBridge.from_config(
            {"method": method, "runtime": runtime},
            department=str(memory_config.get("department", "心内科")),
            top_k=int(memory_config.get("top_k", 5)),
        )
    configured_path = memory_config.get("md_path") if isinstance(memory_config, dict) else None
    max_notes = memory_config.get("max_notes", 3) if isinstance(memory_config, dict) else 3
    max_note_chars = memory_config.get("max_note_chars", 1200) if isinstance(memory_config, dict) else 1200
    memory_path = Path(configured_path) if configured_path else DEFAULT_MEMORY_PATH
    if not memory_path.is_absolute():
        memory_path = BASE_DIR / memory_path
    return MarkdownMemory(
        memory_path,
        max_notes=int(max_notes),
        max_note_chars=int(max_note_chars),
    )
