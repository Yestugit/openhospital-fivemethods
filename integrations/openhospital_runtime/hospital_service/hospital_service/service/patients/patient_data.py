"""Patient profile indexing copied from the standalone patient agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FOUR_HISTORY_KEYS = (
    "present_illness_history",
    "medical_history",
    "personal_history",
    "family_history",
)


@dataclass(frozen=True)
class PatientRecord:
    """Precomputed, read-only lookup data for one patient profile."""

    profile: dict[str, Any]
    history_field_catalog: dict[str, list[str]]
    field_to_history_types: dict[str, list[str]]
    history_entries: list[dict[str, str]]
    history_entries_by_type: dict[str, list[dict[str, str]]]
    history_entries_by_field: dict[str, list[dict[str, str]]]
    chief_complaint_entries: list[dict[str, str]]


def build_patient_record(patient_profile: dict[str, Any]) -> PatientRecord:
    history_entries = flatten_history_entries(patient_profile)
    history_field_catalog = build_history_field_catalog(patient_profile)
    history_entries_by_type = build_history_entries_by_type(history_entries)
    history_entries_by_field = build_history_entries_by_field(history_entries)
    return PatientRecord(
        profile=patient_profile,
        history_field_catalog=history_field_catalog,
        field_to_history_types=build_field_to_history_types(history_field_catalog),
        history_entries=history_entries,
        history_entries_by_type=history_entries_by_type,
        history_entries_by_field=history_entries_by_field,
        chief_complaint_entries=select_history_entries(
            history_entries_by_type=history_entries_by_type,
            history_entries_by_field=history_entries_by_field,
            history_types=["present_illness_history"],
            field_names=["chief_complaint"],
        ),
    )


def build_history_field_catalog(patient_profile: dict[str, Any]) -> dict[str, list[str]]:
    return {
        history_type: collect_leaf_field_names(patient_profile.get(history_type, {}))
        for history_type in FOUR_HISTORY_KEYS
    }


def flatten_history_entries(patient_profile: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    def walk(history_type: str, value: dict[str, Any], prefix: str = "") -> None:
        for key, child in value.items():
            full_key = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(child, dict):
                walk(history_type, child, full_key)
                continue
            value_text = history_value_to_text(child)
            if not value_text:
                continue
            field_path = f"{history_type}.{full_key}"
            entries.append(
                {
                    "history_type": history_type,
                    "field_name": full_key,
                    "field_path": field_path,
                    "content": f"Field: {field_path}\nValue: {value_text}",
                }
            )

    for history_type in FOUR_HISTORY_KEYS:
        history_data = patient_profile.get(history_type, {})
        if isinstance(history_data, dict):
            walk(history_type, history_data)
    return entries


def select_history_entries(
    *,
    history_entries_by_type: dict[str, list[dict[str, str]]],
    history_entries_by_field: dict[str, list[dict[str, str]]],
    history_types: list[str],
    field_names: list[str],
) -> list[dict[str, str]]:
    type_filter = list(dict.fromkeys(item for item in history_types if item in FOUR_HISTORY_KEYS))
    field_filter = list(
        dict.fromkeys(normalize_field_name(item) for item in field_names if str(item).strip())
    )
    if not type_filter and not field_filter:
        return []

    candidates: list[dict[str, str]] = []
    if field_filter:
        for field_name in field_filter:
            candidates.extend(history_entries_by_field.get(field_name, []))
    else:
        for history_type in type_filter:
            candidates.extend(history_entries_by_type.get(history_type, []))

    selected: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for entry in candidates:
        if type_filter and entry["history_type"] not in type_filter:
            continue
        if field_filter and entry["field_name"] not in field_filter:
            continue
        if entry["field_path"] in seen_paths:
            continue
        selected.append(entry)
        seen_paths.add(entry["field_path"])
    return selected


def normalize_field_name(value: Any) -> str:
    field_name = str(value or "").strip().replace("[]", "")
    for history_type in FOUR_HISTORY_KEYS:
        prefix = f"{history_type}."
        if field_name.startswith(prefix):
            field_name = field_name[len(prefix) :]
            break
    return field_name.replace(".", "_")


def collect_leaf_field_names(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return []
    fields: list[str] = []
    for key, child in value.items():
        full_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(child, dict):
            fields.extend(collect_leaf_field_names(child, full_key))
        else:
            fields.append(full_key)
    return fields


def build_field_to_history_types(catalog: dict[str, list[str]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for history_type, field_names in catalog.items():
        for field_name in field_names:
            mapping.setdefault(field_name, [])
            if history_type not in mapping[field_name]:
                mapping[field_name].append(history_type)
    return mapping


def build_history_entries_by_type(
    history_entries: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {history_type: [] for history_type in FOUR_HISTORY_KEYS}
    for entry in history_entries:
        index.setdefault(entry["history_type"], []).append(entry)
    return index


def build_history_entries_by_field(
    history_entries: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for entry in history_entries:
        index.setdefault(entry["field_name"], []).append(entry)
    return index


def history_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()

