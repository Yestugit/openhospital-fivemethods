"""Canonical examination names and historical request aliases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ExaminationNames:
    def __init__(self, aliases: dict[str, str], canonical_names: set[str]):
        self._aliases = dict(aliases)
        self.canonical_names = frozenset(canonical_names)

    @classmethod
    def load(cls, path: str | Path) -> "ExaminationNames":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Examination alias catalog must be a JSON object")
        rows = payload.get("examinations")
        if not isinstance(rows, list):
            raise ValueError("Examination alias catalog must contain an examinations array")

        aliases: dict[str, str] = {}
        canonical_names: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Examination alias entry {index} must be an object")
            name = _required_text(row.get("name"), f"examination alias entry {index}")
            if name in canonical_names:
                raise ValueError(f"Duplicate canonical examination name: {name}")
            canonical_names.add(name)
            values = row.get("aliases", [])
            if not isinstance(values, list):
                raise ValueError(f"Aliases for examination {name} must be an array")
            for alias in (name, *_text_values(values)):
                existing = aliases.get(alias)
                if existing is not None and existing != name:
                    raise ValueError(
                        f"Ambiguous examination alias {alias!r}: {existing!r} and {name!r}"
                    )
                aliases[alias] = name
        return cls(aliases, canonical_names)

    def canonical(self, name: str) -> str:
        text = str(name).strip()
        return self._aliases.get(text, text)

    def canonicalize_many(self, names: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for raw_name in names:
            name = self.canonical(raw_name)
            if name and name not in seen:
                result.append(name)
                seen.add(name)
        return result


def _required_text(value: Any, context: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Missing name in {context}")
    return text


def _text_values(values: list[Any]) -> list[str]:
    return [text for value in values if (text := str(value).strip())]
