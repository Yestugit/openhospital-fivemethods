"""Memory ownership uses one mapping in configuration and saved data."""
from __future__ import annotations

import json
from typing import Mapping

SCOPE_FIELDS = frozenset({"user_id", "agent_id", "team_id", "session_id", "run_id"})


def normalize_scope(scope: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(scope, Mapping):
        raise TypeError("scope must be a mapping of identity fields")
    if set(scope) - SCOPE_FIELDS:
        raise ValueError(f"Unknown scope fields: {sorted(set(scope) - SCOPE_FIELDS)}")
    if any(not isinstance(value, str) or not value.strip() for value in scope.values()):
        raise ValueError("scope values must be nonempty strings")
    return dict(scope)


def scope_key(scope: Mapping[str, str]) -> str:
    return json.dumps(normalize_scope(scope), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
