"""Explicit test services: no network, model downloads or quality claims."""
from __future__ import annotations

import json
import re
from typing import Any

from mematlas.core.plugin import MethodServices


class FixedVectors:
    """Small hand-selected vectors; these are not semantic model embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0, 0.0] if "unrelated" in text else [1.0, 0.0, 0.0] for text in texts]


class FixedDecisions:
    """Return controlled analysis/evolution responses to exercise real storage."""

    def __init__(self, *, evolve: bool = False, malformed: bool = False):
        self.evolve, self.malformed = evolve, malformed

    async def complete(self, messages: list[dict[str, Any]], **options: Any) -> str:
        prompt = messages[-1]["content"]
        if self.malformed:
            return "not json"
        if "should_evolve" not in prompt:
            return json.dumps({"keywords": ["Helios", "port"], "context": "Service configuration", "tags": ["config"]})
        ids = re.findall(r"memory id:([^\s]+)", prompt)
        # Accommodate the pre-migration prompt so core regressions fail on behavior.
        if not ids:
            ids = re.findall(r"'id': '([^']+)'", prompt.split("Neighbors:")[-1])
        return json.dumps({
            "should_evolve": self.evolve,
            "actions": ["strengthen", "update_neighbor"] if self.evolve else [],
            "suggested_connections": ids,
            "tags_to_update": ["config", "updated"],
            "new_context_neighborhood": ["Updated service configuration"] * len(ids),
            "new_tags_neighborhood": [["updated"] for _ in ids],
        })


def services(*, evolve: bool = False, malformed: bool = False) -> MethodServices:
    return MethodServices(llm=FixedDecisions(evolve=evolve, malformed=malformed), embedder=FixedVectors())


def config(directory, **params):
    return {"method": {"method_id": "a-mem", "params": params},
            "scope": {"user_id": "alice"}, "runtime": {"artifact_dir": str(directory)}}
