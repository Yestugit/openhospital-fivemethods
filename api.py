from __future__ import annotations

from typing import Any

from mematlas.core.registry import PluginRegistry, default_registry
from mematlas.catalog import MethodCatalog


class MemAtlas:
    """Small top-level facade for method discovery and memory creation."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self.registry = registry or default_registry()
        self.catalog = MethodCatalog(self.registry)

    def methods(self) -> list[dict[str, Any]]:
        return [card.to_dict() for card in self.catalog.list()]

    def describe(self, method_id: str) -> dict[str, Any]:
        return self.catalog.get(method_id).to_dict()
