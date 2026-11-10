from __future__ import annotations

import importlib
import json
from importlib import metadata
from pathlib import Path
from typing import Iterable

from mematlas.core.errors import PluginNotFoundError
from mematlas.core.plugin import MethodPlugin
from mematlas.core.capabilities import MethodCapabilities, MethodDescriptor, Operation
from mematlas.core.errors import ConfigurationError


class ManifestPlugin(MethodPlugin):
    """Read catalog metadata without importing an algorithm or its dependencies."""

    def __init__(self, directory: Path, module: str):
        directory = Path(directory)
        path = directory / "method.yaml"
        text = path.read_text(encoding="utf-8-sig")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import yaml
            data = yaml.safe_load(text)
        self.module = module
        self.interface_version = data.get("interface_version", 1)
        # Unmigrated descriptors remain discoverable, but can never execute.
        # They do not receive translations to the new algorithm interface.
        data["interface_version"] = self.interface_version
        if self.interface_version != 3:
            supported = {row.value for row in Operation}
            data["capabilities"]["operations"] = [row for row in data["capabilities"]["operations"] if row in supported]
            data["capabilities"]["read_only_mode"] = "unsupported"
        self.descriptor = MethodDescriptor.from_dict(data)

    def _implementation(self):
        if self.interface_version != 3:
            raise ConfigurationError(f"{self.descriptor.method_id}: method requires migration to Memory interface v3")
        plugin = importlib.import_module(self.module).PLUGIN
        if plugin.descriptor.to_dict() != self.descriptor.to_dict():
            raise ConfigurationError("Imported method differs from its manifest")
        return plugin

    def create_method(self, config, services):
        return self._implementation().create_method(config, services)

    @property
    def qa_factory(self):
        # Catalog discovery stays lazy; evaluation resolves the method adapter.
        return self._implementation().qa_factory if self.interface_version == 3 else None

    def healthcheck(self, config, services):
        if self.interface_version != 3:
            return {"ready": False, "status": "migration_required", "interface_version": self.interface_version,
                    "reason": "Method implementation and manifest require Memory interface v3 migration"}
        return self._implementation().healthcheck(config, services)

    def construction_spec(self):
        return {"factory": "mematlas.core.registry:ManifestPlugin",
                "kwargs": {"directory": str(Path(__file__).parents[1] / "methods" / self.module.rsplit(".", 1)[-1]),
                           "module": self.module}}


class UnavailablePlugin(MethodPlugin):
    """Catalog entry for an implementation deferred to the integration stage."""

    def __init__(self, method_id):
        self.descriptor = MethodDescriptor(method_id, method_id, "Control baseline awaiting interface v3 integration",
            MethodCapabilities(frozenset(), frozenset(), frozenset(), frozenset()), interface_version=1)

    def create_method(self, config, services):
        raise ConfigurationError(f"{self.descriptor.method_id}: method requires migration to Memory interface v3")

    def healthcheck(self, config, services):
        return {"ready": False, "status": "migration_required", "reason": "Control baseline implementation requires interface v3 migration"}


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, MethodPlugin] = {}
        self._builtins_loaded = False
        self._discovery_errors: dict[str, str] = {}
        self._loaded_entry_points: set[tuple[str, str]] = set()

    def register(self, plugin: MethodPlugin, *, replace: bool = False) -> None:
        method_id = plugin.descriptor.method_id
        errors = plugin.descriptor.validate()
        if errors:
            raise ValueError("; ".join(errors))
        if method_id in self._plugins and not replace:
            raise ValueError(f"Plugin already registered: {method_id}")
        self._plugins[method_id] = plugin

    def register_module(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            raise ValueError(f"{module_name} does not expose PLUGIN")
        self.register(plugin)

    def load_builtins(self) -> None:
        if self._builtins_loaded:
            return
        from mematlas.methods.builtins import BUILTIN_PLUGIN_MODULES, CONTROL_METHOD_IDS
        for module_name in BUILTIN_PLUGIN_MODULES:
            directory = Path(__file__).parents[1] / "methods" / module_name.rsplit(".", 1)[-1]
            if (directory / "method.yaml").exists():
                self.register(ManifestPlugin(directory, module_name))
        # Control implementations share one module and have no manifests yet.
        # Leave that module untouched until the method migration stage.
        for method_id in CONTROL_METHOD_IDS:
            self.register(UnavailablePlugin(method_id))
        self._builtins_loaded = True

    def discover_entry_points(self, group: str = "mematlas.methods", *, strict: bool = False) -> int:
        self.load_builtins()
        count = 0
        entry_points = metadata.entry_points()
        selected = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, [])
        for entry_point in selected:
            name = str(getattr(entry_point, "name", repr(entry_point)))
            key = (group, name)
            if key in self._loaded_entry_points:
                continue
            try:
                plugin = entry_point.load()
                if isinstance(plugin, type):
                    plugin = plugin()
                self.register(plugin)
            except Exception as exc:
                self._discovery_errors[name] = f"{type(exc).__name__}: {exc}"
                if strict:
                    raise
            else:
                self._discovery_errors.pop(name, None)
                self._loaded_entry_points.add(key)
                count += 1
        return count

    @property
    def discovery_errors(self) -> dict[str, str]:
        return dict(self._discovery_errors)

    def get(self, method_id: str) -> MethodPlugin:
        try:
            return self._plugins[method_id]
        except KeyError as exc:
            raise PluginNotFoundError(
                f"Unknown method {method_id!r}; available: {', '.join(self.ids())}"
            ) from exc

    def ids(self) -> list[str]:
        return sorted(self._plugins)

    def plugins(self) -> Iterable[MethodPlugin]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))

    def describe(self, method_id: str | None = None):
        if method_id:
            return self.get(method_id).descriptor.to_dict()
        return [plugin.descriptor.to_dict() for plugin in self.plugins()]

    def descriptions(self):
        """Return the lightweight method descriptions supplied to an optional LLM router."""
        return [
            {
                "method_id": plugin.descriptor.method_id,
                "name": plugin.descriptor.name,
                "description": plugin.descriptor.summary,
            }
            for plugin in self.plugins()
        ]


_DEFAULT_REGISTRY: PluginRegistry | None = None


def default_registry() -> PluginRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PluginRegistry()
        _DEFAULT_REGISTRY.load_builtins()
        _DEFAULT_REGISTRY.discover_entry_points()
    return _DEFAULT_REGISTRY
