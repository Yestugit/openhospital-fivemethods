from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mematlas.core.capabilities import IntegrationSlot, MethodDescriptor, Operation
from mematlas.core.registry import PluginRegistry, default_registry


@dataclass(frozen=True)
class MethodContract:
    """Small machine-readable contract derived from a method descriptor."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    learning: tuple[str, ...]
    controls: tuple[str, ...]
    scopes: tuple[str, ...]
    default_scope: str

    @classmethod
    def from_descriptor(cls, descriptor: MethodDescriptor) -> "MethodContract":
        operations = descriptor.capabilities.operations
        learning = []
        if Operation.ADD in operations:
            learning.append("unit")
        if Operation.EPISODE_END in operations:
            learning.append("task_boundary")
        controls = tuple(sorted(
            operation.value for operation in operations
            if operation in {
                Operation.RESET,
                Operation.SAVE,
                Operation.LOAD,
                Operation.FLUSH,
                Operation.COMMAND,
            }
        ))
        inputs = {event.value for event in descriptor.capabilities.accepted_events}
        inputs.add("query")
        if Operation.ADD in operations:
            inputs.add("memory_unit")
        output_slots = {
            slot.value for slot in descriptor.capabilities.integration_slots
            if slot != IntegrationSlot.EPISODE
        }
        return cls(
            inputs=tuple(sorted(inputs)),
            outputs=tuple(sorted(output_slots)),
            learning=tuple(learning),
            controls=controls,
            scopes=tuple(sorted(descriptor.capabilities.scopes)),
            default_scope=descriptor.capabilities.default_scope,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "learning": list(self.learning),
            "controls": list(self.controls),
            "scopes": list(self.scopes),
            "default_scope": self.default_scope,
        }


@dataclass(frozen=True)
class MethodCard:
    method_id: str
    name: str
    description: str
    contract: MethodContract
    config_schema: Mapping[str, Any]
    recommended_executor: str
    fidelity: str
    paper_url: str | None
    repository_url: str | None
    limitations: tuple[str, ...]
    requirements: Mapping[str, Any]
    route_hints: tuple[str, ...]
    environment: Mapping[str, Any] | None = None
    interface_version: int = 3
    commands: Mapping[str, Any] | None = None
    input_schemas: Mapping[str, Any] | None = None

    @classmethod
    def from_descriptor(cls, descriptor: MethodDescriptor) -> "MethodCard":
        environment = None
        try:
            from mematlas.environments import environment_spec
            environment = environment_spec(descriptor.method_id).to_dict()
        except ValueError:
            pass
        return cls(
            method_id=descriptor.method_id,
            name=descriptor.name,
            description=descriptor.summary,
            contract=MethodContract.from_descriptor(descriptor),
            config_schema=descriptor.config_schema,
            recommended_executor=descriptor.recommended_executor,
            fidelity=descriptor.fidelity,
            paper_url=descriptor.paper_url,
            repository_url=descriptor.repository_url,
            limitations=descriptor.limitations,
            requirements=descriptor.requirements,
            route_hints=descriptor.route_hints,
            environment=environment,
            interface_version=descriptor.interface_version,
            commands=descriptor.commands,
            input_schemas=descriptor.input_schemas,
        )

    def agent_description(self) -> dict[str, str]:
        return {
            "method_id": self.method_id,
            "name": self.name,
            "description": self.description,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.agent_description(),
            "contract": self.contract.to_dict(),
            "config_schema": dict(self.config_schema),
            "recommended_executor": self.recommended_executor,
            "fidelity": self.fidelity,
            "paper_url": self.paper_url,
            "repository_url": self.repository_url,
            "limitations": list(self.limitations),
            "requirements": dict(self.requirements),
            "route_hints": list(self.route_hints),
            "environment": dict(self.environment) if self.environment else None,
            "interface_version": self.interface_version,
            "status": "available" if self.interface_version == 3 else "migration_required",
            "commands": dict(self.commands or {}),
            "input_schemas": dict(self.input_schemas or {}),
        }


class MethodCatalog:
    """Public method directory without exposing registry internals."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def ids(self) -> list[str]:
        return self.registry.ids()

    def get(self, method_id: str) -> MethodCard:
        return MethodCard.from_descriptor(self.registry.get(method_id).descriptor)

    def list(self) -> list[MethodCard]:
        return [MethodCard.from_descriptor(plugin.descriptor) for plugin in self.registry.plugins()]

    def descriptions(self, *, include_controls: bool = True) -> list[dict[str, str]]:
        cards = self.list()
        if not include_controls:
            cards = [card for card in cards if not card.method_id.startswith("baseline_")]
        return [card.agent_description() for card in cards]
