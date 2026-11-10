"""Checks for the operations and inputs actually used by a caller."""
from dataclasses import dataclass, field

from mematlas.core.capabilities import IntegrationSlot, Operation
from mematlas.core.errors import CapabilityMismatchError
from mematlas.core.models import EventKind, Modality, content_modalities, content_text


@dataclass(frozen=True)
class ProtocolRequirements:
    operations: frozenset[Operation] = field(default_factory=frozenset)
    events: frozenset[EventKind] = field(default_factory=frozenset)
    modalities: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))
    slots: frozenset[IntegrationSlot] = field(default_factory=lambda: frozenset({IntegrationSlot.CONTEXT}))
    episode_fields: frozenset[str] | None = None
    frozen: bool = False
    recovery: bool = False
    scope_count: int = 1

    def check(self, descriptor):
        caps, errors = descriptor.capabilities, []
        if descriptor.interface_version != 3:
            errors.append("method requires migration to Memory interface v3")
        for name, needed, supported in (
            ("operations", self.operations, caps.operations),
            ("modalities", self.modalities, caps.modalities),
            ("output slots", self.slots, caps.integration_slots),
        ):
            if needed - supported:
                errors.append(f"unsupported {name}: {sorted(row.value for row in needed - supported)}")
        if caps.accepted_events and self.events - caps.accepted_events:
            errors.append(f"unsupported events: {sorted(row.value for row in self.events - caps.accepted_events)}")
        if self.frozen and caps.read_only_mode == "unsupported":
            errors.append("frozen evaluation is unsupported")
        if self.recovery and (not caps.supports_persistence or not {Operation.SAVE, Operation.LOAD}.issubset(caps.operations)):
            errors.append("recovery requires save/load support")
        if caps.max_scopes is not None and self.scope_count > caps.max_scopes:
            errors.append(f"method supports at most {caps.max_scopes} scope(s)")
        if self.episode_fields is not None and caps.required_episode_fields - self.episode_fields:
            errors.append(f"missing episode fields: {sorted(caps.required_episode_fields - self.episode_fields)}")
        return errors

    def validate(self, descriptor):
        errors = self.check(descriptor)
        if errors:
            raise CapabilityMismatchError(f"{descriptor.method_id}: " + "; ".join(errors))


def episode_fields(episode):
    return {
        "success": episode.success is not None,
        "reward": episode.reward is not None,
        "task": bool(episode.task and episode.task.strip()),
        "trajectory": any(content_text(row.content).strip() and
                          row.extensions.get("_unit_field") not in {"task", "feedback"}
                          for row in episode.events or ()),
    }


def validate_episode(capabilities, episode):
    available = episode_fields(episode)
    missing = [name for name in capabilities.required_episode_fields if not available.get(name)]
    if missing:
        raise CapabilityMismatchError(f"Episode is missing required fields: {sorted(missing)}")
    for event in episode.events or []:
        validate_modalities(capabilities, event.content)


def validate_modalities(capabilities, content):
    unsupported = content_modalities(content) - capabilities.modalities
    if unsupported:
        raise CapabilityMismatchError(f"Unsupported content modalities: {sorted(row.value for row in unsupported)}")


def validate_scope(capabilities, scope):
    missing = {name.removesuffix("_id") for name in scope} - capabilities.scopes
    if missing:
        raise CapabilityMismatchError(f"Unsupported scope fields: {sorted(missing)}")
