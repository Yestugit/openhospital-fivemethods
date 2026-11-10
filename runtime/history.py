"""One event store, episode references, and an incremental audit journal."""
from copy import deepcopy
import json
import os
from pathlib import Path

from mematlas.core.episodes import EpisodeResult, episode_counts
from mematlas.core.models import MemoryEpisode, MemoryEvent, MemoryUnit
from mematlas.core.provenance import json_sha256


class EpisodeHistory:
    def __init__(self, journal=None):
        self.journal = Path(journal) if journal is not None else None
        self.events = {}
        self.observed = set()
        self.episodes = {}
        self.results = {}
        self.units = {}
        self.processed_units = set()
        self.ended = {}
        self.active_episodes = {}
        self._episode_events = {}
        self._pending = []

    @staticmethod
    def identity(episode_id, scope):
        return json_sha256({"episode_id": episode_id, "scope": scope})

    @staticmethod
    def event_identity(event):
        return json_sha256({"event_id": event.event_id, "scope": event.scope})

    def validate_events(self, events):
        for event in events:
            self._validate_event(event)

    def _validate_event(self, event):
        identity = self.event_identity(event)
        payload = event.to_dict()
        if identity in self.events and self.events[identity] != payload:
            previous = self.events[identity]
            before, after = dict(previous), dict(payload)
            before.pop("episode_id")
            after.pop("episode_id")
            if previous["episode_id"] is not None or before != after:
                raise ValueError("An existing event_id cannot be reused for different event data")
        return identity, payload

    def _record(self, operation, **data):
        if self.journal is not None:
            self._pending.append({"operation": operation, **deepcopy(data)})

    def add_event(self, event):
        identity, payload = self._validate_event(event)
        previous = self.events.get(identity)
        if previous == payload:
            return identity
        self.events[identity] = payload
        if event.episode_id is not None:
            episode = self.identity(event.episode_id, event.scope)
            self._episode_events.setdefault(episode, {})[identity] = None
        if previous is None:
            self._record("event", identity=identity, event=payload)
        else:
            self._record("attach_event", identity=identity, episode_id=event.episode_id)
        return identity

    def mark_observed(self, identities):
        added = set(identities) - self.observed
        if not added.issubset(self.events):
            raise ValueError("Observed event records are missing")
        self.observed.update(added)
        if added:
            self._record("observed", identities=sorted(added))

    def episode_events(self, episode_id, scope):
        identity = self.identity(episode_id, scope)
        return [MemoryEvent.from_dict(self.events[key]) for key in self._episode_events.get(identity, {})]

    def episode(self, identity):
        data = deepcopy(self.episodes[identity])
        data["events"] = [self.events[key] for key in data.pop("event_ids")]
        return MemoryEpisode.from_dict(data)

    def record_episode(self, episode):
        if episode.events is None:
            raise ValueError("Stored episodes require a complete event list")
        identity = self.identity(episode.episode_id, episode.scope)
        data = episode.to_dict()
        data.pop("events")
        data["event_ids"] = [self.event_identity(event) for event in episode.events]
        if len(set(data["event_ids"])) != len(data["event_ids"]):
            raise ValueError("Duplicate event ID in episode")
        if any(event.scope != episode.scope or event.episode_id != episode.episode_id for event in episode.events):
            raise ValueError("Stored event must belong to its episode and scope")
        self.validate_events(episode.events)
        for event in episode.events:
            self.add_event(event)
        self._episode_events[identity] = dict.fromkeys(data["event_ids"])
        if self.episodes.get(identity) != data:
            self.episodes[identity] = data
            self._record("episode", identity=identity, episode=data)
        return identity

    def record_result(self, result, *, episode=None):
        previous = self.results.get(result.episode_id)
        if episode is not None:
            if self.identity(episode.episode_id, episode.scope) != result.episode_id:
                raise ValueError("Result belongs to another episode")
            self.record_episode(episode)
        if previous != result:
            self.results[result.episode_id] = result
            self._record("result", result=result.to_dict())
        self.persist()

    def counts(self, *, exclude=()):
        return episode_counts(row for key, row in self.results.items() if key not in exclude)

    def record_unit(self, unit):
        identity = self.identity(unit.write_ref, unit.scope)
        data = unit.to_dict()
        data.pop("events")
        data["event_ids"] = [self.event_identity(event) for event in unit.events]
        if identity in self.units:
            if self.units[identity] != data:
                raise ValueError("A write_ref cannot be reused for different submission data")
            return identity
        self.validate_events(unit.events)
        for event in unit.events:
            self.add_event(event)
        self.units[identity] = data
        self._record("unit", identity=identity, unit=data)
        return identity

    def unit(self, identity):
        data = deepcopy(self.units[identity])
        data["events"] = [self.events[key] for key in data.pop("event_ids")]
        return MemoryUnit.from_dict(data)

    def mark_processed(self, identity):
        if identity not in self.units:
            raise ValueError("Submission is missing")
        self.processed_units.add(identity)
        self._record("processed_unit", identity=identity)

    def snapshot(self):
        return deepcopy({"schema_version": 4, "events": self.events, "observed": sorted(self.observed),
            "episodes": self.episodes, "results": {key: row.to_dict() for key, row in self.results.items()},
            "units": self.units, "processed_units": sorted(self.processed_units),
            "ended": self.ended, "active_episodes": self.active_episodes})

    def restore(self, data):
        if data.get("schema_version") != 4:
            raise ValueError("Memory history requires schema version 4; old histories require migration")
        restored = EpisodeHistory()
        for key, row in data["events"].items():
            event = MemoryEvent.from_dict(row)
            if self.event_identity(event) != key:
                raise ValueError("Invalid saved event identity")
            restored.add_event(event)
        for key, row in data["episodes"].items():
            payload = deepcopy(row)
            ids = payload.pop("event_ids")
            if not isinstance(ids, list) or any(identity not in restored.events for identity in ids):
                raise ValueError("Episode event records are missing")
            payload["events"] = [restored.events[identity] for identity in ids]
            episode = MemoryEpisode.from_dict(payload)
            if self.identity(episode.episode_id, episode.scope) != key:
                raise ValueError("Invalid saved episode identity")
            if set(ids) != set(restored._episode_events.get(key, {})):
                raise ValueError("Saved episode omitted recorded events")
            restored.record_episode(episode)
        restored.mark_observed(data["observed"])
        for key, row in data["results"].items():
            result = EpisodeResult.from_dict(row)
            if key != result.episode_id or key not in restored.episodes:
                raise ValueError("Invalid saved result identity")
            restored.results[key] = result
        for key, row in data["units"].items():
            payload = deepcopy(row)
            ids = payload.pop("event_ids")
            if any(identity not in restored.events for identity in ids):
                raise ValueError("Unit event records are missing")
            payload["events"] = [restored.events[identity] for identity in ids]
            unit = MemoryUnit.from_dict(payload)
            if self.identity(unit.write_ref, unit.scope) != key:
                raise ValueError("Invalid saved submission identity")
            restored.record_unit(unit)
        restored.processed_units = set(data["processed_units"])
        if not restored.processed_units.issubset(restored.units):
            raise ValueError("Processed submission records are missing")
        restored.ended = deepcopy(data["ended"])
        if set(restored.ended) - restored.episodes.keys() or any(
                row.get("status") not in {"completed", "aborted", "cancelled"} for row in restored.ended.values()):
            raise ValueError("Invalid saved task boundaries")
        restored.active_episodes = deepcopy(data["active_episodes"])
        for owner, ref in restored.active_episodes.items():
            key = self.identity(ref, json.loads(owner))
            if key not in restored.episodes or key in restored.ended:
                raise ValueError("Invalid active task reference")
        self.events, self.observed = restored.events, restored.observed
        self.episodes, self.results = restored.episodes, restored.results
        self.units, self.processed_units = restored.units, restored.processed_units
        self.ended, self.active_episodes = restored.ended, restored.active_episodes
        self._episode_events = restored._episode_events
        self._pending.clear()
        self._record("restore", history=self.snapshot())

    def persist(self):
        if self.journal is not None and self._pending:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"changes": self._pending}, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._pending.clear()

    def reset(self, scope):
        removed = {key for key, row in self.events.items() if row["scope"] == scope}
        for key in removed:
            del self.events[key]
        self.observed.difference_update(removed)
        identities = {key for key, row in self.episodes.items() if row["scope"] == scope}
        for key in list(self._episode_events):
            if not any(event in self.events for event in self._episode_events[key]):
                self._episode_events.pop(key)
        for identity in identities:
            self.episodes.pop(identity, None)
            self.results.pop(identity, None)
            self.ended.pop(identity, None)
        units = {key for key, row in self.units.items() if row["scope"] == scope}
        for key in units:
            self.units.pop(key)
        self.processed_units.difference_update(units)
        self.active_episodes.pop(json.dumps(scope, sort_keys=True, ensure_ascii=False, separators=(",", ":")), None)
        self._record("reset", scope=scope)
        self.persist()
