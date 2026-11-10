"""用真实服务恢复已有 demo 快照，核对冻结、来源、去重和 reset。"""
from __future__ import annotations
from mematlas.test.evaluation.support import add_event

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

from mematlas import Memory, MemoryEvent, RuntimeConfig
from mematlas.core.files import atomic_write_json
from mematlas.methods.a_mem.example import _config_from_environment


def note_contents(state):
    """供 review 阅读的笔记；完整向量仍保留在原始快照中。"""
    output = {}
    for identity, raw in state["core"]["memories"].items():
        note = dict(raw)
        vector = note.pop("embedding")
        note["embedding_dimensions"] = len(vector)
        note["embedding_sha256"] = hashlib.sha256(json.dumps(vector).encode()).hexdigest()
        output[identity] = note
    return {**state, "core": {**state["core"], "memories": output}}


async def check(snapshot: Path, output: Path) -> None:
    config = _config_from_environment()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    config = replace(config, runtime=RuntimeConfig(artifact_dir=str(output / "runtime")))
    async with Memory(config) as memory:
        await memory.load(snapshot)
        before = await memory._method.save()
        count = len(before["core"]["memories"])
        assert count == 6, "Expected the six-note real Agent demo snapshot"
        async with memory.read_only():
            used = await memory.search("Helios 服务最新的端口是多少？", top_k=count)
        assert await memory._method.save() == before
        assert len(used.items) == count
        assert any("9443" in row["content"] for row in used.items)
        assert all(set(row["source_event_ids"]) <= set(before["seen_events"]) for row in used.items)
        # A previously observed event must not generate a second note or model call.
        event = next(iter(payload["history"]["events"].values()))
        await add_event(memory, MemoryEvent.from_dict(event))
        assert await memory._method.save() == before
        atomic_write_json(output / "memory-contents.json", note_contents(before))
        atomic_write_json(output / "frozen-retrieval.json", used.to_dict())
        await memory.reset()
        assert not (await memory.search("Helios")).items
        atomic_write_json(output / "reset-state.json", await memory._method.save())
        atomic_write_json(output / "stats.json", memory.stats())
        atomic_write_json(output / "checks.json", {"mode": "real_qwen_snapshot_restore",
            "snapshot": str(snapshot.resolve()), "notes": count, "frozen_state_unchanged": True,
            "duplicate_event_unchanged": True, "reset_empty": True})
    print("Real snapshot restore/freeze/dedup/reset checks passed:", output.resolve())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    asyncio.run(check(args.snapshot, args.output.resolve()))


if __name__ == "__main__":
    main()
