"""受控 Agent 实跑：真实 Chroma，固定模型响应；不用于模型效果比较。"""
from __future__ import annotations
from mematlas.test.evaluation.support import add_event

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from mematlas import MemAtlasConfig, Memory, MemoryEvent, RuntimeConfig
from mematlas.core.files import atomic_write_json
from mematlas.methods.a_mem import AMemConfig
from mematlas.methods.a_mem.example import run
from mematlas.test.a_mem.fixtures import services


async def check(output: Path) -> None:
    """保存每轮真实写入内容，并验证去重、演化、来源、冻结、恢复和清理。"""
    config = MemAtlasConfig(method=AMemConfig(), scope={"user_id": "alice"},
                            runtime=RuntimeConfig(artifact_dir=str(output.parent / "runtime")))
    async def agent(messages: list[dict[str, str]]) -> str:
        # This fixed test Agent sees only the actual retrieved prompt, no reference answers.
        text = messages[0]["content"]
        return "9443" if "9443" in text else "8443" if "8443" in text else "No information available."

    answers = await run(["Helios port is 8443", "Helios port is now 9443", "What is the Helios port?"],
                        agent, config=config, output=output, services=services(evolve=True))
    assert answers == ["8443", "9443", "9443"], answers
    snapshot = json.loads((output / "memory-3.json").read_text(encoding="utf-8"))
    state = snapshot["state"]
    assert len(state["core"]["memories"]) == 6
    assert len(state["seen_events"]) == 6
    assert any(note["links"] for note in state["core"]["memories"].values())
    atomic_write_json(output / "memory-contents.json", state)
    restored_config = replace(config, runtime=RuntimeConfig(artifact_dir=str(output.parent / "restored-runtime")))
    async with Memory(restored_config, services=services()) as restored:
        await restored.load(output / "memory-3.json")
        before = await restored._method.save()
        async with restored.read_only():
            retrieved = await restored.search("Helios", top_k=6)
        assert await restored._method.save() == before
        assert len(retrieved.items) == 6
        assert all(item["source_event_ids"] for item in retrieved.items)
        atomic_write_json(output / "restored-retrieval.json", retrieved.to_dict())
        await restored.reset()
        assert not (await restored.search("Helios")).items
        await add_event(restored, MemoryEvent("observation", "Fresh state"))
        assert len((await restored.search("Fresh state")).items) == 1
    atomic_write_json(output / "checks.json", {
        "mode": "controlled_providers_real_chroma", "real_model_acceptance": "pending",
        "answers": answers, "notes": 6, "written_events": 6,
        "checks": ["public_agent_example", "observe_learn_dedup", "automatic_evolution",
                   "source_event_ids", "save_load", "frozen_retrieval_unchanged", "reset_and_reuse"],
    })
    print("Controlled checks passed. Real-model and human acceptance remain pending.")
    print("Evidence:", output.resolve())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新的证据目录，不覆盖已有结果")
    args = parser.parse_args(argv)
    asyncio.run(check(args.output.resolve()))


if __name__ == "__main__":
    main()
