"""两种真实 Benchmark 适配器的合成数据联调；不代表官方数据/模型效果。"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, RuntimeConfig
from mematlas.core.files import atomic_write_json
from mematlas.methods.a_mem import AMemConfig
from mematlas.test.a_mem.fixtures import services


class FixtureReader:
    """The controlled reader only sees DefaultQA prompts, never case.reference."""

    config = {"mode": "controlled_fixture_reader"}

    def __init__(self):
        self.prompts: list[list[dict[str, Any]]] = []

    async def complete(self, messages, **options):
        self.prompts.append(messages)
        return "8443" if "8443" in messages[-1]["content"].split("Question:")[0] else "No information available."


def fixture(kind: str):
    if kind == "locomo":
        return [{"sample_id": "synthetic", "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob", "session_1_date_time": "1:00 pm on 1 November, 2026",
            "session_1": [{"speaker": "Alice", "text": "Helios port is 8443", "dia_id": "D1:1"},
                          {"speaker": "Bob", "text": "Noted", "dia_id": "D1:2"}]},
            "qa": [{"question": f"What is the Helios port? Query {i}", "answer": "8443", "category": 4,
                    "evidence": ["D1:1"]} for i in range(3)]}]
    return [{"question_id": f"synthetic-{i}", "question": "What is the Helios port?", "answer": "8443",
             "question_type": "single-session-user", "question_date": "2026-11-02",
             "haystack_sessions": [[{"role": "user", "content": "Helios port is 8443"}]],
             "haystack_session_ids": ["session-1"], "haystack_dates": ["2026-11-01"],
             "answer_session_ids": ["session-1"]} for i in range(1)]


async def check(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for kind in ("locomo", "longmemeval"):
        limit = 3 if kind == "locomo" else 1
        data = output / f"synthetic-{kind}.json"
        atomic_write_json(data, fixture(kind))
        supplied, reader = services(), FixtureReader()
        supplied.objects["answer_llm"] = reader
        memory = MemAtlasConfig(method=AMemConfig(), scope={"user_id": "fixture"},
            runtime=RuntimeConfig(artifact_dir=str(output / kind / "memory")))
        benchmark = {"kind": kind, "path": str(data)}
        if kind == "longmemeval":
            # No real judge is configured; test retrieval rather than fabricate a judge label.
            benchmark["official_judge"] = False
        experiment = ExperimentConfig(methods=(memory,), benchmark=benchmark,
            test_limit=limit, output_dir=str(output / kind / "evaluation"))
        runner = ExperimentRunner()
        report = runner.check(experiment, services=supplied)
        assert report["ready"], report
        result = (await runner.evaluate(experiment, services=supplied))[0]
        assert not result.failures, result.failures
        assert len(result.predictions) == limit
        assert len(reader.prompts) == limit
        for prediction in result.predictions:
            assert prediction.output == "8443"
            assert prediction.applied_memory
            assert any(item["source_event_ids"] for used in prediction.applied_memory for item in used.items)
        atomic_write_json(output / kind / "qa-prompts.json", reader.prompts)
        # Read actual Chroma data for review, without invoking an embedding model.
        import chromadb
        databases = []
        for database in (output / kind / "memory").rglob("chroma.sqlite3"):
            collection = chromadb.PersistentClient(path=str(database.parent)).get_collection(
                "memories", embedding_function=None)
            databases.append({"path": str(database), "notes": collection.get(include=["documents", "metadatas"])})
        assert databases
        atomic_write_json(output / kind / "database-contents.json", databases)
        summaries[kind] = {"cases": limit, "failures": 0, "database_count": len(databases),
                           "metrics": dict(result.metrics)}
    record = {"mode": "synthetic_data_controlled_providers_real_chroma",
              "official_benchmark_acceptance": "pending", "results": summaries}
    atomic_write_json(output / "checks.json", record)
    print(json.dumps({"mode": record["mode"], "cases": {kind: row["cases"] for kind, row in summaries.items()},
                      "evidence": str(output.resolve())}, ensure_ascii=False, indent=2))
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="新的证据目录，不覆盖已有结果")
    args = parser.parse_args(argv)
    asyncio.run(check(args.output.resolve()))


if __name__ == "__main__":
    main()
