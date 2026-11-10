"""校验并导出已完成测评的笔记、检索来源和冻结证据，不调用模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Sequence

from mematlas import ExperimentRunner
from mematlas.core.contributions import applied_sources
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import file_sha256
from mematlas.evaluation.artifacts import recovery_checkpoint
from mematlas.test.a_mem.check_saved_state import note_contents


def export(experiment: Path, output: Path, *, expected_history: int = 419, expected_questions: int = 3) -> dict:
    results = ExperimentRunner.results(experiment)
    if len(results) != 1 or results[0].failures:
        raise ValueError("A single successful method result is required")
    result = results[0]
    if len(result.predictions) != expected_questions:
        raise ValueError("Unexpected question count")
    run = experiment / "runs/001-a-mem"
    position, checkpoint, payload = recovery_checkpoint(run)
    memory_path = checkpoint.parent / payload["memory_file"]
    if file_sha256(memory_path) != payload["memory_sha256"]:
        raise ValueError("Final memory snapshot checksum mismatch")
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    state = memory["state"]
    notes = state["core"]["memories"]
    if position != expected_history or len(notes) != expected_history or len(state["seen_events"]) != expected_history:
        raise ValueError("The complete expected history was not written exactly once")
    measured = [point for point in result.learning_curve if point.get("split") == "primary_test"]
    if not measured or not all(point.get("frozen_verified") for point in measured):
        raise ValueError("Frozen evaluation was not verified")
    cases = []
    for prediction in result.predictions:
        used = applied_sources(prediction.applied_memory)
        original, lineage = set(), set()
        for item in used:
            note = notes[item["memory_id"]]
            original.update(note["source_event_ids"])
            lineage.update(item["source_event_ids"])
        evidence = {str(value).replace("(", "").replace(")", "")
                    for value in prediction.metadata["case_metadata"].get("evidence", []) if value}
        cases.append({"case_id": prediction.case_id, "output": prediction.output, "reference": prediction.reference,
            "original_write_sources": sorted(original), "evolution_input_sources": sorted(lineage),
            "annotated_evidence": sorted(evidence),
            "original_write_source_recall": len(evidence & original) / len(evidence) if evidence else None,
            "evolution_input_source_recall": len(evidence & lineage) / len(evidence) if evidence else None})
    summary = {"mode": "official_data_real_qwen", "history_events": expected_history,
        "questions": expected_questions, "notes": len(notes), "frozen_verified": True,
        "evolution_count": state["core"]["evo_count"],
        "neighbor_updates": sum(sum(row.get("action") == "update_neighbor" for row in note["evolution_history"])
                                for note in notes.values()),
        "metrics": dict(result.metrics), "resource_metrics": dict(result.resource_metrics),
        "original_write_source_recall": mean(row["original_write_source_recall"] for row in cases
                                              if row["original_write_source_recall"] is not None),
        "evolution_input_source_recall": mean(row["evolution_input_source_recall"] for row in cases
                                               if row["evolution_input_source_recall"] is not None),
        "source_interpretation": "Evolution input sources are conservative dependencies, not proof of textual fact coverage.",
        "human_acceptance": "pending", "snapshot_sha256": payload["memory_sha256"]}
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "memory-contents.json", note_contents(state))
    atomic_write_json(output / "case-audit.json", cases)
    atomic_write_json(output / "results.json", [result.to_dict()])
    print(json.dumps({"notes": len(notes), "questions": len(cases), "frozen_verified": True,
                      "output": str(output.resolve())}, ensure_ascii=False))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="新的证据目录")
    parser.add_argument("--expected-history", type=int, default=419)
    parser.add_argument("--expected-questions", type=int, default=3)
    args = parser.parse_args(argv)
    export(args.experiment, args.output, expected_history=args.expected_history,
           expected_questions=args.expected_questions)


if __name__ == "__main__":
    main()
