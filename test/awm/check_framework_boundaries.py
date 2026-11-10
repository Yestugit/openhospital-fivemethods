"""离线复现 AWM 的统一评测接入边界，不调用模型或网页环境。"""
from __future__ import annotations

import argparse
import asyncio
import ast
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

from mematlas import ExperimentConfig, MemAtlasConfig, MemoryEpisode, MemoryEvent
from mematlas.benchmarks.factory import load_benchmark
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.registry import default_registry
from mematlas.evaluation.execution import _EvaluationRun
from mematlas.evaluation.qa import _qa_type
from mematlas.evaluation.validation import validate_cases


def compare_sources(reference: Path) -> dict[str, Any]:
    """只读比较旧框架与当前代码，不导入旧方法或运行旧 runner。"""
    current = Path(__file__).resolve().parents[2]
    result = {}
    for relative in (
        "core/plugin.py", "runtime/memory.py", "evaluation/qa.py",
        "evaluation/execution.py", "benchmarks/factory.py", "methods/awm/method.py",
    ):
        original, revised = reference / relative, current / relative
        if not original.is_file() or not revised.is_file():
            raise FileNotFoundError(f"Missing comparison input: {relative}")
        before, after = original.read_bytes(), revised.read_bytes()
        result[relative] = {
            "same_bytes": before == after,
            "original_sha256": hashlib.sha256(before).hexdigest(),
            "current_sha256": hashlib.sha256(after).hexdigest(),
        }
    for label, root in (("original", reference), ("current", current)):
        tree = ast.parse((root / "methods/awm/method.py").read_text(encoding="utf-8-sig"))
        memory_class = next(node for node in tree.body
                            if isinstance(node, ast.ClassDef) and node.name == "AWMMemory")
        result[label + "_awm_signatures"] = {
            node.name: [arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)]
            for node in memory_class.body if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"observe", "retrieve", "learn", "save", "load", "save_state", "load_state"}
        }
    return result


async def check(reference: Path) -> dict[str, Any]:
    config = MemAtlasConfig.from_dict({
        "method": {"method_id": "awm", "params": {"profile": "mind2web"}},
        "scope": {"user_id": "interface-audit"},
    })
    experiment = ExperimentConfig(methods=(config,), benchmark=None,
                                  protocol="experience_train_test")
    plugin = default_registry().get("awm")
    case = BenchmarkCase("train-task", "Navigate to NBA", scope=config.scope, metadata={
        "memory_extensions": {"awm": {"website": "sports.yahoo"}},
        "awm": {"website": "sports.yahoo", "domain": "Sports", "subdomain": "Sports"},
    })
    report: dict[str, Any] = {
        "kind": "controlled offline framework boundary probes; not a benchmark run",
        "model_calls": 0,
        "provided_reference": str(reference.resolve()),
        "comparison": compare_sources(reference),
        "qa_registration": _qa_type(plugin).__name__,
    }
    try:
        validate_cases(experiment, plugin, [case], [replace(case, case_id="test-task")], None)
    except Exception as error:
        report["training_without_collected_episode"] = {
            "raised": True, "type": type(error).__name__, "message": str(error),
        }
    else:
        report["training_without_collected_episode"] = {"raised": False}

    try:
        load_benchmark("mind2web", "not-read.json")
    except Exception as error:
        report["native_benchmark_registration"] = {
            "raised": True, "type": type(error).__name__, "message": str(error),
        }
    else:
        report["native_benchmark_registration"] = {"raised": False}

    async def controlled_actor(case: BenchmarkCase, memory: Any) -> MemoryEpisode:
        # Explicit probe data: reproduce metadata loss without any LLM or browser.
        return MemoryEpisode(task=str(case.query), output="CLICK [18241]", success=False,
            events=[MemoryEvent("action", "CLICK [18241]")],
            extensions={"awm": {"step_scores": [1, 1, 0]}})

    run = object.__new__(_EvaluationRun)
    run.agent = controlled_actor
    prediction = await run.execute_case(None, case)
    report["prediction_conversion"] = {
        "input_episode": {"success": False, "event_count": 1, "step_scores": [1, 1, 0]},
        "output_prediction": prediction.to_dict(),
        "success_preserved": "success" in prediction.metadata,
        "trajectory_preserved": "events" in prediction.metadata,
        "step_scores_preserved": "step_scores" in prediction.metadata,
    }
    report["code_locations"] = {
        name: {"file": inspect.getsourcefile(function),
               "line": inspect.getsourcelines(function)[1]}
        for name, function in {
            "train_batch": _EvaluationRun._train_batch,
            "execute_case": _EvaluationRun.execute_case,
            "validate_cases": validate_cases,
            "load_benchmark": load_benchmark,
        }.items()
    }
    report["expected_limitations_reproduced"] = (
        report["qa_registration"] == "AWMQA"
        and report["training_without_collected_episode"]["raised"]
        and report["native_benchmark_registration"]["raised"]
        and not report["prediction_conversion"]["success_preserved"]
        and not report["prediction_conversion"]["trajectory_preserved"]
        and not report["prediction_conversion"]["step_scores_preserved"]
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="用户提供的迁移前 MemAtlas 仓库，只读")
    parser.add_argument("--output", type=Path, required=True,
                        help="新的 JSON 报告文件，不覆盖已有结果")
    args = parser.parse_args(argv)
    report = asyncio.run(check(args.reference_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"report": str(args.output.resolve()),
        "expected_limitations_reproduced": report["expected_limitations_reproduced"],
        "model_calls": report["model_calls"]}, ensure_ascii=True))
    if not report["expected_limitations_reproduced"]:
        raise SystemExit("Framework behavior changed; inspect the report before reusing the conclusions")


if __name__ == "__main__":
    main()
