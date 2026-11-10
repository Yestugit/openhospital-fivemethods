from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from mematlas.core.config import MemAtlasConfig
from mematlas.core.plugin import MethodServices
from mematlas.core.registry import default_registry
from mematlas.selection import ConfigCompiler, MethodSelector, NaturalLanguageMethodRouter, TaskSpec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mematlas")
    subparsers = parser.add_subparsers(dest="command", required=True)
    methods = subparsers.add_parser("methods", help="List or describe method plugins")
    methods.add_argument("action", choices=("list", "descriptions", "describe"))
    methods.add_argument("method_id", nargs="?")
    check = subparsers.add_parser("check", help="Validate a config and inspect dependencies")
    check.add_argument("config")
    check.add_argument("--experiment", action="store_true", help="Check a method comparison configuration")
    results = subparsers.add_parser("results", help="Read comparison or continuous-learning results")
    results.add_argument("path")
    recommend = subparsers.add_parser("recommend", help="Recommend methods from a JSON TaskSpec")
    recommend.add_argument("spec", help="Inline JSON TaskSpec")
    recommend.add_argument("--limit", type=int, default=5)
    recommend_text = subparsers.add_parser("recommend-text", help="Parse a natural-language requirement and recommend methods")
    recommend_text.add_argument("requirement")
    _add_model_arguments(recommend_text)
    recommend_text.add_argument("--embedding-model", default="text-embedding-v4")
    recommend_text.add_argument("--embedding-base-url")
    recommend_text.add_argument("--embedding-api-key-env")
    recommend_text.add_argument("--embedding-dimensions", type=int, default=1024)
    recommend_text.add_argument("--limit", type=int, default=5)
    recommend_text.add_argument(
        "--include-controls",
        action="store_true",
        help="Allow no-memory and retrieval controls in the LLM candidate catalog",
    )
    benchmarks = subparsers.add_parser("benchmark", help="Inspect a classic or custom benchmark")
    benchmarks.add_argument("kind", choices=("locomo", "longmemeval", "custom", "mind2web", "hotpotqa",
                                              "reasoningbank_trajectory",
                                              "ael_portfolio_slice", "memcon_action_observation"))
    benchmarks.add_argument("path", nargs="?")
    benchmarks.add_argument("--limit", type=int)
    benchmarks.add_argument("--max-questions-per-conversation", type=int)
    benchmarks.add_argument("--mode", choices=("text", "image_text"), default="text")
    run = subparsers.add_parser("run", help="Run a unified experiment configuration")
    run.add_argument("--config", required=True, help="Experiment JSON, YAML or TOML configuration")
    run.add_argument("--resume-from", help="Resume a training experiment into its new output directory")
    environment = subparsers.add_parser("env", help="Inspect an isolated method environment")
    environment.add_argument("method_id")
    environment.add_argument("--python", dest="python_executable")
    environment.add_argument("--methods-root")
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="qwen3.5-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)


def _llm(args, usage=None):
    from mematlas.providers import OpenAICompatibleLLM
    return OpenAICompatibleLLM(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        default_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
        usage=usage,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )


def _selection_payload(result, *, compiler: ConfigCompiler | None = None, provider_args=None):
    recommendations = []
    for item in result.recommendations:
        payload = {
            "method_id": item.method_id,
            "score": item.score,
            "reasons": item.reasons,
            "description": item.description,
        }
        if compiler is not None:
            try:
                if provider_args is None:
                    config = compiler.compile(item.method_id)
                else:
                    config = compiler.compile(
                        item.method_id,
                        providers={
                            "llm": {"model": provider_args.model, "base_url": provider_args.base_url,
                                    "api_key_env": provider_args.api_key_env},
                            "embedding": {"model": provider_args.embedding_model,
                                "base_url": provider_args.embedding_base_url or provider_args.base_url,
                                "api_key_env": provider_args.embedding_api_key_env or provider_args.api_key_env,
                                "dimensions": provider_args.embedding_dimensions},
                        },
                    )
                payload["config"] = config.to_dict()
                payload["config_errors"] = []
            except ValueError as exc:
                payload["config"] = None
                payload["config_errors"] = [str(exc)]
        recommendations.append(payload)
    return {"recommendations": recommendations, "excluded": result.excluded}


async def _run_evaluation(args, registry):
    from mematlas.evaluation.experiment import ExperimentConfig, ExperimentRunner
    return await ExperimentRunner(registry=registry).evaluate(
        ExperimentConfig.load(args.config), resume_from=args.resume_from)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = default_registry()
    if args.command == "results":
        from mematlas.evaluation.artifacts import read_results
        print(json.dumps([row.to_dict() for row in read_results(args.path)], ensure_ascii=False, indent=2))
        return 0
    if args.command == "methods":
        if args.action == "list":
            for plugin in registry.plugins():
                descriptor = plugin.descriptor
                status = "interface_v3" if descriptor.interface_version == 3 else "migration_required"
                print(f"{descriptor.method_id}\t{descriptor.name}\t{status}\t{descriptor.summary}")
            return 0
        if args.action == "descriptions":
            print(json.dumps(registry.descriptions(), ensure_ascii=False, indent=2))
            return 0
        from mematlas.api import MemAtlas
        if args.method_id is None:
            raise ValueError("methods describe requires method_id")
        print(json.dumps(MemAtlas(registry).describe(args.method_id), ensure_ascii=False, indent=2))
        return 0
    if args.command == "check":
        if args.experiment:
            from mematlas.evaluation.experiment import ExperimentConfig, ExperimentRunner
            result = ExperimentRunner(registry=registry).check(ExperimentConfig.load(args.config))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 1
        config = MemAtlasConfig.load(args.config)
        from mematlas.runtime.configuration import check_config
        result = check_config(config, registry=registry)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1
    if args.command == "recommend":
        result = MethodSelector(registry).recommend(TaskSpec.from_dict(json.loads(args.spec)), limit=args.limit)
        print(json.dumps(_selection_payload(result), ensure_ascii=False, indent=2))
        return 0
    if args.command == "recommend-text":
        result = asyncio.run(NaturalLanguageMethodRouter(
            _llm(args), registry
        ).route(
            args.requirement,
            limit=args.limit,
            include_controls=args.include_controls,
        ))
        print(json.dumps({
            "requirement": args.requirement,
            **_selection_payload(
                result,
                compiler=ConfigCompiler(registry),
                provider_args=args,
            ),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "benchmark":
        from mematlas.benchmarks import load_benchmark
        benchmark = load_benchmark(
            args.kind,
            args.path,
            limit=args.limit,
            mode=args.mode,
            max_questions_per_conversation=args.max_questions_per_conversation,
        )
        cases = tuple(benchmark.cases("test"))
        print(json.dumps({
            "description": dict(benchmark.describe()),
            "case_count": len(cases),
            "history_event_count": sum(len(case.history) for case in cases),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        results = asyncio.run(_run_evaluation(args, registry))
        print(json.dumps({"runs": [
            {"method_id": result.method_id, "predictions": len(result.predictions),
             "failures": len(result.failures), "checkpoints": len(result.learning_curve),
             "output_dir": result.run_metadata.get("output_dir")} for result in results
        ]}, ensure_ascii=False, indent=2))
        return 1 if any(result.failures for result in results) else 0
    if args.command == "env":
        from mematlas.environments import inspect_method_environment
        result = inspect_method_environment(
            args.method_id,
            python_executable=args.python_executable,
            methods_root=args.methods_root,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
