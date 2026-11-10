"""Historical simplified-Mem0 check; retained for reference, no longer executable."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mematlas import ExperimentConfig, ExperimentRunner


async def main(output, api_key_env, model):
    raise RuntimeError("Historical simplified Mem0 check is retired. Use the pinned native environment and current adapter examples.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scope = {"user_id": "live-evaluation"}
    rows = [
        {"case_id": "train-port", "split": "train", "scope": scope,
         "query": "Remember the Vesper service port",
         "history": [{"event_id": "port-fact", "role": "user", "text": "The Vesper service port is 7349."}]},
        {"case_id": "train-timeout", "split": "train", "scope": scope,
         "query": "Remember the Vesper service timeout",
         "history": [{"event_id": "timeout-fact", "role": "user", "text": "The Vesper service timeout is 47 seconds."}]},
        {"case_id": "test-port", "split": "test", "scope": scope,
         "query": "What is the Vesper service port? Return only the number.", "reference": "7349"},
        {"case_id": "test-timeout", "split": "test", "scope": scope,
         "query": "What is the Vesper service timeout in seconds? Return only the number.", "reference": "47"},
    ]
    dataset = output / "cases.json"
    dataset.write_text(json.dumps(rows), encoding="utf-8")
    config = ExperimentConfig.from_dict({
        "methods": [{"method": {"method_id": "mem0", "params": {"infer": False, "hybrid_search": False, "top_k": 3}},
            "scope": scope, "runtime": {"artifact_dir": str(output / "memory")},
            "providers": {"embedding": {"model": "text-embedding-v4", "dimensions": 1024,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": api_key_env}}}],
        "benchmark": {"kind": "custom", "path": str(dataset)},
        "answer_model": {"model": model, "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": api_key_env, "options": {"temperature": 0, "extra_body": {"enable_thinking": False}}},
        "protocol": "experience_learning_curve", "train_limit": 2, "test_limit": 2,
        "checkpoints": [0, 1, 2], "output_dir": str(output / "evaluation"),
    })
    runner = ExperimentRunner()
    results = await runner.evaluate(config)
    result = results[0]
    if result.failures:
        message = json.dumps(list(result.failures), ensure_ascii=False)
        credential = os.getenv(api_key_env)
        raise RuntimeError(message.replace(credential, "[redacted]") if credential else message)
    assert result.metrics["common.exact_match"] == 1.0, result.metrics
    assert [p["checkpoint"] for p in result.learning_curve] == [0, 1, 2]
    assert all(p["frozen_verified"] for p in result.learning_curve)
    assert "common.cost" not in result.resource_metrics
    assert "common.total_evaluation_cost" not in result.resource_metrics
    assert all("combined_cost" not in p for p in result.learning_curve)
    assert runner.results(output / "evaluation")[0].to_dict() == result.to_dict()
    print(json.dumps({"status": "completed", "method": "mem0", "memory_variant": "infer=false, hybrid_search=false",
        "answer_model": model, "embedding_model": "text-embedding-v4",
        "points": [{"trained": p["checkpoint"], "exact_match": p["common.exact_match"],
                    "answers": [row["output"] for row in p["predictions"]]} for p in result.learning_curve],
        "tokens": result.resource_metrics.get("common.total_tokens"),
        "output_dir": str(output / "evaluation")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default="qwen3.5-flash")
    args = parser.parse_args()
    asyncio.run(main(args.output, args.api_key_env, args.model))

