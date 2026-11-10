"""Print a method's Memory config; no model download or experiment execution.

Usage: python scripts/memory-adapter-config.py memora --output config.json
"""
import argparse
import json
from pathlib import Path

METHODS = ("memos", "mem0", "memmachine", "emem", "memora")


def configuration(method_id, *, root=None, llm_model="qwen3.5-flash",
                  llm_base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
                  neo4j_uri=None):
    if method_id not in METHODS:
        raise ValueError("Unknown memory adapter")
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    params = {"write_policy": "episode"} if method_id == "mem0" else {}
    if neo4j_uri is not None:
        if method_id != "memos":
            raise ValueError("--neo4j-uri applies only to MemOS")
        params["neo4j_uri"] = neo4j_uri
    return {
        "method": {"method_id": method_id, "params": params},
        "scope": {"user_id": "openhospital-adapter-check", "agent_id": "doctor"},
        "runtime": {
            "executor": "subprocess",
            "python_executable": str(root / (".venv-" + method_id) / "Scripts" / "python.exe"),
            "artifact_dir": str(root / ".mematlas" / "native-adapter-runs" / method_id),
            "source_roots": {method_id: str(root / ".mematlas" / "upstream" / method_id)}
                if method_id in {"emem", "memora"} else {},
        },
        "providers": {
            "llm": {"model": llm_model, "base_url": llm_base_url, "api_key_env": "MODEL_API_KEY",
                    "options": {"max_tokens": 8192}},
            "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B", "dimensions": 1024,
                          "base_url": "http://127.0.0.1:8766/v1", "api_key_env": "LOCAL_EMBEDDING_API_KEY"},
        },
        "metadata": {"purpose": "native_adapter_integration", "formal_experiment_defined": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--llm-model", default="qwen3.5-flash")
    parser.add_argument("--llm-base-url", default="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--neo4j-uri", help="MemOS Neo4j URI; omit to use the method default")
    options = parser.parse_args()
    text = json.dumps(configuration(options.method, llm_model=options.llm_model,
                                   llm_base_url=options.llm_base_url, neo4j_uri=options.neo4j_uri), indent=2, ensure_ascii=False) + "\n"
    if options.output:
        with options.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
