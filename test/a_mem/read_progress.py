"""只读统计长历史建库进度，不启动 Chroma 客户端或调用模型。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Sequence


def read(run: Path, log: Path | None = None, *, usage_log: Path | None = None,
         checkpoints: Path | None = None) -> dict:
    databases = list(run.rglob("chroma.sqlite3"))
    count = 0
    for database in databases:
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as connection:
            count += connection.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    usage = usage_log or run / "usage.jsonl"
    rows = []
    if usage.exists():
        for line in usage.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrent writer may not have finished the final line.
    positions = [int(path.stem.removeprefix("learned_")) for path in checkpoints.glob("learned_*.json")] if checkpoints else []
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run.resolve()),
            "stored_notes": count, "llm_calls": sum(row.get("resource_kind") == "llm" for row in rows),
            "checkpoint_events": max(positions, default=0),
            "embedding_calls": sum(row.get("resource_kind") == "embedding" for row in rows),
            "known_tokens": sum(row.get("prompt_tokens", 0) + row.get("completion_tokens", 0) for row in rows),
            "failed_calls": sum(row.get("status") != "success" for row in rows),
            "invalid_output_warnings": (sum("invalid JSON/fields" in line for line in log.read_text(encoding="utf-8").splitlines())
                                        if log and log.exists() else 0)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="本次 Memory 实际运行目录")
    parser.add_argument("--log", type=Path, help="控制台日志，用于统计无效模型输出警告")
    parser.add_argument("--journal", type=Path, help="追加真实进度的 JSONL 文件")
    parser.add_argument("--usage-log", type=Path, help="实验层重定向后的 usage.jsonl，可省略")
    parser.add_argument("--checkpoints", type=Path, help="实验层 snapshots 目录，可省略")
    args = parser.parse_args(argv)
    record = read(args.run, args.log, usage_log=args.usage_log, checkpoints=args.checkpoints)
    text = json.dumps(record, ensure_ascii=False)
    if args.journal:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        with args.journal.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
