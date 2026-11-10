"""Read-only audit of saved demo results; never opens or changes a native database.

Run from the project root with the demo interpreter. The JSON output contains
checks of actual trajectories, Agent inputs, learned contributions and stocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from mematlas.core.provenance import json_sha256


def audit_saved_results(data_dir: Path) -> dict:
    reports = {}
    for file in sorted(data_dir.glob("*/session.json")):
        session = json.loads(file.read_text(encoding="utf-8"))
        method = session["method"]
        checkpoint = json.loads((file.parent / "checkpoint.json").read_text(encoding="utf-8"))
        digest = checkpoint.pop("payload_sha256")
        assert digest == json_sha256(checkpoint), f"{method}: corrupt checkpoint"
        episodes = {row["episode_id"]: row for row in checkpoint["history"]["episodes"].values()}
        events = checkpoint["history"]["events"]
        snapshot = session["snapshot"]
        inventory = [item for layer in snapshot["layers"] for item in layer["items"]]
        assert len(inventory) == snapshot["counts"]["stored"], method
        checked, chats, excluded, awaiting = 0, 0, [], []
        for turn in session["turns"]:
            if turn.get("source") == "verification_error" or turn.get("error"):
                excluded.append({"turn_id": turn["id"], "reason": turn.get("error") or turn["source"]})
                continue
            if not turn.get("learned"):
                awaiting.append(turn["id"])
                continue
            episode = episodes[turn["id"]]
            assert episode["task"] == turn["user"] and episode["output"] == turn["assistant"], (method, turn["id"])
            trajectory = [events[key] for key in episode["event_ids"]]
            assert [(row["role"], row["content"]) for row in trajectory] == [
                ("user", turn["user"]), ("assistant", turn["assistant"])], (method, turn["id"])
            if turn["source"] == "seed":
                assert episode["applied_memory"] == [], (method, "unused seed retrieval was credited")
            else:
                retrieval = turn["retrieval"]
                rows = retrieval.get("context") or retrieval.get("items", [])
                memory_text = "\n\n".join(str(row["content"]) for row in rows)
                agent_input = turn["agent_input"]
                assert memory_text == agent_input["memory_text"], (method, "retrieval was changed")
                assert memory_text in agent_input["messages"][0]["content"], (method, "memory absent from Agent input")
                assert episode["applied_memory"] == [retrieval], (method, "applied contribution changed")
                if method == "memrl":
                    assert type(episode["success"]) is bool and episode["success"] == turn["success"], method
                chats += 1
            checked += 1

        native = checkpoint["state"]
        details = {}
        if method == "mem0":
            records = native["core"]["memories"]
            assert len(records) == len(inventory), (method, "stock differs from saved memories")
            assert all(any(str(item["id"]).endswith(str(key)) for item in inventory) for key in records), method
            details = {"native_memories": len(records), "profile": "additive"}
        elif method == "memoryos":
            page_by_id = {}
            for state in native["instances"].values():
                for segment in state["mid_term_sessions"].values():
                    for page in segment["details"]:
                        page_by_id[page["page_id"]] = page
            mid_items = next(layer["items"] for layer in snapshot["layers"] if layer["id"] == "mid_term")
            assert len(mid_items) == len(page_by_id), (method, "mid-term page count differs")
            for item in mid_items:
                meta = item["metadata"]
                page = page_by_id[meta["page_id"]]
                assert meta["meta_info"] == page["meta_info"], (method, "invented displayed summary")
                if meta.get("presentation") == "stored_summary":
                    assert item["content"] == page["meta_info"].strip(), method
                assert page["user_input"] and page["agent_response"], method
            details = {"native_mid_term_pages": len(page_by_id), "stored_summaries_match": True}
        elif method == "memrl":
            actual_ids = {}
            for entry in native["services"].values():
                root = Path(entry["snapshot_root"])
                rows = json.loads((root / "cube" / "textual_memory.json").read_text(encoding="utf-8"))
                for row in rows:
                    actual_ids[str(row["id"])] = row
            assert len(actual_ids) == len(inventory), (method, "stock differs from native cube")
            for item in inventory:
                identity = item["metadata"]["memory_id"]
                assert identity in actual_ids, method
                native_q = actual_ids[identity]["payload"]["metadata"]["q_value"]
                assert abs(float(item["metadata"]["q_value"]) - float(native_q)) < 1e-12, (method, identity)
            details = {"native_memories": len(actual_ids), "displayed_q_matches_native": True,
                       "pending": sum(map(len, native["pending_batch"].values()))}
        reports[method] = {"run_id": file.parent.name, "saved_at": session["updated_at"],
                           "checked_learned_turns": checked, "checked_live_chats": chats,
                           "excluded": excluded, "unlearned_turn_ids": awaiting,
                           "inventory": snapshot["counts"], "native": details, "passed": True}
    assert set(reports) == {"mem0", "memrl", "memoryos"}, "Missing a completed method run"
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "demo" / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_saved_results(args.data_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
