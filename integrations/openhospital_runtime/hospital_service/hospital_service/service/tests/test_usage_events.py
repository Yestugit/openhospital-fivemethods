import json
from hospital_service.service.usage_events import CallUsageStore, record_model_call


def test_interrupted_journal_keeps_valid_records_and_unknown_cost(tmp_path):
    path = tmp_path / "calls.jsonl"
    store = CallUsageStore(path)
    with store.scope(dataset_key="train", team_id="team", component="judge",
                     accounting_context={"stage":"training", "case_id":"visit:1"}):
        record_model_call({"prompt_tokens":10,"completion_tokens":2},model="model",endpoint="https://example.test")
    with path.open("ab") as stream:
        stream.write(b'{"incomplete')
    restored = CallUsageStore(path)
    rows = restored.records(dataset_key="train", team_id="team")
    assert len(rows) == 2 and rows[0]["stage"] == "training" and rows[0]["case_id"] == "visit:1"
    assert not rows[1]["usage_known"]
    again = CallUsageStore(path)
    assert again.records(dataset_key="train", team_id="team") == rows


def test_examination_repairs_and_failed_calls_are_durable(tmp_path):
    store = CallUsageStore(tmp_path / "usage.jsonl")
    with store.scope(dataset_key="train", team_id="one", component="examination", patient_id="p"):
        record_model_call({"prompt_tokens": 10, "completion_tokens": 2}, model="m", endpoint="https://example.test/v1")
        record_model_call({"prompt_tokens": 12, "completion_tokens": 3}, model="m", endpoint="https://example.test/v1", attempt=2)
        record_model_call(None, model="m", endpoint="https://example.test/v1", status="failed", attempt=3)
    rows = store.records(dataset_key="train", team_id="one")
    assert len(rows) == 3 and sum(row["prompt_tokens"] for row in rows) == 22
    assert all(row["phase"] == "examination" for row in rows)
    assert not rows[-1]["usage_known"] and rows[-1]["status"] == "failed"
    restored = CallUsageStore(tmp_path / "usage.jsonl")
    assert restored.records(dataset_key="train", team_id="one") == rows
    assert not restored.records(dataset_key="train", team_id="other")


def test_chat_retry_preserves_paid_invalid_response(monkeypatch):
    from hospital_service.service.llm_client import OpenAICompatibleChatClient
    from contextlib import nullcontext
    from unittest.mock import Mock
    import urllib.request
    store = CallUsageStore()
    replies = [dict(usage={"prompt_tokens": 10, "completion_tokens": 2}, choices=[]),
               dict(usage={"prompt_tokens": 20, "completion_tokens": 3}, choices=[{"message": {"content": "ok"}}])]
    def respond(*args, **kwargs):
        return nullcontext(Mock(read=lambda: json.dumps(replies.pop(0)).encode()))
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    monkeypatch.setattr("hospital_service.service.llm_client.time.sleep", lambda _: None)
    with store.scope(dataset_key="train", team_id="one", component="examination"):
        result = OpenAICompatibleChatClient(model_api="https://example.test/v1", api_key="test", model="m",
                                             timeout_seconds=1, retries=2)._chat_sync(messages=[], temperature=0.1, extra_body=None)
    assert result.content == "ok"
    assert sum(row["prompt_tokens"] for row in store.records(dataset_key="train", team_id="one")) == 30
