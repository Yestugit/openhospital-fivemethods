"""Doctor format recovery preserves valid decisions and never invents a fallback."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mematlas.integrations.openhospital.application import load_doctor_runtime


@pytest.fixture
def doctor(tmp_path):
    cls, _ = load_doctor_runtime()
    agent = cls(config={"rsi_strict_outputs": True, "log_llm_prompts": True})
    agent._timeline = []
    agent.logger = SimpleNamespace(output_dir=tmp_path, write_event=lambda **event: None)
    agent.runtime_settings = SimpleNamespace(team_id="format-test")
    return agent


def fake_llm(doctor, responses):
    calls = []
    async def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=responses.pop(0), usage={"prompt_tokens": 10, "completion_tokens": 2})
    doctor.llm = SimpleNamespace(chat=chat, model="test-model")
    return calls


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows extended path regression")
@pytest.mark.asyncio
async def test_prompt_event_and_final_logs_survive_windows_max_path(doctor, tmp_path):
    from mematlas.integrations.openhospital.execution import _sdk_output_path
    from hospital_agent_sdk.event_logger import EventLogger
    # Reproduce a deeper path than the failed parallel checkpoint, then use
    # the real doctor prompt writer and the real SDK event/result writers.
    directory = tmp_path / ("root" * 25) / ("parallel" * 12) / "test" / "test_20261104_120239_131500"
    assert len(str(directory / "llm_prompts")) > 260
    doctor.logger = EventLogger(_sdk_output_path(directory))
    calls = fake_llm(doctor, ['{"examinations": ["CBC"], "reason": "unchanged"}'])
    result = await doctor._call_llm(prompt="complete clinical input", default={"examinations": [], "reason": ""},
                                  prompt_name="exam_item", patient_id="Patient_06479")
    assert result["examinations"] == ["CBC"] and len(calls) == 1
    doctor.logger.write_final_result({"patient_id": "Patient_06479", "diagnosis": ["recorded"]})
    doctor.logger.write_evaluation_result("Patient_06479", {"score": 1})
    prompt = doctor.logger.output_dir / "llm_prompts/exam_item_Patient_06479.txt"
    assert "complete clinical input" in prompt.read_text(encoding="utf-8")
    assert doctor.logger.events_path.is_file()
    assert doctor.logger.evaluation_results_path.is_file()
    assert doctor.logger.latest_final_result("Patient_06479")["diagnosis"] == ["recorded"]


@pytest.mark.asyncio
async def test_valid_decision_keeps_original_request_and_single_call(doctor):
    raw = '{"examinations": ["CBC"], "reason": "unchanged"}'
    calls = fake_llm(doctor, [raw])
    result = await doctor._call_llm(prompt="original clinical input", default={"examinations": [], "reason": ""},
                                    prompt_name="exam_item", patient_id="p1")
    assert result == json.loads(raw)
    assert len(calls) == 1 and "extra_body" not in calls[0]
    assert calls[0]["messages"][-1]["content"] == "original clinical input"
    assert len(doctor._timeline) == 1


@pytest.mark.asyncio
async def test_real_unescaped_quote_gets_bounded_constrained_repair(doctor):
    malformed = '{"examinations": ["C反应蛋白（CRP）"], "reason": "检测"C反应蛋白（CRP）”作为指标"}'
    repaired = json.dumps({"examinations": ["C反应蛋白（CRP）"], "reason": '检测"C反应蛋白（CRP）”作为指标'}, ensure_ascii=False)
    calls = fake_llm(doctor, [malformed, malformed, repaired])
    result = await doctor._call_llm(prompt="clinical input", default={"examinations": [], "reason": ""},
                                    prompt_name="exam_item", patient_id="p1")
    assert result["examinations"] == ["C反应蛋白（CRP）"]
    assert len(calls) == 3 and len(doctor._timeline) == 1
    for call in calls[1:]:
        assert call["extra_body"] == {"response_format": {"type": "json_object"}}
        assert "原始医生决策（只修复此内容）：\n" + malformed in call["messages"][-1]["content"]
        assert call["messages"][-1]["content"].startswith("修复下面的医生决策")
    logs = list((doctor.logger.output_dir / "llm_prompts").glob("*.txt"))
    assert len(logs) == 3 and any(malformed in f.read_text(encoding="utf-8") for f in logs)


@pytest.mark.asyncio
async def test_persistent_bad_output_pauses_without_learning_or_default(doctor):
    from agent.agent import DoctorOutputPaused
    calls = fake_llm(doctor, ["invalid"] * 4)
    with pytest.raises(DoctorOutputPaused, match="3 JSON repairs"):
        await doctor._call_llm(prompt="clinical input", default={"examinations": ["DO NOT USE"]},
                               prompt_name="exam_item", patient_id="p1")
    assert len(calls) == 4 and doctor._timeline == []
    assert len(list((doctor.logger.output_dir / "llm_prompts").glob("*.txt"))) == 4


@pytest.mark.asyncio
async def test_repair_cannot_replace_invalid_decision_with_empty_or_wrong_types(doctor):
    from agent.agent import DoctorOutputPaused
    echoed_wrapper = json.dumps({"task": "修复 JSON 格式", "original_response": "invalid"})
    calls = fake_llm(doctor, ["invalid", "{}", '{"examinations": "CBC"}', echoed_wrapper])
    with pytest.raises(DoctorOutputPaused):
        await doctor._call_llm(prompt="clinical input", default={"examinations": []},
                               prompt_name="exam_item", patient_id="p1")
    assert len(calls) == 4 and doctor._timeline == []


@pytest.mark.parametrize("raw,expected", [
    ('```json\n{"a": {"b": "}"}}\n```', {"a": {"b": "}"}}),
    ('Here is JSON: {"a": 1} Done.', {"a": 1}),
    ('{"a": 1} corrected: {"a": 2}', None),
    ('{"broken": "quote"oops", "nested": {"a": 1}}', None),
    ('[]', None),
    ('{"a": "truncated', None),
])
def test_parser_does_not_select_conflicting_or_nested_fragments(doctor, raw, expected):
    from agent.agent import parse_json_object
    assert parse_json_object(raw) == expected


@pytest.mark.asyncio
async def test_exhausted_format_repair_is_a_resumable_worker_pause(monkeypatch):
    from mematlas.integrations.openhospital import suite
    config = SimpleNamespace(methods=[SimpleNamespace(metadata={})])
    monkeypatch.setattr(suite.ExperimentConfig, "load", lambda path: config)
    class Runner:
        async def evaluate(self, *args, **kwargs):
            return [SimpleNamespace(failures=[{"error_type": "DoctorOutputPaused"}])]
    monkeypatch.setattr(suite, "ExperimentRunner", Runner)
    assert await suite.worker(SimpleNamespace(config="unused", source="saved-snapshot")) == 75
