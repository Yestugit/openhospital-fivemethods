"""Preserve benchmark evidence through the default QA path."""
import json
import re
from datetime import date

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, MemoryContribution, MethodServices
from mematlas.benchmarks import LoCoMoBenchmark, LongMemEvalBenchmark
from mematlas.core.models import content_text
from mematlas.evaluation.qa import DefaultQA
from mematlas.providers import CallableLLM


def locomo_file(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([{
        "sample_id": "people",
        "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob",
            "session_1_date_time": "1:00 pm on 1 January, 2024",
            "session_1": [
                {"speaker": "Alice", "text": "I visited Paris.", "dia_id": "D1:1"},
                {"speaker": "Bob", "text": "I visited Rome.", "dia_id": "D1:2",
                 "img_file": "rome.jpg"},
            ],
        },
        "qa": [{"question": "Where did Alice visit?", "answer": "Paris",
                "category": 4, "evidence": ["D1:1"]}],
    }]), encoding="utf-8")
    return path


def test_locomo_keeps_identity_dates_and_image_evidence(tmp_path):
    dataset = LoCoMoBenchmark(locomo_file(tmp_path), mode="image_text")
    first, second = next(iter(dataset.cases())).history
    assert content_text(first.content) == "Alice: I visited Paris."
    assert content_text(second.content) == "Bob: I visited Rome."
    assert (first.role, second.role) == ("user", "assistant")
    assert (first.event_id, second.event_id) == ("D1:1", "D1:2")
    assert first.timestamp.date() == date(2024, 1, 1)
    assert first.extensions["speaker"] == "Alice"
    assert second.content[1]["type"] == "image"
    assert second.content[1]["uri"].endswith("rome.jpg")


@pytest.mark.asyncio
@pytest.mark.parametrize("method_id", ["mem0", "memoryos"])
async def test_default_qa_receives_named_people_from_real_method_storage(tmp_path, method_id):
    prompts = []

    def answer(messages, **options):
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        if "Alice: I visited Paris." in prompt:
            return "Paris"
        return "unavailable"

    params = {"implementation_mode": "offline"}
    supplied = MethodServices(objects={"answer_llm": CallableLLM(answer)})
    if method_id == "mem0":
        from mematlas.test.mem0.native_support import require_native, services
        require_native()
        params = {"infer": False, "embedding_dims": 16, "threshold": 0}
        supplied = services()
        supplied.objects["answer_llm"] = CallableLLM(answer)
    experiment = ExperimentConfig.from_dict({
        "methods": [{"method": {"method_id": method_id, "params": params},
                     "runtime": {"artifact_dir": str(tmp_path / "memory")}}],
        "benchmark": {"kind": "locomo", "path": str(locomo_file(tmp_path))},
        "output_dir": str(tmp_path / "report"),
    })
    result = (await ExperimentRunner().evaluate(experiment,
        services=supplied))[0]
    assert not result.failures, result.failures
    assert result.predictions[0].output == "Paris"
    assert result.predictions[0].applied_memory
    assert len(prompts) == 1


@pytest.mark.asyncio
async def test_default_qa_uses_benchmark_question_date_without_reference_leakage(tmp_path):
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps([
        {"question_id": f"question-{i}", "question_type": "temporal-reasoning",
         "question": "How many days ago did I visit Paris?", "question_date": question_date,
         "answer": "private-reference-for-scoring-only", "haystack_sessions": [],
         "answer_session_ids": ["private-evidence-for-scoring-only"]}
        for i, question_date in enumerate(("2024-01-03", "2024-02-03"))
    ]), encoding="utf-8")
    prompts = []

    def answer(messages, **options):
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        question_date = re.search(r"Question date: (\d{4}-\d{2}-\d{2})", prompt)
        if question_date is None:
            return "unavailable"
        return str((date.fromisoformat(question_date[1]) - date(2024, 1, 1)).days)

    reader = DefaultQA(None, MethodServices(objects={"answer_llm": CallableLLM(answer)}))
    contribution = MemoryContribution(items=[{"content": "Visited Paris on 2024-01-01."}])
    class RetrievedMemory:
        async def search(self, query, **options):
            return contribution

    predictions = [await reader(case, RetrievedMemory()) for case in LongMemEvalBenchmark(path).cases()]
    assert [prediction.output for prediction in predictions] == ["2", "33"]
    assert all(prediction.applied_memory == [contribution] for prediction in predictions)
    assert all("private-reference-for-scoring-only" not in prompt and
               "private-evidence-for-scoring-only" not in prompt for prompt in prompts)
