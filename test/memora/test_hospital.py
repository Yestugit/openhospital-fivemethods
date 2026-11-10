"""Public OpenHospital lifecycle with the real pinned Memora/Chroma backend."""
import json
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("omegaconf")

from mematlas import Memory, MethodServices
from mematlas.core.state_files import unpack_state
from mematlas.integrations.openhospital.bridge import OpenHospitalMemoryIntegration
from mematlas.methods.memora.utils import get_blob
from mematlas.test.memora.native_support import ScriptedLLM, FixedEmbedding

SOURCE = Path(__file__).resolve().parents[2] / ".mematlas" / "upstream" / "memora"


def config(path, executor="in_process"):
    return {"method": {"method_id": "memora"}, "scope": {"user_id": "hospital", "agent_id": "doctor"},
        "runtime": {"executor": executor, "operation_timeout_s": 30, "artifact_dir": str(path), "source_roots": {"memora": str(SOURCE)}}}


def native_snapshot(path):
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 4
    state = unpack_state(checkpoint["state"])
    return json.loads(get_blob(state["state"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("timing,delay", [("immediate", 0), ("batch", 0), ("delayed", 1)])
async def test_hospital_boundary_feedback_freeze_and_new_worker_restore(tmp_path, timing, delay):
    supplied = MethodServices(llm=ScriptedLLM(), embedder=FixedEmbedding())
    memory = Memory(config(tmp_path / "first"), services=supplied)
    hospital = OpenHospitalMemoryIntegration(memory, top_k=None, feedback_timing=timing, feedback_delay=delay)
    state = {"timeline": [{"kind": "message", "role": "user", "text": "I am allergic to penicillin."},
                          {"kind": "message", "role": "assistant", "text": "I will verify your medication allergy."}]}
    report = {"diagnosisAccuracy": 0.8, "examinationPrecision": 0.7, "treatmentOverallScore": 0.9}
    checkpoint = tmp_path / "checkpoint.json"
    try:
        await hospital.start()
        hospital.configure_training_batch(["p1", "p2"], batch_index=1)
        for patient in ("p1", "p2"):
            before = len(supplied.llm.calls)
            await hospital.record_execution(patient_id=patient, case_state=state,
                                            final_result={"diagnosis": "Documented penicillin allergy"})
            assert len(supplied.llm.calls) == before
            await hospital.learn_case(patient_id=patient, case_state=state,
                final_result={"diagnosis": "Documented penicillin allergy"}, evaluation_report=report,
                reflection={"future_strategy": "Check allergy before choosing antibiotics"})
        await hospital.deliver_feedback(batch_end=True)
        backend = next(iter(memory._method.backends.values()))
        assert type(backend.agent).__module__ == "memora.core.memory"
        segments = [messages[0]["content"] for messages, options in supplied.llm.calls
                    if options.get("response_format", {}).get("json_schema", {}).get("name") == "SegmentationOutput"]
        assert any("Documented penicillin allergy" in prompt for prompt in segments)
        before = await memory._method.save()
        async with memory.read_only():
            contexts = await hospital.retrieve(patient_id="held-out", phase="diagnosis", mode="test",
                chat_history=[{"role": "user", "content": "penicillin allergy"}])
            assert contexts
        assert await memory._method.save() == before
        await memory.save(checkpoint)
        native = native_snapshot(checkpoint)
        integration_checkpoint = tmp_path / "hospital.json"
        expected_count = 1 if timing == "delayed" else 2
        await hospital.snapshot_to(integration_checkpoint, learned_count=expected_count)
        fingerprint = await hospital.state_fingerprint()
    finally:
        await hospital.close()
    # A different worker and artifact directory must restore original vectors;
    # both provider factories fail immediately if recovery tries inference.
    denied = MethodServices(llm=ScriptedLLM(deny=True), embedder=FixedEmbedding(deny=True))
    async with Memory(config(tmp_path / "second", "subprocess"), services=denied) as restored:
        restored_hospital = OpenHospitalMemoryIntegration(restored, top_k=None, feedback_timing=timing, feedback_delay=delay)
        assert await restored_hospital.restore_from(integration_checkpoint) == expected_count
        assert await restored_hospital.state_fingerprint() == fingerprint
        assert restored.stats()["submitted_units"] == expected_count
        assert len(restored_hospital.history.pending_feedback) == (1 if timing == "delayed" else 0)
        again = tmp_path / "recovered.json"
        await restored.save(again)
        recovered = native_snapshot(again)
        assert native["scopes"] == recovered["scopes"]
        for key in native["native"]:
            assert get_blob(native["native"][key]["collection"]) == get_blob(recovered["native"][key]["collection"])
        async with restored.read_only():
            await restored.index()
        await restored.save(again)
        assert recovered == native_snapshot(again)
