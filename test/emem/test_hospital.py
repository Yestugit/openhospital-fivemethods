"""Public OpenHospital lifecycle with original EMem graph/stores and a local API."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("igraph")
pytest.importorskip("transformers")

from mematlas import Memory, MethodServices
from mematlas.core.state_files import unpack_state
from mematlas.integrations.openhospital.bridge import OpenHospitalMemoryIntegration
from mematlas.methods.emem.utils import get_blob
from mematlas.test.emem.native_support import FixedEmbedding
from mematlas.test.emem.test_native import endpoint

SOURCE = Path(__file__).resolve().parents[2] / ".mematlas" / "upstream" / "emem"


def config(path, endpoint_url, executor="in_process"):
    return {"method": {"method_id": "emem"}, "scope": {"user_id": "hospital", "agent_id": "doctor"},
        "providers": {"llm": {"model": "gpt-4o-mini", "base_url": endpoint_url, "api_key_env": "EMEM_NATIVE_TEST_KEY"},
                      "embedding": {"model": "configured-embedding"}},
        "runtime": {"executor": executor, "operation_timeout_s": 30, "artifact_dir": str(path), "source_roots": {"emem": str(SOURCE)}}}


def native_snapshot(path):
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 4
    state = unpack_state(checkpoint["state"])
    return json.loads(get_blob(state["state"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("timing,delay", [("immediate", 0), ("batch", 0), ("delayed", 1)])
async def test_hospital_boundary_feedback_freeze_and_new_worker_restore(tmp_path, endpoint, monkeypatch, timing, delay):
    monkeypatch.setenv("EMEM_NATIVE_TEST_KEY", "local-test-placeholder")
    supplied = MethodServices(embedder=FixedEmbedding())
    memory = Memory(config(tmp_path / "first", endpoint[0]), services=supplied)
    hospital = OpenHospitalMemoryIntegration(memory, top_k=None, feedback_timing=timing, feedback_delay=delay)
    state = {"timeline": [{"kind": "message", "role": "user", "text": "I am allergic to penicillin."},
                          {"kind": "message", "role": "assistant", "text": "I will verify your medication allergy."}]}
    report = {"diagnosisAccuracy": 0.8, "examinationPrecision": 0.7, "treatmentOverallScore": 0.9}
    checkpoint = tmp_path / "checkpoint.json"
    try:
        await hospital.start()
        hospital.configure_training_batch(["p1", "p2"], batch_index=1)
        for patient in ("p1", "p2"):
            before = len(endpoint[1])
            await hospital.record_execution(patient_id=patient, case_state=state,
                                            final_result={"diagnosis": "Documented penicillin allergy"})
            assert len(endpoint[1]) == before
            await hospital.learn_case(patient_id=patient, case_state=state,
                final_result={"diagnosis": "Documented penicillin allergy"}, evaluation_report=report,
                reflection={"future_strategy": "Check allergy before choosing antibiotics"})
        await hospital.deliver_feedback(batch_end=True)
        backend = next(iter(memory._method.backends.values()))
        assert type(backend.engine).__module__ == "emem.EMem"
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
    count = len(endpoint[1])
    async with Memory(config(tmp_path / "second", endpoint[0], "subprocess"),
                      services=MethodServices(embedder=FixedEmbedding(deny=True))) as restored:
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
            original, loaded = native["native"][key], recovered["native"][key]
            assert original["sources"] == loaded["sources"]
            assert set(original["files"]) == set(loaded["files"])
            for name in original["files"]:
                assert get_blob(original["files"][name]["blob"]) == get_blob(loaded["files"][name]["blob"])
        async with restored.read_only():
            await restored.index()
        await restored.save(again)
        assert recovered == native_snapshot(again)
    assert len(endpoint[1]) == count
