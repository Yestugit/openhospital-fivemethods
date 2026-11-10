"""Both benchmark adapters use the unmodified DefaultQA and preserve sources."""
import pytest

from mematlas.test.a_mem.check_benchmarks import check


@pytest.mark.asyncio
async def test_benchmark_adapters_complete_three_locomo_and_one_longmemeval_fixture_cases(tmp_path):
    result = await check(tmp_path / "run")
    assert result["official_benchmark_acceptance"] == "pending"
    assert result["results"]["locomo"]["cases"] == 3
    assert result["results"]["longmemeval"]["cases"] == 1


def test_multiple_longmemeval_scopes_are_rejected_before_running(tmp_path):
    from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig
    from mematlas.core.files import atomic_write_json
    from mematlas.methods.a_mem import AMemConfig
    from mematlas.test.a_mem.check_benchmarks import fixture, FixtureReader
    from mematlas.test.a_mem.fixtures import services
    data = fixture("longmemeval")
    data.append({**data[0], "question_id": "other-scope"})
    path = tmp_path / "synthetic.json"
    atomic_write_json(path, data)
    supplied = services()
    supplied.objects["answer_llm"] = FixtureReader()
    experiment = ExperimentConfig(methods=(MemAtlasConfig(method=AMemConfig()),),
        benchmark={"kind": "longmemeval", "path": str(path)}, test_limit=2)
    report = ExperimentRunner().check(experiment, services=supplied)
    assert not report["ready"]
    assert "at most 1 scope" in str(report)
