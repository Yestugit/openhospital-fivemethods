"""Reproduce the adrenal panel failure without calling any model."""
import json
from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from mematlas.integrations.openhospital.application import load_doctor_runtime
from mematlas.integrations.openhospital.config import OPENHOSPITAL_ROOT


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(OPENHOSPITAL_ROOT / "hospital_service"))
    from hospital_service.service.exam import generator
    return generator


def payload(names):
    return json.dumps({"examination": "肾上腺功能检查", "indicators": [
        {"name": name, "value": f"{10 + i} ng/mL", "status": "abnormal" if i == 0 else "normal"}
        for i, name in enumerate(names)]}, ensure_ascii=False)


def parse(module, names, expected):
    return module.parse_generated_result(raw=payload(names), examination="肾上腺功能检查",
                                         expected_names=expected, model_name="offline")


def test_actual_adrenal_spacing_failure_preserves_values_status_and_catalog_order(generator):
    expected = ("促肾上腺皮质激素(血浆)", "皮质醇(凌晨8点)", "皮质醇(下午4点)", "醛固酮(卧位)", "醛固酮(直立位)")
    returned = ("促肾上腺皮质激素 （血浆）", "皮质醇 (凌晨 8 点)", "皮质醇 (下午 4 点)", "醛固酮 (卧位)", "醛固酮 (直立位)")
    result = parse(generator, returned, expected)
    assert list(result.result) == list(expected)
    assert list(result.result.values()) == [f"{10 + i} ng/mL" for i in range(5)]
    assert result.abnormal_indicators == expected[:1]
    rows = generator._extract_valid_indicator_rows(payload(returned), frozenset(expected))
    assert set(rows) == set(expected)


@pytest.mark.parametrize("name", ["皮质醇(下午8点)", "皮质醇(凌晨4点)", "皮质醇(血浆)", "游离皮质醇(凌晨8点)"])
def test_clinically_different_indicators_still_rejected(generator, name):
    with pytest.raises(ValueError, match="missing indicators"):
        parse(generator, [name], ("皮质醇(凌晨8点)",))


def test_alias_duplicates_and_catalog_collisions_are_rejected(generator):
    with pytest.raises(ValueError, match="duplicate indicators"):
        parse(generator, ["皮质醇(凌晨8点)", "皮质醇 （凌晨 8 点）"], ("皮质醇(凌晨8点)",))
    with pytest.raises(ValueError, match="missing indicators"):
        parse(generator, ["A  B"], ("AB", "A B"))


@pytest.mark.parametrize("failure", [500, 429, "timeout"])
def test_hospital_transient_failure_retries_same_payload_then_pauses(monkeypatch, failure):
    load_doctor_runtime()
    from hospital_agent_sdk import client
    from hospital_agent_sdk.llm import LLMRequestPaused
    bodies, sleeps = [], []
    def request(req, **kwargs):
        bodies.append(req.data)
        if failure == "timeout": raise TimeoutError("offline")
        raise HTTPError(req.full_url, failure, "fixture", {}, BytesIO(b"offline"))
    monkeypatch.setattr(client.urllib.request, "urlopen", request)
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    caller = client.ServiceHTTPClient("http://127.0.0.1:8001", 120)
    with pytest.raises(LLMRequestPaused):
        caller._request_json("POST", "/exam/results", {"items": ["肾上腺功能检查"]}, None)
    assert len(bodies) == 8 and len(set(bodies)) == 1
    assert sleeps == [2, 4, 8, 16, 30, 30, 30]


def test_hospital_permanent_error_is_not_retried(monkeypatch):
    load_doctor_runtime()
    from hospital_agent_sdk import client
    calls = []
    def request(req, **kwargs):
        calls.append(req)
        raise HTTPError(req.full_url, 401, "fixture", {}, BytesIO(b"offline"))
    monkeypatch.setattr(client.urllib.request, "urlopen", request)
    with pytest.raises(client.ServiceClientError):
        client.ServiceHTTPClient("http://127.0.0.1:8001", 120)._request_json("POST", "/exam/results", {}, None)
    assert len(calls) == 1


def test_hospital_third_retry_succeeds_without_replaying_doctor_decision(monkeypatch):
    load_doctor_runtime()
    from hospital_agent_sdk import client
    calls = []
    def request(req, **kwargs):
        calls.append(req.data)
        if len(calls) < 4: raise HTTPError(req.full_url, 500, "fixture", {}, BytesIO(b"offline"))
        return nullcontext(SimpleNamespace(read=lambda: b'{"results": {"real": "result"}}'))
    monkeypatch.setattr(client.urllib.request, "urlopen", request)
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    result = client.ServiceHTTPClient("http://127.0.0.1:8001", 120)._request_json("POST", "/exam/results", {}, None)
    assert result == {"results": {"real": "result"}} and len(set(calls)) == 1
