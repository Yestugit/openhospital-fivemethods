"""The JSON compatibility boundary never silently accepts truncation/fallback."""
from copy import deepcopy
import json

import pytest
from pydantic import BaseModel
from openai.types.chat import ChatCompletion
from mematlas.methods.emem.structured import JSONTransport
from mematlas.providers.requests import LLMRequestPaused


class FilteredEDUs(BaseModel):
    selected_edus: list[str]


def request():
    candidates = ['中文 "检查" 与 \\ 路径', '{"nested":"原文"}']
    return {'model': 'controlled', 'max_tokens': 8192, 'response_format': FilteredEDUs,
            'messages': [{'role': 'system', 'content': 'Copy relevant EDUs exactly.'},
                         {'role': 'user', 'content': 'Query: 检查\n\nCandidate EDUs:\n' + json.dumps(candidates)}]}


def response(text, finish='stop'):
    return ChatCompletion(id='controlled', created=1, model='controlled', object='chat.completion',
                          choices=[{'index': 0, 'finish_reason': finish,
                                    'message': {'role': 'assistant', 'content': text}}],
                          usage={'prompt_tokens': 15, 'completion_tokens': 8, 'total_tokens': 23})


@pytest.mark.asyncio
async def test_unicode_wire_preserves_candidates_and_repairs_schema_and_membership(tmp_path):
    original = request()
    pristine = deepcopy(original)
    native = json.loads(original['messages'][-1]['content'].split('Candidate EDUs:\n')[1])
    replies = [response('```json\n{}\n```'), response('{"selected_edus":42}'),
               response('{"selected_edus":["invented"]}'),
               response(json.dumps({'selected_edus': native}, ensure_ascii=False))]
    sent = []
    async def send(params):
        sent.append(deepcopy(params))
        return replies[len(sent)-1]
    result = await JSONTransport(tmp_path/'responses.jsonl').invoke(original, send)
    assert result is replies[-1] and original == pristine
    assert len(sent) == 4 and all(p['response_format'] == {'type': 'json_object'} for p in sent)
    converted = sent[0]['messages'][1]['content'].split('Candidate EDUs:\n')[1]
    assert json.loads(converted) == native and '\\u' not in converted
    rows = [json.loads(line) for line in (tmp_path/'responses.jsonl').read_text(encoding='utf8').splitlines()]
    assert len({r['logical_request_id'] for r in rows}) == 1
    assert [r['json_attempt'] for r in rows] == [1, 2, 3, 4]
    assert all(r['response']['usage']['total_tokens'] == 23 for r in rows)
    assert rows[-1]['validation_error'] is None


@pytest.mark.asyncio
async def test_truncation_has_bounded_budget_and_cannot_become_successful_fallback(tmp_path):
    sent = []
    async def send(params):
        sent.append(deepcopy(params))
        return response('{"selected_edus":[]}', finish='length')
    with pytest.raises(LLMRequestPaused, match='3 repairs'):
        await JSONTransport(tmp_path/'responses.jsonl').invoke(request(), send)
    assert [p['max_tokens'] for p in sent] == [8192, 16384, 32768, 32768]
    assert len((tmp_path/'responses.jsonl').read_text(encoding='utf8').splitlines()) == 4


@pytest.mark.asyncio
async def test_transport_failure_does_not_spend_json_repair_budget(tmp_path):
    async def send(params):
        raise ConnectionError('controlled outage')
    with pytest.raises(ConnectionError):
        await JSONTransport(tmp_path/'responses.jsonl').invoke(request(), send)
    assert not (tmp_path/'responses.jsonl').exists()


def test_candidate_validation_preserves_native_matching_tolerance():
    candidate='The patient has a penicillin allergy.'
    accepted=response(json.dumps({'selected_edus':[candidate[:-1]]}))
    assert JSONTransport.validate(accepted,FilteredEDUs,[candidate]).selected_edus==[candidate[:-1]]
    with pytest.raises(ValueError,match='native matching'):
        JSONTransport.validate(response('{"selected_edus":["unrelated invented content"]}'),FilteredEDUs,[candidate])
