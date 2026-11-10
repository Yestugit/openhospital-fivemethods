"""The external demo environment evaluates actual actions, never model self-ratings."""
import pytest
from mematlas.methods.expel.example import execute_agent, environment_config


class Agent:
    def __init__(self, answers):
        self.answers = iter(answers)
    async def complete(self, messages, **kwargs):
        return next(self.answers)


@pytest.mark.asyncio
async def test_demo_accepts_single_bare_finish_like_upstream_parser():
    actual = await execute_agent('Find project A','A',{'A':'Kyoto'},
        Agent(['Action 1: Lookup[A]','Finish[Kyoto]']),budget=2)
    assert actual['success'] is True and actual['output']=='Kyoto'
    assert actual['calls'][-1]['response']=='Finish[Kyoto]'
    assert actual['events'][-1].content=='Action 2: Finish[Kyoto]'


@pytest.mark.asyncio
async def test_demo_does_not_turn_a_wrong_answer_into_success():
    actual = await execute_agent('Find project A','A',{'A':'Kyoto'},
        Agent(['Action 1: Finish[Oslo]']),budget=1)
    assert actual['success'] is False and actual['output']=='Oslo'


def test_example_uses_plain_rule_operations_and_public_provider_config(monkeypatch):
    monkeypatch.setenv('EXPEL_API_KEY','test-placeholder')
    cfg = environment_config()
    assert cfg.method.method_id=='expel'
    assert 'response_format' not in cfg.providers['llm']['options']
    assert cfg.providers['llm']['api_key_env']=='EXPEL_API_KEY'
