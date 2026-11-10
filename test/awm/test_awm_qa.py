from mematlas.evaluation.validation import submit_episode
from dataclasses import replace
import json
import pytest
from mematlas import Memory,ExperimentConfig,MemoryEpisode
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.plugin import MethodServices
from awm_fixtures import config,services,episode


class Browser:
    def __init__(self):self.actions=[];self.closed=False
    async def reset(self):return {'observation':'[textbox 1] Search [button 2] Go','done':False}
    async def step(self,action):
        self.actions.append(action)
        return {'observation':'Result found','done':True,'success':True}
    async def close(self):self.closed=True


class Actor:
    def __init__(self):self.calls=[]
    async def complete(self,messages,**kwargs):
        self.calls.append(messages);return "Use the search box.\n```fill('1', 'item')```"


class Mind2WebActor:
    def __init__(self):self.calls=[]
    async def complete(self,messages,**kwargs):
        self.calls.append(messages);return 'CLICK [143]'


class OneStepMind2Web:
    async def reset(self):return {'observation':'<a id=143> NBA </a>','done':False}
    async def step(self,action):return {'observation':'done','done':True,'success':action=='CLICK [143]'}
    async def close(self):pass


@pytest.mark.asyncio
async def test_registered_qa_requires_environment_and_retains_real_web_trace(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    from mematlas.evaluation.qa import _qa_type
    actor=Actor();browser=Browser();factories=[]
    def factory(task,options):factories.append((task,options));return browser
    supplied=services();supplied.objects.update(answer_llm=actor,awm_environment_factory=factory)
    qa=AWMQA(ExperimentConfig(methods=(config(tmp_path,profile='webarena'),),benchmark=None),supplied)
    case=BenchmarkCase('q','Find item',reference='SECRET GOLD ACTION',scope={'user_id':'alice'},
        metadata={'awm':{'website':'shop','template_id':1}})
    async with Memory(config(tmp_path,profile='webarena'),services=supplied) as memory:
        assert _qa_type(memory.plugin) is AWMQA
        before=await memory._method.save()
        async with memory.read_only():result=await qa(case,memory)
        assert result.success is True and result.events and result.applied_memory
        assert 'SECRET GOLD ACTION' not in json.dumps(actor.calls)
        assert 'SECRET GOLD ACTION' not in json.dumps(factories)
        assert before==await memory._method.save()
        assert browser.closed and browser.actions==["fill('1', 'item')"]
        await submit_episode(memory, result)
        assert supplied.llm.calls


@pytest.mark.asyncio
async def test_mind2web_qa_requests_native_literal_action_format(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    actor=Mind2WebActor();supplied=services()
    supplied.objects.update(answer_llm=actor,awm_environment_factory=lambda task,options:OneStepMind2Web())
    cfg=config(tmp_path,profile='mind2web')
    qa=AWMQA(ExperimentConfig(methods=(cfg,),benchmark=None),supplied)
    case=BenchmarkCase('q','Open the NBA page',scope={'user_id':'alice'},metadata={'awm':{
        'website':'shop','domain':'sports','subdomain':'basketball','max_steps':1}})
    async with Memory(cfg,services=supplied) as memory:
        result=await qa(case,memory)
    prompt=actor.calls[0][0]['content']
    assert 'CLICK [143]' in prompt and 'Do not omit the brackets.' in prompt
    assert 'backticks, or code block' in prompt
    assert 'CLICK 143' in prompt and 'CLICK [id=143]' in prompt
    assert result.success is True and result.events[-2].content=='CLICK [143]'


@pytest.mark.asyncio
async def test_missing_browser_fails_instead_of_plain_qa(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    supplied=services();supplied.objects['answer_llm']=Actor()
    qa=AWMQA(ExperimentConfig(methods=(config(tmp_path),),benchmark=None),supplied)
    case=BenchmarkCase('q','question',metadata={'awm':{'website':'shop','domain':'Shopping','subdomain':'General'}})
    with pytest.raises(ValueError,match='awm_environment_factory'):await qa(case,None)


def test_prepare_episode_only_copies_declared_metadata_and_tail(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    e=episode('one')
    case=BenchmarkCase('q','question',reference='SECRET',metadata={'awm':{'website':'shop'}})
    changed=AWMQA.prepare_episode(case,e,last_for_scope=True)
    assert changed.extensions['awm']['end_of_training'] is True
    assert changed.events==e.events and 'SECRET' not in json.dumps(changed.to_dict())


def test_plain_conversation_cases_fail_qa_preflight(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    with pytest.raises(ValueError,match='web navigation'):
        AWMQA.validate_cases(ExperimentConfig(methods=(config(tmp_path),),benchmark=None),[],[BenchmarkCase('q','chat?')])


def test_webarena_inference_preflight_requires_real_template_id(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    case=BenchmarkCase('q','Find item',metadata={'awm':{'website':'shop'}})
    with pytest.raises(ValueError,match='template_id'):
        AWMQA.validate_cases(ExperimentConfig(methods=(config(tmp_path,profile='webarena'),),benchmark=None),[],[case])


@pytest.mark.asyncio
async def test_live_qa_cannot_label_generated_actions_as_canonical(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    supplied=services();supplied.objects.update(answer_llm=Actor(),awm_environment_factory=lambda task,options:Browser())
    cfg=config(tmp_path,profile='webarena',induction_mode='offline')
    qa=AWMQA(ExperimentConfig(methods=(cfg,),benchmark=None),supplied)
    case=BenchmarkCase('q','Find item',metadata={'awm':{'website':'shop','template_id':1,'canonical':True}})
    async with Memory(cfg,services=supplied) as memory:
        result=await qa(case,memory)
        assert result.extensions['awm']['canonical'] is False
        with pytest.raises(Exception,match='canonical'):await submit_episode(memory, result)


def test_actual_public_preflight_uses_memory_extensions_contract(tmp_path):
    from mematlas.evaluation.validation import validate_cases
    from mematlas.core.registry import default_registry
    from mematlas import MemAtlasConfig
    case=BenchmarkCase('q','Find item',scope={'user_id':'alice'},metadata={
        'memory_extensions':{'awm':{'website':'shop'}},'awm':{'template_id':1}})
    value=ExperimentConfig(methods=(config(tmp_path,profile='webarena'),),benchmark=None)
    validate_cases(value,default_registry().get('awm'),[],[case],None)


def test_mixed_profiles_fail_clearly_instead_of_using_first_profile(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    value=ExperimentConfig(methods=(config(tmp_path/'a',profile='webarena'),config(tmp_path/'b')),benchmark=None)
    with pytest.raises(ValueError,match='mixed AWM profiles'):
        AWMQA.validate_cases(value,[],[])
