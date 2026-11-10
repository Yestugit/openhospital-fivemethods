"""Native HotpotQA protocol tests; model/search fixtures are explicit, not quality results."""
from dataclasses import replace
import json
from pathlib import Path
import pytest

from mematlas import Memory
from mematlas.core.plugin import MethodServices
from mematlas.core.evaluation import Prediction
from test_expel_integration import config, services


def dataset(path):
    rows=[dict(id=str(i),question=f'Question {i}?',answer='private answer',context={
        'title':['Public page'], 'sentences':[['Public evidence.']]},supporting_facts={'title':['SECRET SUPPORT']}) for i in range(5)]
    path.write_text(json.dumps(rows),encoding='utf-8')
    return path


def test_hotpotqa_factory_splits_scoring_and_conditions(tmp_path):
    from mematlas.benchmarks.factory import load_benchmark
    value=load_benchmark('hotpotqa',dataset(tmp_path/'data.json'))
    train,test=list(value.cases('train')),list(value.cases('test'))
    assert len(train)==2 and len(test)==3
    assert not {c.case_id for c in train}&{c.case_id for c in test}
    assert 'private answer' not in json.dumps(train[0].metadata)
    assert 'SECRET SUPPORT' not in json.dumps(train[0].metadata)
    assert not train[0].history and train[0].episode is None
    score=value.native_score([Prediction('x','The Private answer.','private answer')])
    assert score['success_rate']==1 and score['answer_f1']==1
    assert value.describe()['environment']=='wikipedia_docstore_online'


class Pages:
    def __init__(self):self.calls=[]
    async def search(self,term):
        self.calls.append(term)
        return 'First paragraph.\n\nThe verified answer is Kyoto.\n\nKyoto has temples.'


@pytest.mark.asyncio
async def test_docstore_summary_lookup_and_actual_finish_feedback():
    from mematlas.benchmarks.hotpotqa_env import HotpotQAEnvironment
    backend=Pages()
    env=HotpotQAEnvironment('Kyoto',backend,max_steps=7)
    assert 'cannot Lookup' in (await env.step('Lookup[answer]'))
    assert (await env.step('Search[city]'))=='First paragraph.'
    assert (await env.step('Lookup[Kyoto]')).startswith('(Result 1/2)')
    assert (await env.step('Lookup[Kyoto]')).startswith('(Result 2/2)')
    assert (await env.step('Lookup[Kyoto]'))=='No More Results'
    assert not env.success
    assert await env.step('Finish[Kyoto]')=='Answer is CORRECT'
    assert env.success and env.terminated


@pytest.mark.asyncio
async def test_network_failure_is_not_fabricated_as_task_failure():
    from mematlas.benchmarks.hotpotqa_env import HotpotQAEnvironment
    class Broken:
        async def search(self,term):raise RuntimeError('network unavailable')
    env=HotpotQAEnvironment('x',Broken())
    with pytest.raises(RuntimeError,match='network unavailable'):
        await env.step('Search[x]')


class Actions:
    def __init__(self):self.calls=[]
    async def complete(self,messages,**kwargs):
        self.calls.append(messages)
        return 'Thought: Use evidence.\nAction 1: Search[city]' if len(self.calls)==1 else 'Finish[Kyoto]'


@pytest.mark.asyncio
async def test_registered_qa_executes_tools_without_showing_answer_to_model(tmp_path):
    from mematlas.methods.expel.qa import ExpeLQA
    from mematlas.evaluation.qa import _qa_type
    from mematlas.core.evaluation import BenchmarkCase
    from mematlas import ExperimentConfig
    model=Actions(); supplied=services()
    supplied.objects.update(answer_llm=model,hotpotqa_docstore_factory=Pages)
    experiment=ExperimentConfig(methods=(config(tmp_path),),benchmark=None)
    qa=ExpeLQA(experiment,supplied)
    case=BenchmarkCase('hotpotqa:test:one','Where is the record?',reference='Kyoto',
        scope={'user_id':'alice'},metadata={'benchmark_id':'hotpotqa','split':'test','max_steps':7})
    async with Memory(config(tmp_path),services=supplied) as memory:
        assert _qa_type(memory.plugin) is ExpeLQA
        before=await memory._method.save()
        async with memory.read_only():
            episode=await qa(case,memory)
        assert episode.success and episode.output=='Kyoto'
        assert [e.kind.value for e in episode.events]==['action','tool_result','action','tool_result']
        assert episode.applied_memory
        assert 'Kyoto' not in json.dumps(model.calls[0])
        assert before==await memory._method.save()


def test_source_conversion_only_accepts_fixed_archive(tmp_path):
    from mematlas.methods.expel.hotpotqa_example import convert_upstream_sample
    path=tmp_path/'bad.joblib';path.write_bytes(b'untrusted pickle')
    with pytest.raises(ValueError,match='SHA'):
        convert_upstream_sample(path,tmp_path/'out.json')


@pytest.mark.asyncio
async def test_react_thought_turn_does_not_consume_an_environment_action(tmp_path):
    from mematlas.methods.expel.qa import ExpeLQA
    from mematlas.methods.expel.hotpotqa_qa import execute_hotpotqa
    from mematlas.core.evaluation import BenchmarkCase
    from mematlas import ExperimentConfig
    class Thinking:
        def __init__(self):self.calls=[]
        async def complete(self,messages,**options):
            self.calls.append((messages,options))
            return ['Thought 1: I need evidence.','Action 1: Search[city]','Action 2: Finish[Kyoto]'][len(self.calls)-1]
    model=Thinking(); supplied=MethodServices(objects={'answer_llm':model,'hotpotqa_docstore_factory':Pages})
    qa=ExpeLQA(ExperimentConfig(methods=(config(tmp_path),),benchmark=None),supplied)
    case=BenchmarkCase('train','Where?',reference='Kyoto',metadata={'split':'train','max_steps':2})
    result=await execute_hotpotqa(qa,case,None)
    assert result.success
    assert len(model.calls)==3
    assert all(options['stop']==['\n','\n\n'] for _,options in model.calls)
    assert 'Thought 1: I need evidence.' in result.events[0].content


@pytest.mark.asyncio
async def test_infrastructure_failure_preserves_partial_actor_output_without_false_episode(tmp_path):
    from mematlas.methods.expel.qa import ExpeLQA
    from mematlas.methods.expel.hotpotqa_qa import execute_hotpotqa
    from mematlas.core.evaluation import BenchmarkCase
    from mematlas import ExperimentConfig
    records=[]
    class Broken:
        async def search(self,term):raise RuntimeError('network unavailable')
    supplied=MethodServices(objects={'answer_llm':Actions(),'hotpotqa_docstore_factory':Broken,
                                     'hotpotqa_trace_sink':records.append})
    qa=ExpeLQA(ExperimentConfig(methods=(config(tmp_path),),benchmark=None),supplied)
    case=BenchmarkCase('train','Where?',reference='Kyoto',metadata={'split':'train','max_steps':2})
    with pytest.raises(RuntimeError,match='network unavailable'):
        await execute_hotpotqa(qa,case,None)
    assert len(records)==1 and records[0]['episode'] is None
    assert records[0]['status']=='infrastructure_error'
    assert records[0]['turns'][-1]['model_calls'][0]['raw_output']


@pytest.mark.asyncio
async def test_dynamic_retrieval_uses_new_thought_before_next_action_and_keeps_latest_thought(tmp_path):
    from mematlas.methods.expel.qa import ExpeLQA
    from mematlas.methods.expel.hotpotqa_qa import execute_hotpotqa
    from mematlas.core.evaluation import BenchmarkCase
    from mematlas import ExperimentConfig,MemoryContribution
    class Thinking:
        def __init__(self):self.index=0
        async def complete(self,messages,**options):
            self.index+=1
            return ['Thought 1: Find evidence.','Action 1: Search[city]','Action 2: Finish[Kyoto]'][self.index-1]
    class Used:
        def __init__(self):self.queries=[]
        async def search(self,query,**options):
            self.queries.append(options['extensions']['expel'])
            return MemoryContribution()
    supplied=MethodServices(objects={'answer_llm':Thinking(),'hotpotqa_docstore_factory':Pages})
    qa=ExpeLQA(ExperimentConfig(methods=(config(tmp_path),),benchmark=None),supplied)
    memory=Used()
    result=await execute_hotpotqa(qa,BenchmarkCase('test','Where?',reference='Kyoto',metadata={'split':'test','max_steps':2}),memory)
    assert result.success and len(memory.queries)==3
    assert memory.queries[1]['thought']=='Thought 1: Find evidence.'
    assert memory.queries[1]['last_step_type']=='thought'
    assert memory.queries[2]['thought']=='Thought 1: Find evidence.'


@pytest.mark.asyncio
async def test_dynamic_success_examples_replace_static_examples(tmp_path):
    from mematlas.methods.expel.qa import ExpeLQA
    from mematlas.methods.expel.hotpotqa_qa import execute_hotpotqa
    from mematlas.core.evaluation import BenchmarkCase
    from mematlas import ExperimentConfig,MemoryContribution
    class Memory:
        async def search(self,*args,**kwargs):
            return MemoryContribution(items=[{'content':'Learned successful trajectory.', 'memory_type':'successful_trial'}])
    model=Actions();supplied=MethodServices(objects={'answer_llm':model,'hotpotqa_docstore_factory':Pages})
    qa=ExpeLQA(ExperimentConfig(methods=(config(tmp_path),),benchmark=None),supplied)
    result=await execute_hotpotqa(qa,BenchmarkCase('test','Where?',reference='Kyoto',metadata={'split':'test','max_steps':2}),Memory())
    assert result.success
    prompt=json.dumps(model.calls[0])
    assert 'Learned successful trajectory.' in prompt
    assert 'Colorado orogeny' not in prompt


@pytest.mark.asyncio
async def test_environment_transitions_match_pinned_upstream_qaenv():
    import ast,re
    from typing import Tuple
    from mematlas.benchmarks.hotpotqa_env import HotpotQAEnvironment
    from mematlas.benchmarks.hotpotqa import normalize_answer
    source=Path(__file__).parent/'reference/native/expel_hotpotqa_env.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    class Base:
        def is_terminated(self):return self.terminated
        def is_truncated(self):return self.curr_step>self.max_steps
    class Explorer:
        def search(self,term):return 'Public summary.'
        def lookup(self,term):return 'Public detail.'
    def parse(text):
        match=re.match(r'^(\w+)\[(.+)\]$',text)
        return match.groups() if match else (None,None)
    scope={'BaseEnv':Base,'Tuple':Tuple,'DocstoreExplorer':lambda obj:obj,'Wikipedia':Explorer,
           'parse_action':parse,'EM':lambda a,b:normalize_answer(a)==normalize_answer(b)}
    exec(compile(tree,str(source),'exec'),scope)
    original=scope['QAEnv']('Question','Kyoto',max_steps=3,explorer=Explorer())
    class Backend:
        async def search(self,term):return 'Public summary.'
    adapted=HotpotQAEnvironment('Kyoto',Backend(),max_steps=3)
    for action in ['Invalid','Search[city]','Finish[the Kyoto]']:
        observation,reward,terminated,truncated,position=original.step(action)
        assert await adapted.step(action)==observation
        assert (adapted.success,adapted.terminated,adapted.truncated,adapted.curr_step)==(reward,terminated,truncated,position)
