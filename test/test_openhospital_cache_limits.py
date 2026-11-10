"""Stable-prefix caching and stopping after sealed clinical checkpoints."""
import json
from types import SimpleNamespace

import pytest

from mematlas.integrations.openhospital.application import load_doctor_runtime
from mematlas.integrations.openhospital.progress import HospitalPaused, HospitalProgress


def test_dynamic_retrieval_does_not_invalidate_short_term_cache():
    load_doctor_runtime()
    from agent.prompt import llm_messages, format_prompt, NEXT_ACTION_PROMPT
    short = '[Short-term conversation]\n' + 'Clinical evidence with "quotes", {braces}, Unicode 病例\n' * 150
    requests = []
    for index in range(2):
        prompt = format_prompt(NEXT_ACTION_PROMPT, {"memory_notes": [short, f"[Historical memory]\nDynamic {index}"],
            "chat_history": [f"question {index}"], "examinations": {"result": index}})
        messages = llm_messages("unchanged system", prompt, cache_prefix=True)
        assert len(messages) == 3
        assert ''.join(block['text'] for row in messages[1:] for block in row['content']) == prompt
        assert 'Dynamic' not in messages[1]['content'][0]['text']
        requests.append(messages)
    assert requests[0][1] == requests[1][1]
    assert requests[0][2] != requests[1][2]
    other = format_prompt(NEXT_ACTION_PROMPT, {"memory_notes": [short + 'new case'], "chat_history": [], "examinations": {}})
    assert llm_messages("unchanged system", other, cache_prefix=True)[1] != requests[0][1]


@pytest.mark.asyncio
async def test_only_repeated_exam_phases_enable_cache_and_counts_are_per_patient():
    cls, _ = load_doctor_runtime()
    from agent.prompt import format_prompt, EXAM_ITEM_PROMPT
    prompt = format_prompt(EXAM_ITEM_PROMPT, {"memory_notes": ['[Short-term conversation]\n' + 'evidence ' * 800],
        "chat_history": [], "examinations": {}, "category": 'blood', "exam_items": ['CBC']})
    calls = []
    async def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='{}', usage={})
    agent = cls(config={"prompt_cache": "ephemeral"})
    agent.llm = SimpleNamespace(chat=chat)
    for phase, patient in [('exam_item','p1'),('exam_item','p1'),('exam_item','p1'),('exam_item','p2'),
                           ('exam_category','p1'),('exam_category','p1'),('department','p1'),('disease_and_treatment','p1')]:
        await agent._llm_text(prompt, system_prompt='system', temperature=.2, prompt_name=phase, patient_id=patient)
    assert [len(call['messages']) for call in calls] == [3,3,3,3,3,3,3,3]


def test_formal_stop_targets_match_complete_batches_and_required_tests(tmp_path):
    from mematlas.integrations.openhospital.suite import make_experiment, validate_stop_targets, CONDITIONS
    configs = {c: make_experiment(tmp_path,c,embedding='api') for c in CONDITIONS}
    targets = {'repeated': {'training':100,'tests':0}, 'novel_same_department': {'training':100,'tests':0},
               'novel_departments': {'training':50,'tests':100}}
    assert validate_stop_targets(configs,targets) == targets
    for bad in [{'repeated':{'training':99,'tests':0}}, {'novel_departments':{'training':50,'tests':50}},
                {'repeated':{'training':True,'tests':0}}, {'unknown':{'training':100,'tests':0}}]:
        with pytest.raises(ValueError): validate_stop_targets(configs,bad)




@pytest.mark.asyncio
async def test_operational_stop_keeps_sealed_training_and_heldout_curves(tmp_path, monkeypatch):
    from mematlas import BatchResult
    from mematlas.integrations.openhospital.rsi import _HospitalRunner
    from mematlas.evaluation.streams import _ExperienceStreamPlan
    from mematlas.evaluation.artifacts import recovery_checkpoint
    from mematlas.test.evaluation.test_openhospital_adaptation import bridge_for, STATE, FINAL, REPORT
    monkeypatch.delenv('OPH_STOP_FILE',raising=False)
    monkeypatch.setenv('OPH_STOP_TRAIN_AFTER','1')
    monkeypatch.setenv('OPH_STOP_TEST_AFTER','4')
    plan = _ExperienceStreamPlan(('p1','p2','p3'),('t1','t2'),(),batch_count=3,
        checkpoints=(0,1,2,3),training_batches=(('p1',),('p2',),('p3',)))
    progress = HospitalProgress(tmp_path/'evaluation',plan)
    bridge,_ = bridge_for(tmp_path/'memory')
    calls=[]
    async def train(ids,memory,index):
        progress.check_limits()
        memory.configure_training_batch(ids,batch_index=index)
        for patient in ids:
            await memory.learn_case(patient_id=patient,case_state=STATE,final_result=FINAL,evaluation_report=REPORT,reflection={})
            progress.training[patient]={"report":REPORT}
            calls.append(patient)
        return BatchResult(tuple(memory.history.result(memory._occurrence(p),scope=memory.scope) for p in ids))
    async def test(ids,memory,checkpoint,split):
        progress.check_limits()
        for patient in ids:
            progress.tests[f'{checkpoint}-{patient}']={}
            calls.append(f'{checkpoint}-{patient}')
        return {'case_reports':[{'patient_id':p,'department':'blood','report':REPORT} for p in ids]}
    runner=_HospitalRunner(plan,bridge,tmp_path/'run',train,test)
    with pytest.raises(HospitalPaused,match='Stop target reached'): await runner.run()
    assert calls == ['0-t1','0-t2','p1','1-t1','1-t2']
    assert recovery_checkpoint(tmp_path/'run')[0] == 1
    assert [row['checkpoint'] for row in runner._curve if row.get('split')=='primary_test'] == [0,1]
    assert any(row.get('batch_index')==1 for row in runner._curve)
    # A resumed worker with already reached limits pauses before any clinical action.
    from mematlas.integrations.openhospital.execution import HospitalExecution
    execution=HospitalExecution(plan=plan,output_dir=tmp_path/'resumed',source_dir=None,agent_class=None,
        doctor_config=None,runtime_builder=None,service_urls={},team_id='unused')
    execution.progress.training=progress.training
    execution.progress.tests=progress.tests
    with pytest.raises(HospitalPaused): await execution.train_batch(('p2',),None,2)
    with pytest.raises(HospitalPaused): await execution.test_batch(('t1',),None,1,'primary_test')
