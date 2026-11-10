"""Synthetic, explicitly labelled training feedback tests scheduling, not task quality."""
import json
import pytest

from mematlas import ExperimentConfig, ExperimentRunner
from test_expel_integration import config, episode, services


@pytest.mark.asyncio
async def test_labeled_custom_benchmark_compiles_only_at_final_training_episode(tmp_path):
    data = []
    for identity,success,task in [('failed',False,'Verify record'),('passed',True,'Verify record'),('other',True,'Other record')]:
        value = episode(identity, task=task, success=success)
        data.append(dict(case_id=identity,split='train',scope={'user_id':'alice'},query=task,
            episode=value.to_dict()))
    data.append(dict(case_id='frozen',query='New record',reference='fixture-answer',scope={'user_id':'alice'}))
    path=tmp_path/'labeled-fixture.json'
    path.write_text(json.dumps(data),encoding='utf-8')
    supplied = services()
    from mematlas.providers import CallableLLM
    supplied.objects['answer_llm']=CallableLLM(lambda messages,**kwargs:'fixture-answer')
    experiment = ExperimentConfig(methods=(config(tmp_path/'memory'),),
        benchmark={'kind':'custom','path':str(path)},protocol='experience_train_test',
        output_dir=str(tmp_path/'report'))
    runner = ExperimentRunner()
    assert runner.check(experiment,services=supplied)['ready']
    result=(await runner.evaluate(experiment,services=supplied))[0]
    assert not result.failures
    assert len(supplied.llm.calls)==2
    assert len(result.predictions)==1
    used=result.predictions[0].applied_memory
    assert used and any(i['memory_type']=='insight' for c in used for i in c.items)
    assert (tmp_path/'report/results.json').exists()
