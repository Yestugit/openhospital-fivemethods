import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import json_sha256
from mematlas.integrations.openhospital.efficiency import make_config, ROOT, METHODS
from mematlas.integrations.openhospital.planning import build_plan
from mematlas.integrations.openhospital.config import HospitalOptions

from mematlas.integrations.openhospital import parallel



def suite(tmp_path):
    settings={'identity':'parallel-test','smoke':True,'bolt_port':18687,'hospital_port':18001,'embedding_port':18766,
              'embedding_python':str(ROOT/'.venv-embedding/Scripts/python.exe'),'embedding_device':'cpu',
              'model_dir':str(ROOT/'models/Qwen3-Embedding-0.6B')}
    m={'settings':settings,'source_signature':{},'source_repairs':[],'model_environment':{},'jobs':{}}
    for method in METHODS:
        c=make_config(tmp_path,method,settings);p=tmp_path/method/'attempt-001/config.json';atomic_write_json(p,c.to_dict())
        m['jobs'][method]={'status':'pending','position':0,'attempt':1,'config':str(p),'output':c.output_dir,'source':None}
    plan=build_plan(HospitalOptions.from_experiment(c),checkpoints=(),seed=0)
    m['case_ids']=list(plan.plan.train_ids);m['case_ids_sha256']=json_sha256(m['case_ids'])
    return m


def test_isolated_ports_scopes_config_and_original_cohort(tmp_path,monkeypatch):
    m=suite(tmp_path);monkeypatch.setattr(parallel.ExperimentRunner,'check',lambda *_:{'ready':True})
    before=copy.deepcopy(m)
    a=parallel.prepare_controller(tmp_path,m,'emem',[0,1]);b=parallel.prepare_controller(tmp_path,m,'memora',[2,3])
    aa=json.loads(a.read_text());bb=json.loads(b.read_text())
    assert aa['settings']['embedding_port']!=bb['settings']['embedding_port']
    assert aa['settings']['hospital_port']!=bb['settings']['hospital_port']
    assert aa['case_ids']==bb['case_ids']==before['case_ids']
    ca=parallel.ExperimentConfig.load(m['jobs']['emem']['config']);cb=parallel.ExperimentConfig.load(m['jobs']['memora']['config'])
    assert ca.methods[0].scope!=cb.methods[0].scope
    assert ca.benchmark['world_dir']!=cb.benchmark['world_dir']
    assert ca.methods[0].runtime.artifact_dir!=cb.methods[0].runtime.artifact_dir
    assert ca.methods[0].providers['embedding']['base_url']!=cb.methods[0].providers['embedding']['base_url']
    assert m['jobs']['memmachine']==before['jobs']['memmachine']
    # Restore an existing controller without rewriting its world/config/attempt.
    aa['jobs']['emem'].update(status='paused',attempt=2,source='snapshot')
    atomic_write_json(a,aa)
    assert parallel.prepare_controller(tmp_path,m,'emem',[9])==a
    recovered=json.loads(a.read_text())
    assert recovered['jobs']['emem']['attempt']==2 and recovered['settings']['cpu_affinity']==[0,1]


def test_transition_refuses_existing_world_or_wrong_schedule(tmp_path,monkeypatch):
    m=suite(tmp_path);monkeypatch.setattr(parallel.ExperimentRunner,'check',lambda *_:{'ready':True})
    out=Path(m['jobs']['emem']['output']);out.mkdir(parents=True);(out/'execution.json').write_text('{}')
    with pytest.raises(ValueError,match='existing hospital world'):parallel.prepare_controller(tmp_path,m,'emem',[])
    m['case_ids_sha256']='different'
    with pytest.raises(ValueError,match='case order'):parallel.prepare_controller(tmp_path,m,'memora',[])


@pytest.mark.asyncio
async def test_failed_controller_does_not_cancel_other_and_parent_is_only_suite_writer(tmp_path,monkeypatch):
    m=suite(tmp_path)
    for method in METHODS[:3]:m['jobs'][method].update(status='completed',position=2,checkpoint_cleanup={'status':'completed'})
    atomic_write_json(tmp_path/'suite.json',m)
    monkeypatch.setattr(parallel,'source_signature',lambda:{})
    monkeypatch.setattr(parallel,'model_environment',lambda:{})
    monkeypatch.setattr(parallel,'terminal_status',lambda *_:None)
    monkeypatch.setattr(parallel,'report',lambda *_:None)
    def prepare(run,manifest,method,affinity):
        p=run/method/'controller-suite.json';atomic_write_json(p,manifest);return p
    monkeypatch.setattr(parallel,'prepare_controller',prepare)
    processes={};polls=[]
    class Process:
        pid=__import__('os').getpid()
        def __init__(self,method):self.method=method;self.poll=0;self.waited=False
        @property
        def returncode(self):
            self.poll+=1
            if self.method=='memora' and self.poll<3:return None
            p=tmp_path/self.method/'controller-suite.json';d=json.loads(p.read_text());j=d['jobs'][self.method]
            j.update(status='paused' if self.method=='emem' else 'completed',position=2,
                     checkpoint_cleanup={'status':'completed'} if self.method=='memora' else {})
            atomic_write_json(p,d)
            return 75 if self.method=='emem' else 0
        async def wait(self):self.waited=True;return 75 if self.method=='emem' else 0
    async def spawn(*command,**kwargs):
        method=command[-1];processes[method]=Process(method);return processes[method]
    async def sleep(_):polls.append(1)
    async def terminate(process):pytest.fail('One method failure cancelled the other')
    monkeypatch.setattr(parallel.asyncio,'create_subprocess_exec',spawn)
    monkeypatch.setattr(parallel.asyncio,'sleep',sleep)
    monkeypatch.setattr(parallel,'terminate_owned_process',terminate)
    assert await parallel.execute_parallel(SimpleNamespace(run_dir=str(tmp_path)))==75
    state=json.loads((tmp_path/'suite.json').read_text())
    assert state['jobs']['emem']['status']=='paused' and state['jobs']['memora']['status']=='completed'
    assert processes['memora'].waited and polls
