"""ExpeL's benchmark-side ReAct execution; Memory never executes tools or scores answers."""
from __future__ import annotations
import re
from uuid import uuid4

from mematlas.benchmarks.hotpotqa_env import HotpotQAEnvironment, WikipediaDocstore
from mematlas.core.contributions import contribution_text
from mematlas.core.models import MemoryEpisode, MemoryEvent
from mematlas.core.telemetry import active_usage
from mematlas.providers.instrumentation import MeteredService
from .hotpotqa_prompts import FEWSHOTS, SYSTEM_INSTRUCTION


def parse_action(output: str,step: int) -> tuple[str,str]:
    """Single bracket action, including bare actions, follows upstream LLM_PARSER.

    Multiple actions are invalid. Unfinished closing brackets are repaired as in
    upstream; raw output is separately journaled.
    """
    actions=re.findall(r'\b\w+\[[^\]\n]+(?:\]|$)',output)
    if len(actions)!=1:return output,'N/A'
    action=actions[0]
    if not action.endswith(']'):action+=']'
    prefix=output[:output.find(actions[0])]
    prefix=re.sub(r'(?im)action\s*\d*\s*:\s*$','',prefix).strip()
    return ((prefix+'\n') if prefix else '')+f'Action {step}: {action}',action


async def execute_hotpotqa(qa,case,memory,*,reflections=()):
    partial={'case_id':case.case_id,'split':case.metadata.get('split'),'episode':None,'turns':[]}
    try:
        return await _execute_hotpotqa(qa,case,memory,reflections=reflections,partial=partial)
    except Exception as exc:
        sink=qa.services.object('hotpotqa_trace_sink')
        if sink:
            sink({**partial,'status':'infrastructure_error','error':{'type':type(exc).__name__,'message':str(exc)}})
        raise


async def _execute_hotpotqa(qa,case,memory,*,reflections,partial):
    if qa.llm is None:raise ValueError('HotpotQA requires an answer model')
    training=case.metadata.get('split')=='train'
    if not training and memory is None:raise ValueError('HotpotQA test needs Memory')
    factory=qa.services.object('hotpotqa_docstore_factory')
    backend=factory() if factory else WikipediaDocstore()
    partial['wikipedia_requests']=getattr(backend,'records',[])
    env=HotpotQAEnvironment(case.reference,backend,max_steps=case.metadata.get('max_steps',7))
    used,events,turns=[],[],partial['turns']
    thought=action=step_text=last_step_type=''
    trajectory=[]
    llm=qa.llm
    active=active_usage()
    if active is not None and not isinstance(llm,MeteredService):llm=MeteredService(llm,active[0],'llm')
    for step in range(1,env.max_steps+1):
        generated=[];model_calls=[]
        turn={'step':step,'model_calls':model_calls}
        turns.append(turn)  # Journal the in-progress turn even if its model or tool fails.
        for thought_turn in range(3):
            memory_text='';has_dynamic=False
            if not training:
                contribution=await memory.search(case.query,scope=case.scope,top_k=case.metadata.get('top_k'),
                    extensions={'expel':{'thought':thought,'action':action,'step':step_text,'last_step_type':last_step_type}})
                used.append(contribution);memory_text=contribution_text(contribution)
                has_dynamic=any(item.get('memory_type')=='successful_trial' for item in contribution.items)
            # Dynamic successful demonstrations replace static examples. With no retrieved
            # examples we explicitly retain source static fewshots as the fallback.
            prompt=('Examples:\n'+('' if has_dynamic else '\n\n'.join(FEWSHOTS))+
                '\n\nExperience references:\n'+memory_text+'\n\nPrevious reflections:\n'+'\n'.join(reflections)+
                '\n\nQuestion: '+str(case.query)+'\n'+'\n'.join(trajectory))
            messages=[{'role':'system','content':SYSTEM_INSTRUCTION+
                '\nGenerate only the next single Thought or Action line. Never generate Observations. '
                'Use Finish[answer] with only the concise answer. Current action budget: '+str(env.max_steps-step+1)},
                {'role':'user','content':prompt+'\n'+'\n'.join(generated)}]
            call={'messages':messages};model_calls.append(call)
            response=await llm.complete(messages,model=qa.model,temperature=0,max_tokens=400,stop=['\n','\n\n'])
            formatted,action=parse_action(str(response),step)
            generated.append(formatted)
            call['raw_output']=str(response)
            if action!='N/A' or '[' in str(response):break
            thought=str(response);last_step_type='thought'
        formatted='\n'.join(generated)
        turn.update(messages=messages,raw_output=str(response),action=action)
        observation=await env.step(action)  # Infrastructure errors propagate, never become false feedback.
        observation=f'Observation {step}: {observation}'
        events.extend([MemoryEvent('action',formatted,role='assistant'),MemoryEvent('tool_result',observation,role='tool')])
        trajectory.extend([formatted,observation]);step_text=formatted+'\n'+observation
        thought_match=re.search(r'(?im)^Thought\s*\d*\s*:\s*(.*)',formatted)
        thought=thought_match.group(0) if thought_match else thought
        last_step_type='observation'
        turn.update(observation=observation,terminated=env.terminated,truncated=env.truncated)
        if env.terminated or env.truncated:break
    episode=MemoryEpisode(episode_id=f'{case.case_id}:{uuid4().hex}',task=str(case.query),events=events,
        output=env.output,success=env.success,reward=float(env.success),scope=case.scope,applied_memory=used,
        feedback='Environment exact match: '+str(env.success)+'; '+('Finish' if env.terminated else 'step budget exhausted'),
        extensions={'expel':{'reflections':list(reflections)}})
    sink=qa.services.object('hotpotqa_trace_sink')
    if sink:
        sink({'case_id':case.case_id,'split':case.metadata.get('split'),'episode':episode.to_dict(),
              'turns':turns,'wikipedia_requests':getattr(backend,'records',[])})
    return episode
