"""AWM's native Mind2Web acting path, selected by the registered AWMQA.

The benchmark owns HTML, candidate filtering and scoring. The actor receives
only the current eligible observation and *past* canonical history. Actual
responses are retained separately for online induction, never replaced by gold.
"""
from __future__ import annotations

from copy import deepcopy

from mematlas.benchmarks.mind2web import PROTOCOL
from mematlas.benchmarks.mind2web_env import extract_action
from mematlas.core.contributions import contribution_text
from mematlas.core.models import MemoryEpisode, MemoryEvent, content_text
from mematlas.core.telemetry import active_usage
from mematlas.providers.instrumentation import MeteredService

SYSTEM = (
    'You are a large language model trained to navigate the web. Output the next action and wait '
    'for the next observation. Here is the action space:\n'
    '1. `CLICK [id]`: Click on an HTML element with its id.\n'
    '2. `TYPE [id] [value]`: Type a string into the element with the id.\n'
    '3. `SELECT [id] [value]`: Select a value for an HTML element by its id.'
)


def _token_counter(model):
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model or '')
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')
    def count(messages):
        return 3 + sum(3 + sum(len(encoding.encode(value)) for value in m.values())
                       for m in messages)
    return count, encoding.name


async def execute_mind2web(qa, case, memory) -> MemoryEpisode:
    benchmark = qa.services.object('mind2web_benchmark')
    if benchmark is None:
        raise ValueError('Mind2Web native QA requires its registered benchmark through ExperimentRunner')
    if qa.llm is None:
        raise ValueError('Mind2Web native QA requires an answer model')
    env = benchmark.environment(case.case_id)
    concrete = benchmark.concrete_examples(case.case_id)
    count_tokens, tokenizer = _token_counter(qa.model)
    model = qa.llm
    active = active_usage()
    if active is not None and not isinstance(model, MeteredService):
        model = MeteredService(model, active[0], 'llm')
    state = env.reset()
    events, applied, records = [], [], []
    output = ''
    try:
        while not state['done']:
            if not state['can_act']:
                events.append(MemoryEvent('feedback',
                    f"Mind2Web step {state['step_index']}: skipped by candidate rank gate; no model action."))
                state = env.step(None)
                continue
            used = await memory.search(case.query, scope=case.scope,
                extensions=qa.retrieval_extensions(case.metadata))
            applied.append(used)
            history = deepcopy(state['history'])
            query = history + [{'role': 'user', 'content': 'Observation: `' + state['observation'] + '`'}]
            query[0]['content'] = f'Task: {content_text(case.query)}\nTrajectory:\n' + query[0]['content']
            system = [{'role': 'system', 'content': SYSTEM}]
            if count_tokens(system + query) > benchmark.max_context_tokens:
                records.append({'step_index': state['step_index'], 'status': 'context_limit',
                                'query_tokens': count_tokens(system + query)})
                events.append(MemoryEvent('feedback',
                    f"Mind2Web step {state['step_index']}: skipped by context limit; no model action."))
                state = env.step(None, reason='context_limit')
                continue
            workflow = contribution_text(used).strip()
            exemplars = ([[{'role': 'user', 'content': workflow}]] if workflow else []) + concrete
            demos = []
            included = 0
            for example in exemplars:
                if count_tokens(system + demos + example + query) > benchmark.max_context_tokens:
                    break
                demos.extend(example)
                included += 1
            messages = system + demos + query
            raw = str(await model.complete(messages, model=qa.model,
                temperature=qa.temperature, stop=['Task:', 'obs:']))
            action = extract_action(raw)
            events.extend([
                MemoryEvent('observation', query[-1]['content'], role='tool'),
                # Upstream online induction uses the raw actor response. Keep
                # that response even when parsing/scoring rejects the action.
                MemoryEvent('action', raw, role='assistant', extensions={'awm': {'parsed_action': action}}),
            ])
            records.append({'step_index': state['step_index'], 'status': 'predicted',
                            'messages': messages, 'raw_output': raw, 'action': action,
                            'included_exemplars': included, 'available_exemplars': len(exemplars)})
            output = action
            state = env.step(action)
        episode = MemoryEpisode(task=content_text(case.query), scope=case.scope, events=events,
            output=output, success=state['success'], applied_memory=applied,
            feedback='Mind2Web teacher-forced native task success=' + str(state['success']),
            extensions={'awm': {'canonical': False, 'mind2web': {
                'protocol': PROTOCOL, 'step_scores': state['step_scores'],
                'tokenizer': tokenizer, 'max_context_tokens': benchmark.max_context_tokens,
                'token_estimation': 'AWM chat formula; cl100k_base fallback for unrecognized models',
            }}})
        return qa.prepare_episode(case, episode, last_for_scope=False)
    finally:
        sink = qa.services.object('awm_trace_sink')
        if sink:
            sink({'case_id': case.case_id, 'protocol': PROTOCOL, 'model_calls': records,
                  'events': [event.to_dict() for event in events],
                  'step_scores': deepcopy(env.scores)})
