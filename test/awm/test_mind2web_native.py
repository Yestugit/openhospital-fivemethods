"""Mind2Web protocol regressions; synthetic HTML and actors, no quality claims."""
from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, Memory
from mematlas.benchmarks.factory import load_benchmark
from mematlas.core.plugin import MethodServices
from awm_fixtures import config, services


def ranked_task(identity, ranks=(0, 8, 0)):
    actions = []
    for index, rank in enumerate(ranks):
        node = str(10 + index)
        html = (f'<html backend_node_id="1"><body backend_node_id="2">'
                f'<a backend_node_id="{node}"><text>Page {index}</text></a>'
                '<a backend_node_id="99"><text>Other page</text></a></body></html>')
        actions.append({'action_uid': f'{identity}-{index}', 'cleaned_html': html,
                        'raw_html': html, 'operation': {'op': 'CLICK', 'value': ''},
                        'pos_candidates': [{'backend_node_id': node, 'rank': rank}],
                        'neg_candidates': [{'backend_node_id': '99', 'rank': 1}]})
    return {'annotation_id': identity, 'confirmed_task': f'Navigate {identity}',
            'website': 'shop', 'domain': 'Shopping', 'subdomain': 'General',
            'actions': actions, 'action_reprs': [f'Open page {i}' for i in range(len(actions))]}


def write_dataset(tmp_path, **changes):
    data = {'train': [ranked_task('train', (0,))], 'test': [ranked_task('test')],
            'exemplars': [[{'role': 'user', 'content': 'TRAIN ONLY DEMO',
                           'specifier': 'Website: shop\nDomain: Shopping\nSubdomain: General'},
                          {'role': 'assistant', 'content': 'Action: `CLICK [99]`'}]],
            'top_k_elements': 5, 'previous_top_k_elements': 3, 'seed': 7}
    data.update(changes)
    path = tmp_path / 'mind2web.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


class Actor:
    def __init__(self, outputs):
        self.config = {'responses': list(outputs)}
        self.outputs = iter(outputs)
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        return next(self.outputs)


def test_rank_gate_and_teacher_forcing_continue_after_wrong_action():
    from mematlas.benchmarks.mind2web_env import Mind2WebEnvironment
    env = Mind2WebEnvironment(ranked_task('test'), top_k=5, previous_top_k=3)
    first = env.reset()
    assert first['can_act'] is True
    second = env.step('CLICK [99]')
    assert second['done'] is False and second['can_act'] is False
    third = env.step(None)
    assert third['can_act'] is True
    # Previous messages are canonical teacher-forced actions, never predictions.
    assert [m['content'] for m in third['history'] if m['role'] == 'assistant'] == [
        'Action: `CLICK [10]` (Open page 0)', 'Action: `CLICK [11]` (Open page 1)']
    final = env.step('CLICK [12]')
    assert final['done'] is True and final['success'] is False
    assert [s['step_success'] for s in final['step_scores']] == [0.0, 0.0, 1.0]
    assert final['step_scores'][1]['reason'] == 'candidate_rank'
    assert final['step_scores'][1]['prediction'] is None


@pytest.mark.asyncio
async def test_registered_benchmark_qa_and_scoring_keep_full_episode(tmp_path):
    from mematlas.methods.awm.qa import AWMQA
    path = write_dataset(tmp_path)
    benchmark = load_benchmark('mind2web', path)
    supplied = services()
    actor = Actor(['Action: `CLICK [99]`', 'Action: `CLICK [12]`'])
    supplied.objects['answer_llm'] = actor
    cfg = ExperimentConfig(methods=(config(tmp_path / 'memory', induction_mode='offline'),),
                           benchmark={'kind': 'mind2web', 'path': str(path)},
                           protocol='experience_train_test', output_dir=str(tmp_path / 'report'))
    runner = ExperimentRunner()
    assert runner.check(cfg, services=supplied)['ready']
    result = (await runner.evaluate(cfg, services=supplied))[0]
    assert result.failures == ()
    assert len(result.predictions) == 1
    episode = result.predictions[0].metadata['episode']
    assert episode['success'] is False and episode['events']
    scores = episode['extensions']['awm']['mind2web']['step_scores']
    assert [s['step_success'] for s in scores] == [0.0, 0.0, 1.0]
    assert result.metrics['native.step_success_rate'] == pytest.approx(1 / 3)
    assert result.metrics['native.action_f1'] == pytest.approx(2 / 3)
    assert result.metrics['native.success_rate'] == 0.0
    assert result.metrics['native.candidate_misses'] == 1
    assert len(actor.calls) == 2
    assert actor.calls[0][1]['role'] == 'user'  # Workflow in upstream user role.
    assert any('TRAIN ONLY DEMO' in m['content'] for m in actor.calls[0])
    assert 'specifier' not in json.dumps(actor.calls)
    assert 'CLICK [12]' not in json.dumps(actor.calls[0])  # Future gold withheld.
    assert 'Action: `CLICK [10]` (Open page 0)' in json.dumps(actor.calls[1])
    assert 'Action: `CLICK [99]` (Open page 0)' not in json.dumps(actor.calls[1])
    assert (tmp_path / 'report/results.json').exists()


@pytest.mark.asyncio
async def test_online_training_is_collected_by_registered_qa(tmp_path):
    path = write_dataset(tmp_path, training_mode='agent', test=[ranked_task('test', (0,))])
    supplied = services()
    actor = Actor(['Action: `CLICK [10]`', 'Action: `CLICK [10]`'])
    supplied.objects['answer_llm'] = actor
    cfg = ExperimentConfig(methods=(config(tmp_path / 'memory', induction_mode='online'),),
                           benchmark={'kind': 'mind2web', 'path': str(path)},
                           protocol='experience_train_test', output_dir=str(tmp_path / 'report'))
    runner = ExperimentRunner()
    assert runner.check(cfg, services=supplied)['ready']
    result = (await runner.evaluate(cfg, services=supplied))[0]
    assert result.failures == ()
    assert len(actor.calls) == 2
    assert len(supplied.llm.calls) == 1
    # Upstream online induction takes '# ' + the actual last input message,
    # including the Observation wrapper, followed by the raw actor response.
    assert '# Task: Navigate train\nTrajectory:\nObservation: `' in supplied.llm.calls[0]['messages'][0]['content']
    assert result.predictions[0].applied_memory[0].items
    assert result.metrics['native.success_rate'] == 1.0


def test_manifest_rejects_overlapping_tasks_and_missing_candidate_ranks(tmp_path):
    row = ranked_task('same')
    with pytest.raises(ValueError, match='disjoint'):
        load_benchmark('mind2web', write_dataset(tmp_path, train=[row], test=[row]))
    row = ranked_task('test')
    del row['actions'][0]['pos_candidates'][0]['rank']
    with pytest.raises(ValueError, match='rank'):
        load_benchmark('mind2web', write_dataset(tmp_path, test=[row]))


@pytest.mark.parametrize('response, expected', [
    ('Action: `CLICK [10]`', 'CLICK [10]'), ('CLICK [10]', ''),
    ('Action: `CLICK [99]` and `CLICK [10]`', 'CLICK [99]'), ('garbage', '')])
def test_native_parser_matches_upstream_single_backtick_extraction(response, expected):
    from mematlas.benchmarks.mind2web_env import extract_action
    assert extract_action(response) == expected


@pytest.mark.asyncio
async def test_all_candidate_misses_finish_without_model_or_empty_induction(tmp_path):
    path = write_dataset(tmp_path, training_mode='agent',
                         train=[ranked_task('train', (8,))], test=[ranked_task('test', (8, 8))])
    supplied = services()
    actor = Actor([])  # Any unintended call fails immediately.
    supplied.objects['answer_llm'] = actor
    cfg = ExperimentConfig(methods=(config(tmp_path / 'memory'),),
                           benchmark={'kind': 'mind2web', 'path': str(path)},
                           protocol='experience_train_test', output_dir=str(tmp_path / 'report'))
    result = (await ExperimentRunner().evaluate(cfg, services=supplied))[0]
    assert result.failures == ()
    assert actor.calls == [] and supplied.llm.calls == []
    assert result.metrics['native.candidate_misses'] == 2
    assert result.metrics['native.step_success_rate'] == 0.0


def test_dom_and_native_metrics_match_pinned_upstream():
    from mematlas.benchmarks import mind2web_dom as actual
    from mind2web_smoke import _env_helpers
    reference = _env_helpers()
    for step in ranked_task('probe')['actions']:
        for k in (3, 5):
            assert actual.get_top_k_obs(step, k, use_raw=False) == reference.get_top_k_obs(step, k, use_raw=False)
        assert actual.get_target_obs_and_act(step) == reference.get_target_obs_and_act(step)
    for text in ('CLICK [10]', 'TYPE [10] [Hello world]', 'CLICK 10', ''):
        assert actual.parse_act_str(text) == reference.parse_act_str(text)


def test_exemplar_filter_uses_site_then_subdomain_and_never_current_task(tmp_path):
    bank = [[{'role': 'user', 'content': label, 'specifier': tags}]
            for label, tags in [
                ('correct', 'Shopping General shop'),
                ('other-site', 'Shopping General other'),
                ('wrong-domain', 'Travel General shop'),
                ('self', 'Shopping General shop\nTask: Navigate train')]]
    path = write_dataset(tmp_path, exemplars=bank, retrieve_top_k=10)
    benchmark = load_benchmark('mind2web', path)
    assert benchmark.concrete_examples('train') == [[{'role': 'user', 'content': 'correct'}]]
    bank = [bank[1]]
    benchmark = load_benchmark('mind2web', write_dataset(tmp_path, exemplars=bank))
    assert benchmark.concrete_examples('test') == [[{'role': 'user', 'content': 'other-site'}]]


@pytest.mark.asyncio
async def test_context_budget_scores_zero_and_continues_without_model(tmp_path):
    path = write_dataset(tmp_path, max_context_tokens=1, test=[ranked_task('test', (0, 0))])
    supplied = services()
    actor = Actor([])
    supplied.objects['answer_llm'] = actor
    cfg = ExperimentConfig(methods=(config(tmp_path / 'memory', induction_mode='offline'),),
                           benchmark={'kind': 'mind2web', 'path': str(path)},
                           protocol='experience_train_test', output_dir=str(tmp_path / 'report'))
    result = (await ExperimentRunner().evaluate(cfg, services=supplied))[0]
    assert result.failures == () and actor.calls == []
    scores = result.predictions[0].metadata['episode']['extensions']['awm']['mind2web']['step_scores']
    assert [s['reason'] for s in scores] == ['context_limit', 'context_limit']


@pytest.mark.asyncio
async def test_actor_messages_and_step_scores_match_original_eval_sample(tmp_path):
    """Execute upstream eval_sample unchanged, replacing only network/token IO."""
    from awm_fixtures import WORKFLOW
    from mind2web_smoke import _env_helpers
    row = ranked_task('test')
    path = write_dataset(tmp_path)
    responses = ['Action: `CLICK [99]`', 'Action: `CLICK [12]`']
    oracle_calls = []
    answers = iter(responses)
    def generate_response(*, messages, **kwargs):
        oracle_calls.append(deepcopy(messages))
        return next(answers), {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    reference_root = Path(__file__).parent / 'reference/mind2web'
    namespace = {**vars(_env_helpers()), 'np': SimpleNamespace(sum=sum), 'Path': Path, 'json': json,
                 'generate_response': generate_response,
                 'num_tokens_from_messages': lambda *args: 1, 'MAX_TOKENS': {'fixture': 16385}}
    parser_source = ast.parse((reference_root / 'utils/llm.py').read_text(encoding='utf-8-sig'))
    extractor = next(n for n in parser_source.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_from_response')
    exec(compile(ast.Module(body=[extractor], type_ignores=[]), 'upstream-parser', 'exec'), namespace)
    source = ast.parse((reference_root / 'memory.py').read_text(encoding='utf-8-sig'))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_sample')
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'upstream-eval', 'exec'), namespace)
    namespace['get_exemplars'] = lambda args: [
        [{'role': 'user', 'content': WORKFLOW}],
        [{'role': 'user', 'content': 'TRAIN ONLY DEMO'},
         {'role': 'assistant', 'content': 'Action: `CLICK [99]`'}]]
    args = SimpleNamespace(model='fixture', top_k_elements=5, previous_top_k_elements=3,
                           log_dir=str(tmp_path / 'oracle'), benchmark='test_task', website='shop',
                           suffix='workflow', temperature=0.0)
    namespace['eval_sample'](0, args, row)
    oracle = json.loads((tmp_path / 'oracle/fixture/test_task/shop/workflow/0.json').read_text())[-1]
    supplied = services()
    actor = Actor(responses)
    supplied.objects['answer_llm'] = actor
    cfg = ExperimentConfig(methods=(config(tmp_path / 'memory', induction_mode='offline'),),
                           benchmark={'kind': 'mind2web', 'path': str(path)},
                           protocol='experience_train_test', output_dir=str(tmp_path / 'report'))
    result = (await ExperimentRunner().evaluate(cfg, services=supplied))[0]
    assert result.failures == ()
    assert actor.calls == oracle_calls
    steps = result.predictions[0].metadata['episode']['extensions']['awm']['mind2web']['step_scores']
    for key in ('element_acc', 'action_f1', 'step_success'):
        assert [s[key] for s in steps] == oracle[key]


@pytest.mark.asyncio
async def test_native_resume_reuses_completed_training_and_frozen_prediction(tmp_path):
    from dataclasses import replace
    path = write_dataset(tmp_path, training_mode='agent', test=[ranked_task('test', (0,))])
    supplied = services()
    actor = Actor(['Action: `CLICK [10]`', 'Action: `CLICK [10]`'])
    supplied.objects['answer_llm'] = actor
    cfg = ExperimentConfig(methods=(config(tmp_path / 'memory'),),
                           benchmark={'kind': 'mind2web', 'path': str(path)},
                           protocol='experience_train_test', output_dir=str(tmp_path / 'first'))
    first = (await ExperimentRunner().evaluate(cfg, services=supplied))[0]
    assert first.failures == ()
    resumed = (await ExperimentRunner().evaluate(replace(cfg, output_dir=str(tmp_path / 'resumed')),
                    services=supplied, resume_from=tmp_path / 'first'))[0]
    assert resumed.failures == ()
    assert resumed.predictions == first.predictions
    assert len(actor.calls) == 2 and len(supplied.llm.calls) == 1


def test_offline_method_rejects_agent_training_before_allocation(tmp_path):
    path = write_dataset(tmp_path, training_mode='agent')
    supplied = services()
    supplied.objects['answer_llm'] = Actor([])
    cfg = ExperimentConfig(methods=(config(tmp_path / 'memory', induction_mode='offline'),),
                           benchmark={'kind': 'mind2web', 'path': str(path)},
                           protocol='experience_train_test')
    check = ExperimentRunner().check(cfg, services=supplied)
    assert check['ready'] is False
    assert 'canonical' in json.dumps(check)


def test_public_benchmark_cli_inspects_registered_mind2web(tmp_path, capsys):
    from mematlas.cli.main import main
    path = write_dataset(tmp_path)
    assert main(['benchmark', 'mind2web', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['case_count'] == 1
    assert result['description']['protocol'] == 'mind2web_teacher_forcing_v1'
