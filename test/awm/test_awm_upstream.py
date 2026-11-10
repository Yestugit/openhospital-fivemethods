"""AST pure-function comparisons against pinned AWM, without executing benchmark runners."""
import ast
import hashlib
import json
from pathlib import Path
import re
import pytest
from mematlas.methods.awm.core import AWMCore

ROOT=Path(__file__).parent/'reference'


def functions(path,names,extra=None):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'re':re,**(extra or {})};exec(compile(tree,str(path),'exec'),scope);return scope


def test_pinned_inventory_and_original_prompts():
    from mematlas.methods.awm import prompts
    source=json.loads((ROOT/'source.json').read_text(encoding='utf-8'))
    for row in source['files']:assert hashlib.sha256((ROOT/row['path']).read_bytes().replace(b"\r\n", b"\n")).hexdigest()==row['sha256']
    for name,path in [('MIND2WEB_INSTRUCTION','mind2web/prompt/instruction_action.txt'),('MIND2WEB_ONE_SHOT','mind2web/prompt/one_shot_action.txt'),
                      ('WEBARENA_INSTRUCTION','webarena/prompt/instruction.txt'),('WEBARENA_ONE_SHOT','webarena/prompt/one_shot.txt')]:
        assert getattr(prompts,name)==(ROOT/path).read_text(encoding='utf-8')


@pytest.mark.parametrize('mode',['online','offline'])
def test_mind2web_formatting_and_filter_match_original(mode):
    original=functions(ROOT/'mind2web/utils/data.py',{'format_examples','filter_workflows','is_website_header'})
    examples=[{'confirmed_task':'Search for item','action_reprs':['# Real HTML\nCLICK [1]'],
               'domain':'Shopping','subdomain':'General','website':'shop'}]
    from mematlas.methods.awm import prompts
    separator=',' if mode=='offline' else ', '
    body='Website: '+separator.join(['Shopping','General','shop'])+'\n'+original['format_examples'](examples,suffix='# Summary Workflows')
    expected='\n\n'.join([prompts.MIND2WEB_INSTRUCTION,prompts.MIND2WEB_ONE_SHOT,body])
    core=AWMCore('mind2web')
    assert core.induction_prompt(examples,mode)==expected
    response='# shop\n\n## Workflow\nDescription\n1. Read\n2. Click\n\n# delta\n\nold example'
    assert core.filter_response(response,'shop')==original['filter_workflows'](response,'shop')


def test_webarena_formatting_and_safe_action_filter_match_source_literals():
    original=functions(ROOT/'webarena/induce_prompt.py',{'format_examples','format_trajectory','remove_invalid_steps'})
    values=["click('12')","click('-1')","click('word')","fill('2', 'text')","scroll(0, 5)","noop()","select_option('2', 'x')"]
    assert AWMCore.valid_actions(values)==original['remove_invalid_steps'](values)
    examples=[{'query':'Find item','think_list':['Use search.'],'action_list':[["fill('1', 'item')"]]}]
    assert AWMCore._webarena_examples(examples)==original['format_examples'](examples)


def test_parser_matches_original_blocks(tmp_path):
    original=functions(ROOT/'mind2web/workflow/retrieve.py',{'clean_workflow_name','load_workflows'})
    text='## Workflow: `search`\nFind things.\n1. Read\n2. Click\n\nshort fragment'
    path=tmp_path/'shop.txt';path.write_text(text,encoding='utf-8')
    expected=original['load_workflows'](str(path).replace('\\','/'))
    actual=AWMCore.parse_workflows(text,'shop')
    for row in actual:row.pop('id')
    assert actual==expected
