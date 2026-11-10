"""Native trajectory splitting from pinned ExpeL; see UPSTREAM_LICENSE.

Only regex flag placement is repaired for modern Python. Public events carry
the actual native transcript; no synthetic thoughts or reflections are created.
"""

from __future__ import annotations

import re

from typing import List



def hotpotqa_step_identifier(line: str) -> str:
    line = line.strip()
    pattern = re.compile('(?i)^action(?:\\s+(\\d+))?:')
    match = pattern.match(line)
    if match:
        return 'action'
    pattern = re.compile('(?i)^observation(?:\\s+(\\d+))?:')
    match = pattern.match(line)
    if match:
        return 'observation'
    return 'thought'

def hotpotqa_cycler(lines: str) -> List[str]:
    new_lines = []
    scratch_pad = ''
    for line in lines.split('\n'):
        pattern = re.compile('(?i)^action(?:\\s+(\\d+))?:')
        match = pattern.match(line)
        if match:
            if scratch_pad != '':
                new_lines.append(scratch_pad.strip())
                scratch_pad = ''
            new_lines.append(line)
            continue
        pattern = re.compile('(?i)^thought(?:\\s+(\\d+))?:')
        match = pattern.match(line)
        if match:
            if scratch_pad != '':
                new_lines.append(scratch_pad.strip())
                scratch_pad = ''
            new_lines.append(line)
            continue
        scratch_pad += line + '\n'
    if scratch_pad != '':
        new_lines.append(scratch_pad.strip())
    return new_lines

def hotpotqa_step_stripper(step: str, step_type: str):
    if step_type == 'observation':
        return re.sub('(?i)^observation(?:\\s+(\\d+))?:', 'Observation:', step)
    if step_type == 'action':
        return re.sub('(?i)^action(?:\\s+(\\d+))?:', 'Action:', step)
    if step_type == 'thought':
        return re.sub('(?i)^thought(?:\\s+(\\d+))?:', 'Thought:', step)
    return step

def alfworld_step_identifier(line: str) -> str:
    line = line.strip()
    if line.lower().startswith('> think'):
        return 'thought'
    if line.lower().startswith('>'):
        return 'action'
    return 'observation'

def alfworld_cycler(lines: str) -> List[str]:
    return lines.split('\n')

def alfworld_step_stripper(step: str, step_type: str):
    return step

def webshop_step_identifier(line: str) -> str:
    line = line.strip()
    if line.lower().startswith('action: think['):
        return 'thought'
    if line.lower().startswith('action'):
        return 'action'
    return 'observation'

def webshop_cycler(lines: str) -> List[str]:
    listed_lines = []
    listed_steps = lines.split('\n\n')
    for step in listed_steps:
        listed_lines.extend(step.split('\n', 1))
    return listed_lines

def webshop_step_stripper(step: str, step_type: str):
    if step_type != 'task':
        return step
    return step.strip('Instruction:\n').strip()

def environment_name(task: str, benchmark: str) -> str | None:
    if benchmark != 'alfworld':
        return benchmark
    for keyword, name in [('clean','pick_clean_then_place'),('hot','pick_heat_then_place'),
        ('heat','pick_heat_then_place'),('cool','pick_cool_then_place'),('look','look_at_obj'),
        ('examine','look_at_obj'),('two','pick_two_obj'),('put','pick_and_place')]:
        if keyword in task:
            return name
    return None


def strip_task(task: str, benchmark: str) -> str:
    return task.split('___')[0] if benchmark == 'alfworld' else task


def strip_component(text: str, kind: str, benchmark: str) -> str:
    """Upstream normalizes native query step numbers, preserving stored raw components."""
    stripper = (hotpotqa_step_stripper if benchmark in {'hotpotqa','medicine'} else
                alfworld_step_stripper if benchmark == 'alfworld' else webshop_step_stripper)
    return stripper(text, kind)


def trajectory_fields(text: str, benchmark: str, reflections: list[str]) -> dict[str, list[str]]:
    if benchmark in {'hotpotqa', 'medicine'}:
        cycler, identify, strip = hotpotqa_cycler, hotpotqa_step_identifier, hotpotqa_step_stripper
    elif benchmark == 'alfworld':
        cycler, identify, strip = alfworld_cycler, alfworld_step_identifier, alfworld_step_stripper
    else:
        cycler, identify, strip = webshop_cycler, webshop_step_identifier, webshop_step_stripper
    fields = {'thought': [], 'action': [], 'step': [], 'reflection': list(reflections)}
    current = []
    for line in cycler(text):
        kind = identify(line)
        if kind in {'thought', 'action'}:
            fields[kind].append(line)
        current.append(strip(line, kind))
        if kind == 'observation':
            fields['step'].append('\n'.join(current).strip())
            current = []
    if current:
        fields['step'].append('\n'.join(current).strip())
    return {key: [value for value in values if value.strip()] for key, values in fields.items()}

