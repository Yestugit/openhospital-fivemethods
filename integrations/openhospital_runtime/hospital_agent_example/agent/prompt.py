"""Prompt templates for the baseline doctor agent."""

from __future__ import annotations

import json
from string import Template
from typing import Any, Dict


DOCTOR_SYSTEM_PROMPT = """你是一名医生，正在根据患者信息完成一次诊疗。

基本要求：
- 根据已有对话、已开检查和检查结果做判断。
- 不要编造病史或检查结果；检查结果以系统返回为准。
- 面向患者的问题使用中文。
- 检查名称、检查类别、科室名称和诊断名称要优先使用给定的标准名称。
- 按用户消息要求输出 JSON，不要输出 Markdown 或额外说明。
"""


JSON_REPAIR_SYSTEM_PROMPT = """请把输入改写为合法 JSON 对象。
只输出 JSON，不要输出解释、Markdown、代码围栏或额外文本。"""


NEXT_ACTION_PROMPT = """根据当前病例状态，选择下一步操作。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

可选动作：
- ask_patient：问诊。选择该动作时，同时输出这一步要问患者的问题。
- order_examination：开检查。后续会先选检查类别，再从该类别中选择具体检查。
- final_diagnosis：进入诊断和治疗。后续会先选科室，再从该科室中选择疾病并给出治疗方案。

判断要点：
- 在诊断开始阶段，或者缺少诊断疾病所必需的关键信息时，优先 ask_patient。
- 如果某项检查会影响诊断或治疗，选择 order_examination。
- 如果信息已经足够，选择 final_diagnosis。

请只返回 JSON：
{
  "action": "ask_patient | order_examination | final_diagnosis",
  "question": "action=ask_patient 时填写一个中文问题，否则为空字符串",
  "reason": "简要说明选择这个动作的原因"
}
"""


EXAM_CATEGORY_PROMPT = """请选择下一步最合适的检查类别。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

可选检查类别：
$exam_categories

要求：
- 只能从“可选检查类别”中选择一个标准类别名称。
- 选择最能帮助确认诊断或指导治疗的检查类别。
- 避免选择已经无法提供增量信息的类别。

请只返回 JSON：
{
  "category": "一个标准检查类别名称",
  "reason": "选择该类别的原因"
}
"""


EXAM_ITEM_PROMPT = """请从指定检查类别中选择这次要开的具体检查。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

检查类别：
$category

该类别下的标准检查名称：
$exam_items

要求：
- 只能从“该类别下的标准检查名称”中选择。
- 选择当前确实需要的检查，不要为了凑数量而选择无关检查。
- 不要选择已经做过的检查。

请只返回 JSON：
{
  "examinations": ["标准检查名称1", "标准检查名称2"],
  "reason": "选择这些检查的原因"
}
"""


DEPARTMENT_PROMPT = """请选择最可能负责当前诊断的科室。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

可选科室：
$departments

要求：
- 只能从“可选科室”中选择一个标准科室名称。
- 根据患者表现、检查结果和主要鉴别诊断选择。

请只返回 JSON：
{
  "department": "一个标准科室名称",
  "reason": "选择该科室的原因"
}
"""


DISEASE_AND_TREATMENT_PROMPT = """请在指定科室中选择最可能的疾病，并给出治疗方案。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

科室：
$department

该科室下的标准疾病名称：
$diseases

要求：
- diagnosis 只能填写一个“该科室下的标准疾病名称”。
- treatment_plan 用中文给出治疗方案，重点说明有效性、个性化和安全性：
  1. 有效性：针对诊断给出主要治疗、必要药物或操作。
  2. 个性化：结合患者病史、症状、检查结果、年龄/妊娠/基础病/过敏史等因素调整。
  3. 安全性：说明禁忌、监测、复诊和需要立即就医的危险信号。
- reasoning 简要说明诊断依据和治疗考虑。

请只返回 JSON：
{
  "diagnosis": "一个标准疾病名称",
  "treatment_plan": "具体治疗方案",
  "reasoning": "诊断和治疗推理"
}
"""


EVALUATION_REFLECTION_PROMPT = """训练病例已经完成，并收到了评估结果。请根据患者对话记录和评估明细写一段可复用的简短反思。

患者对话记录：
$chat_history

评估明细：
$evaluation_details

要求：
- reflection 是唯一会写入 memory 的字段。
- reflection.profile 用 1-2 句话概括患者简介，重点保留从对话记录中可复用的症状线索、关键背景或特殊风险。
- 评估明细只包含 diagnosisDetail、examinationDetail、treatmentDetail；反思时分别对照其中的 submitted/ordered、expected/reference、matched、reasoning 和分数信息。
- 诊断、检查、治疗反思要写清楚本次遗漏或做对的要点。
- 内容要短，适合后续病例作为参考摘要，不要复制长篇参考治疗原文。

请只返回 JSON：
{
  "reflection": {
    "profile": "患者简介",
    "diagnosis_reflection": "诊断方面的经验",
    "examination_reflection": "检查选择方面的经验",
    "treatment_reflection": "治疗方案方面的经验",
    "future_strategy": "以后遇到类似病例的简短策略"
  }
}
"""


def format_prompt(template: str, variables: Dict[str, Any]) -> str:
    """Format a prompt with JSON-safe values."""
    prepared = {}
    for key, value in variables.items():
        if isinstance(value, str):
            prepared[key] = value
        else:
            prepared[key] = json.dumps(value, ensure_ascii=False, indent=2)
    return Template(template).safe_substitute(prepared)


def llm_messages(system_prompt: str, prompt: str, *, cache_prefix: bool = False) -> list[dict]:
    """Cache stable short-term history without rewriting or moving any text.

    MaaS caches at message boundaries. Consecutive user messages keep the
    original text in order, with the changing case state after the breakpoint.
    """
    messages = [{"role": "system", "content": system_prompt}]
    marker = "\n\n记忆摘要：\n"
    memory_start = prompt.find(marker)
    boundary = -1
    if cache_prefix and memory_start >= 0:
        position = memory_start + len(marker)
        try:
            notes, end = json.JSONDecoder().raw_decode(prompt, position)
            if isinstance(notes, list):
                boundary = end
                if notes and isinstance(notes[0], str) and notes[0].startswith("[Short-term conversation]\n"):
                    position += 1
                    while prompt[position].isspace():
                        position += 1
                    _, boundary = json.JSONDecoder().raw_decode(prompt, position)
        except ValueError:
            pass
    # Skip small prefixes that may not meet the API's token minimum.
    if cache_prefix and 0 <= memory_start < boundary and boundary >= 4096:
        messages.extend([
            {"role": "user", "content": [{"type": "text", "text": prompt[:boundary],
                                          "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": prompt[boundary:]}]},
        ])
    else:
        messages.append({"role": "user", "content": prompt})
    return messages
