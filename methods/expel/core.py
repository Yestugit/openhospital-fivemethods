from __future__ import annotations

import re
from typing import Any, Mapping


def parse_rules(llm_text: str) -> list[tuple[str, str]]:
    pattern = r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)"
    matches = re.findall(pattern, llm_text)
    result = []
    for operation, text in matches:
        text = text.strip()
        if text and not any(word in text for word in ("ADD", "AGREE", "EDIT")) and text.endswith("."):
            result.append(("ADD" if "ADD" in operation else operation.strip(), text))
    return result


def _rule_index(rules: list[tuple[str, int]], text: str) -> int | None:
    return next((index for index, (rule, _) in enumerate(rules) if rule in text), None)


def update_rules(
    rules: list[tuple[str, int]],
    operations: list[tuple[str, str]],
    list_full: bool = False,
) -> list[tuple[str, int]]:
    result = list(rules)
    normalized = []
    for operation, text in operations:
        operation_type = operation.split()[0]
        index = _rule_index(result, text)
        if operation_type == "ADD":
            if index is None:
                normalized.append((operation, text))
        elif operation_type == "EDIT":
            if index is not None:
                normalized.append((f"AGREE {index + 1}", result[index][0]))
            elif (len(operation.split()) == 2 and operation.split()[1].isdigit()
                  and 1 <= int(operation.split()[1]) <= len(result)):
                normalized.append((operation, text))
        elif index is not None:
            normalized.append((operation, text))
    for operation_type in ("REMOVE", "AGREE", "EDIT", "ADD"):
        for operation, text in normalized:
            if operation.split()[0] != operation_type:
                continue
            if operation_type == "ADD":
                result.append((text, 2))
                continue
            index = _rule_index(result, text)
            if operation_type == "EDIT":
                index = int(operation.split()[1]) - 1
                result[index] = (text, result[index][1] + 1)
            elif index is not None and operation_type == "AGREE":
                result[index] = (result[index][0], result[index][1] + 1)
            elif index is not None:
                result[index] = (result[index][0], result[index][1] - (3 if list_full else 1))
    return sorted((item for item in result if item[1] > 0), key=lambda item: item[1], reverse=True)


class ExpeLCore:
    """ExpeL 内部规则处理：组装提炼提示、解析规则操作并更新规则计数。

    成功经历的检索索引由 retrieval.py 提供，公共操作及学习时机由 method.py 组织。
    """

    DEFAULT_FEWSHOTS = {"hotpotqa": 6, "alfworld": 2, "webshop": 2, "medicine": 2}

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    def critique_messages(
        self,
        *,
        benchmark: str,
        success_history: str,
        fail_history: str | None,
        task: str | None,
        existing_rules: list[str],
        max_num_rules: int,
    ) -> list[dict[str, str]]:
        from .prompts import (INSTRUCTIONS, CRITIQUE_SUMMARY_SUFFIX, human_critique_existing_rules_template,
                              human_critique_existing_rules_all_success_template)
        template = human_critique_existing_rules_template if fail_history is not None else human_critique_existing_rules_all_success_template
        instruction = INSTRUCTIONS[benchmark]["compare" if fail_history is not None else "success"]
        prompt = template.format(instruction="",
            task=task, success_history=success_history, fail_history=fail_history,
            existing_rules="\n".join(f"{index}. {rule}" for index, rule in enumerate(existing_rules, 1)))
        prompt += CRITIQUE_SUMMARY_SUFFIX["full" if len(existing_rules) >= max_num_rules else "not_full"]
        # Upstream uses HumanMessagePromptTemplate even for its 'system' instruction.
        return [{"role": "user", "content": "You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. " + instruction},
                {"role": "user", "content": prompt}]

    @staticmethod
    def parse_rules(output: str):
        return parse_rules(output)

    @staticmethod
    def update_rules(rules, operations, list_full=False):
        return update_rules(rules, operations, list_full)

    @staticmethod
    def format_rules(benchmark: str, rules: list[str]) -> str:
        from .prompts import RULE_TEMPLATES
        return RULE_TEMPLATES[benchmark].format(rules="\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, 1)))

    @staticmethod
    def get_fewshot_max_tokens(benchmark: str) -> int:
        from .prompts import FEWSHOT_TOKEN_LIMITS
        if benchmark == "medicine":
            return 2048
        return FEWSHOT_TOKEN_LIMITS[benchmark]
