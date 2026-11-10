from __future__ import annotations

import re
import ast
from typing import Any
from uuid import uuid4


class AWMCore:
    """Workflow induction prompt, parsing, filtering, and prompt utilization."""

    def __init__(self, profile: str):
        self.profile = profile

    @staticmethod
    def _mind2web_examples(examples: list[dict[str, Any]]) -> str:
        lines = []
        for index, example in enumerate(examples):
            lines.append(f"Query #{index + 1}: {example['confirmed_task']}")
            lines.append("Actions and Environments:")
            lines.extend(example["action_reprs"])
            lines.append("")
        return "\n".join(lines) + "\n\n# Summary Workflows"

    @staticmethod
    def valid_actions(actions: list[str]) -> list[str]:
        output = []
        for action in actions:
            try:
                expression = ast.parse(action.strip(), mode="eval").body
            except SyntaxError:
                output.append(action)
                continue
            if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
                name = expression.func.id
                if name in {"scroll", "noop"}:
                    continue
                if name in {"click", "fill"}:
                    if not expression.args or not isinstance(expression.args[0], ast.Constant):
                        continue
                    value = expression.args[0].value
                    if not isinstance(value, str):
                        continue
                    if name == 'click':
                        try:
                            if type(ast.literal_eval(value)) is not int:
                                continue
                        except (ValueError,SyntaxError):
                            continue
            output.append(action)
        return output

    @staticmethod
    def _webarena_examples(examples: list[dict[str, Any]]) -> str:
        formatted = []
        for example in examples:
            trajectory = []
            for thought, actions in zip(example["think_list"], example["action_list"]):
                actions = AWMCore.valid_actions(actions)
                if not actions:
                    continue
                action_text = "\n".join(actions)
                trajectory.append(
                    f"<think>\n{thought}\n</think>\n<action>\n{action_text}\n</action>"
                )
            formatted.append(f"Query: {example['query']}\nActions:\n" + "\n\n".join(trajectory))
        return "\n\n".join(["## Concrete Examples", *formatted, "## Summary Workflows"])

    def induction_prompt(self, examples: list[dict[str, Any]], induction_mode: str = 'online') -> str:
        if self.profile == "medicine":
            trajectories = "\n\n".join(f"Case: {example['query']}\n{example['trajectory']}" for example in examples)
            return (
                "Induce reusable clinical workflows from successful simulated encounters. "
                "Preserve the ordered questions, evidence, examinations and decisions. "
                "Generalize patient-specific facts; avoid copying a patient's diagnosis as a universal rule. "
                "Use only observed evidence. For each workflow output one paragraph of at least four lines: "
                "## workflow name, applicability, then numbered decision steps. Separate workflows with a blank line. "
                "Keep the clinical decision steps conditional on evidence.\n\n" + trajectories
            )
        from . import prompts
        instruction = getattr(prompts, self.profile.upper() + "_INSTRUCTION")
        one_shot = getattr(prompts, self.profile.upper() + "_ONE_SHOT")
        if self.profile == "mind2web":
            tags = examples[0]["domain"], examples[0]["subdomain"], examples[0]["website"]
            separator = ',' if induction_mode == 'offline' else ', '
            body = f"Website: {separator.join(tags)}\n" + self._mind2web_examples(examples)
        else:
            body = self._webarena_examples(examples)
        return f"{instruction}\n\n{one_shot}\n\n{body}"

    @staticmethod
    def _is_website_header(block: str, website: str) -> bool:
        lines = block.strip().split("\n")
        return (
            len(lines) == 1
            and lines[0].strip().startswith("#")
            and lines[0].strip().lower().endswith(website.lower())
        )

    def filter_response(self, text: str, website: str) -> str:
        if self.profile == "medicine":
            return text.strip()
        if self.profile == "webarena":
            return (
                text
                + "\n\nclick('id') # input string id value for all actions"
                + "\n\nselect_option('id', 'value') # for dropdown menu"
            )
        blocks = text.split("\n\n")
        for index, block in enumerate(blocks):
            if self._is_website_header(block, website):
                blocks = blocks[index + 1 :]
                break
        for index, block in enumerate(blocks):
            if self._is_website_header(block, "delta"):
                blocks = blocks[:index]
                break
        return "\n\n".join(block for block in blocks if "delta" not in block.lower())

    @staticmethod
    def parse_clinical_workflows(text: str) -> list[dict[str, str]]:
        workflows = []
        for match in re.finditer(r"(?ms)^##\s+([^\n]+)\n(.*?)(?=^##\s+|\Z)", text):
            name, body = match.group(1).strip(), match.group(2).strip()
            body = re.sub(r"\s+(?=\d+[.)]\s)", "\n", body)
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            if len(lines) < 3 or sum(bool(re.match(r"\d+[.)]\s", line)) for line in lines) < 2:
                continue
            workflows.append({"id":uuid4().hex, "website":"medicine", "name":"## " + name,
                "docstring":lines[0], "content":"## " + name + "\n" + body})
        return workflows

    @staticmethod
    def parse_workflows(text: str, website: str) -> list[dict[str, str]]:
        workflows = []
        for block in text.split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) < 4:
                continue
            name = lines[0].lstrip("#").strip()
            if ":" in name:
                name = name.split(":", 1)[1].strip()
            match = re.search(r"`([^`]+)`", name)
            if match:
                name = match.group(1)
            heading = f"## {name}"
            workflows.append({
                "id": uuid4().hex,
                "website": website,
                "name": heading,
                "docstring": lines[1].strip(),
                "content": "\n".join([heading, *lines[1:]]),
            })
        return workflows
