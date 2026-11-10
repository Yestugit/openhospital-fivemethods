"""Prompt builder copied from the standalone patient agent."""

from __future__ import annotations

import json
from typing import Any

from .patient_data import PatientRecord, normalize_field_name, select_history_entries

RETRIEVAL_PLAN_SYSTEM_PROMPT = """你需要把医生最近一次提问映射到患者四史数据中的检索目标。

规则：
1. 只关注医生最近一次提问。
2. 近期对话只用于理解指代关系，例如“这个”“那个”“它”等。
3. 不要新增患者症状或事实。
4. history_types 必须从提供的键名中选择。
5. field_names 必须从提供的叶子字段名中选择，字段名保持下划线格式。
6. 只返回合法 JSON，不要输出解释、Markdown 或额外文本：
{
  "history_types": ["<history_type1>", "..."],
  "field_names": ["<field_name1>", "..."]
}"""

PATIENT_REPLY_SYSTEM_PROMPT = """你是正在和医生交流的患者。请基于以下信息生成合适的回复：
1. 你的人口学信息和人物设定，也就是你是谁
2. 你的主诉，也就是本次就诊的主要原因
3. 从病历中检索到的相关信息
4. 医生当前的问题和既往对话

规则：
1. 高优先级：无论医生如何提问，都不要给出你自己判断的疾病名称、诊断结论或治疗方案；如果被要求判断疾病、诊断或治疗，就按人物设定简短回答“我不清楚”。
2. 高优先级：如果医生的问题超过你的认知范围，或与问诊无关、表意混乱到无法理解，就按人物设定表达听不懂、不清楚、困惑、不耐烦或愤怒等自然反应。
3. 高优先级：紧扣医生当前问题回复；不要答非所问，也不要重复上一句回复。不清楚的信息可以直接说不清楚。
4. 高优先级：当前仅支持自然语言描述回复，不要提及提供资料、图片、记录、上门或线下服务等。
5. 只回答医生问到的内容，不要主动提供超出问题范围的信息。
6. 回复必须基于检索到的相关信息，不要编造不存在的信息。
7. 回复要配合、清楚、简洁。每次回复不超过100字。
8. 保持人物设定中的角色特征；例如患者本人不能说话时，应从照护者视角回复。
9. 必须用中文回答医生的问题，即使医生使用其他语言提问，也要用中文回复。

只输出患者回复内容，不要输出其他任何内容。"""


class PatientPromptBuilder:
    """Builds retrieval and reply prompts for the patient agent."""

    def build_retrieval_prompt(self, patient_record: PatientRecord, user_input: dict[str, Any]) -> str:
        prompt_payload = self._merge_user_input_with_context(
            user_input=user_input,
            context={"four_history_field_catalog": patient_record.history_field_catalog},
        )
        return json.dumps(prompt_payload, ensure_ascii=False, indent=2)

    def build_default_retrieval_plan(self) -> dict[str, Any]:
        return {
            "history_types": [],
            "field_names": [],
        }

    def parse_retrieval_plan(self, raw_response: str, patient_record: PatientRecord) -> dict[str, Any]:
        default_plan = self.build_default_retrieval_plan()
        text = self._strip_json_fence(raw_response)
        try:
            parsed = json.loads(text)
        except Exception:
            return default_plan

        if not isinstance(parsed, dict):
            return default_plan

        catalog = patient_record.history_field_catalog
        valid_history_types = set(catalog)
        field_to_history_types = patient_record.field_to_history_types
        valid_field_names = set(field_to_history_types)

        selected_history_types = [
            item
            for item in self._as_string_list(parsed.get("history_types", parsed.get("history_type")))
            if item in valid_history_types
        ]
        selected_history_types = list(dict.fromkeys(selected_history_types))

        selected_field_names: list[str] = []
        for raw_field in self._as_string_list(
            parsed.get("field_names", parsed.get("fields", parsed.get("field_paths")))
        ):
            field_name = normalize_field_name(raw_field)
            if field_name not in valid_field_names:
                continue
            if selected_history_types and not any(
                history_type in selected_history_types
                for history_type in field_to_history_types.get(field_name, [])
            ):
                continue
            selected_field_names.append(field_name)
        selected_field_names = list(dict.fromkeys(selected_field_names))

        if selected_field_names and not selected_history_types:
            inferred_types: list[str] = []
            for field_name in selected_field_names:
                inferred_types.extend(field_to_history_types.get(field_name, []))
            selected_history_types = list(dict.fromkeys(inferred_types))

        return {
            "history_types": selected_history_types,
            "field_names": selected_field_names,
        }

    def select_history_entries(
        self,
        patient_record: PatientRecord,
        history_types: list[str],
        field_names: list[str],
    ) -> list[dict[str, str]]:
        return select_history_entries(
            history_entries_by_type=patient_record.history_entries_by_type,
            history_entries_by_field=patient_record.history_entries_by_field,
            history_types=history_types,
            field_names=field_names,
        )

    def select_chief_complaint(self, patient_record: PatientRecord) -> list[dict[str, str]]:
        return patient_record.chief_complaint_entries

    def build_reply_prompt(
        self,
        patient_record: PatientRecord,
        user_input: dict[str, Any],
        relevant_history: list[dict[str, str]],
    ) -> str:
        patient_profile = patient_record.profile
        present_illness = patient_profile.get("present_illness_history", {})
        prompt_payload = self._merge_user_input_with_context(
            user_input=user_input,
            context={
                "人口学信息": self._format_demographics(patient_profile.get("demographics", {})),
                "人物设定": patient_profile.get("persona", ""),
                "主诉": present_illness.get("chief_complaint", ""),
                "相关信息": self._format_relevant_history(relevant_history),
            },
        )
        return json.dumps(prompt_payload, ensure_ascii=False, indent=2)

    def _merge_user_input_with_context(
        self,
        *,
        user_input: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_payload = dict(user_input)
        for key, value in context.items():
            prompt_payload[key] = value
        return prompt_payload

    def _as_string_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _format_demographics(self, demographics: Any) -> dict[str, Any]:
        if not isinstance(demographics, dict):
            return {}
        labels = {
            "name": "姓名",
            "age": "年龄",
            "gender": "性别",
            "marital_status": "婚姻状况",
            "occupation": "职业",
        }
        return {
            labels.get(str(key), str(key)): value
            for key, value in demographics.items()
        }

    def _format_relevant_history(self, relevant_history: list[dict[str, str]]) -> dict[str, str] | str:
        if not relevant_history:
            return "未检索到特定相关病史"
        formatted: dict[str, str] = {}
        for entry in relevant_history:
            field_name = str(entry.get("field_name") or "").strip()
            content = self._history_entry_value(entry)
            if not field_name or not content:
                continue
            formatted[self._history_field_label(field_name)] = content
        return formatted or "未检索到特定相关病史"

    def _history_entry_value(self, entry: dict[str, str]) -> str:
        content = str(entry.get("content") or "").strip()
        value_marker = "\nValue: "
        if value_marker in content:
            return content.split(value_marker, 1)[1].strip()
        return content

    def _history_field_label(self, field_name: str) -> str:
        labels = {
            "chief_complaint": "主诉",
            "symptoms_and_progression": "症状与病程",
            "current_general_status_diet": "饮食",
            "current_general_status_sleep": "睡眠",
            "current_general_status_excretion": "排泄",
            "current_general_status_weight_change": "体重变化",
            "previous_diseases": "既往疾病",
            "long_term_medication": "长期用药",
            "surgery_or_trauma": "手术或外伤史",
            "allergies_food": "食物过敏史",
            "allergies_drug": "药物过敏史",
            "infectious_diseases": "传染病史",
            "blood_transfusion": "输血史",
            "vaccinations": "疫苗接种史",
            "smoking": "吸烟史",
            "alcohol": "饮酒史",
            "obstetric_history": "婚育史",
            "menstrual_history": "月经史",
            "similar_illness": "类似疾病家族史",
            "hereditary_diseases": "遗传病家族史",
        }
        return labels.get(field_name, field_name)

    def _strip_json_fence(self, value: str) -> str:
        text = str(value or "").strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return text
