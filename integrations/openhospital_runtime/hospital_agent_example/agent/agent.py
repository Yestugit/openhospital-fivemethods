"""A readable baseline doctor agent.

The SDK handles train/test orchestration and service calls. This file shows a
clear reference flow: decide action -> ask/order exam/diagnose -> train reflection.
"""

from __future__ import annotations

import json
import inspect
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from hospital_agent_sdk import AgentBuilder, BasicAgent, load_config
from .memory import build_memory
from .prompt import (
    DEPARTMENT_PROMPT,
    DISEASE_AND_TREATMENT_PROMPT,
    DOCTOR_SYSTEM_PROMPT,
    EVALUATION_REFLECTION_PROMPT,
    EXAM_CATEGORY_PROMPT,
    EXAM_ITEM_PROMPT,
    JSON_REPAIR_SYSTEM_PROMPT,
    NEXT_ACTION_PROMPT,
    format_prompt,
    llm_messages,
)


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR.parent / ".env.local", override=False)
REF_DATA_DIR = BASE_DIR / "data" / "ref_data"


class DoctorOutputPaused(RuntimeError):
    """No valid decision after bounded format repair; preserve the case boundary."""


class MyDoctorAgent(BasicAgent):
    """Baseline doctor agent with explicit, easy-to-read steps."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, memory: Any = None):
        super().__init__(config=config, memory=memory)

        # 标准目录来自 data/ref_data。所有类别、科室、检查、疾病都尽量从这些标准名称中选择。
        self.examination_catalog = load_examination_catalog()
        self.disease_catalog = load_disease_catalog()
        self.exam_categories = list(self.examination_catalog.keys())
        self.departments = list(self.disease_catalog.keys())
        self.exam_category_map = build_name_map(self.exam_categories)
        self.department_map = build_name_map(self.departments)

    async def train(self, patient_id: str) -> Dict[str, Any]:
        return await self._run_agent(patient_id=patient_id, mode="train")

    async def test(self, patient_id: str) -> Dict[str, Any]:
        return await self._run_agent(patient_id=patient_id, mode="test")

    async def _run_agent(self, patient_id: str, mode: str) -> Dict[str, Any]:
        self._timeline = []
        case_state: Dict[str, Any] = {
            "timeline": self._timeline,
            "patient_id": patient_id,
            "mode": mode,
            "memory_notes": [],
            "chat_history": [],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
            "decision_trace": [],
            "exam_decision_trace": [],
        }
        memory_notes = await self._memory_notes(case_state, phase="initial")
        case_state["memory_notes"] = memory_notes

        # 步骤 1：循环判断下一步动作。这里只决定“问诊 / 开检查 / 诊断”。
        max_decision_steps = max(1, int(self.config.get("max_decision_steps", 12)))
        decision_steps = 0
        decision_limit_reached = True
        while decision_steps < max_decision_steps:
            decision_steps += 1
            chat_history = case_state.get("chat_history", [])
            examinations = self._examination_context(case_state)
            memory_notes = await self._memory_notes(case_state, phase="next_action")
            case_state["memory_notes"] = memory_notes
            action_prompt = format_prompt(
                NEXT_ACTION_PROMPT,
                {
                    "memory_notes": memory_notes,
                    "chat_history": chat_history,
                    "examinations": examinations,
                },
            )
            if not case_state["chat_history"]:
                fallback_action = {
                    "action": "ask_patient",
                    "question": "请您按时间顺序描述这次最主要的不适、何时开始、如何变化，以及伴随症状。",
                    "reason": "需要先了解主诉和现病史。",
                }
            else:
                fallback_action = {
                    "action": "final_diagnosis",
                    "question": "",
                    "reason": "已有信息足以进入最终诊疗，或继续收集信息的收益有限。",
                }

            decision = await self._call_llm(
                prompt=action_prompt,
                default=fallback_action,
                prompt_name="next_action",
                patient_id=patient_id,
            )
            action = str(decision.get("action", "")).strip().lower()
            if action not in {"ask_patient", "order_examination", "final_diagnosis"}:
                action = fallback_action["action"]

            question = clean_text(decision.get("question"))
            if action == "ask_patient" and not question:
                question = "请您描述这次最主要的不适、开始时间和伴随症状。"

            decision = {
                "action": action,
                "question": question if action == "ask_patient" else "",
                "reason": clean_text(decision.get("reason")),
            }
            case_state["decision_trace"].append(decision)

            # 步骤 2A：如果继续问诊，就把本轮问题发给患者，并保存问答历史。
            if action == "ask_patient":
                answer = await self.actions.ask_patient(
                    patient_id,
                    {
                        "question": decision["question"],
                        "chat_history": case_state["chat_history"],
                    },
                )
                self._timeline.extend([
                    {"kind": "message", "role": "assistant", "text": decision["question"]},
                    {"kind": "message", "role": "user", "text": str(answer)},
                ])
                case_state["chat_history"].extend(
                    [
                        {"from": "doctor", "text": decision["question"]},
                        {"from": "patient", "text": answer},
                    ]
                )
                continue

            # 步骤 2B：如果开检查，先选检查类别，再从该类别里选具体标准检查名称。
            if action == "order_examination":
                chat_history = case_state.get("chat_history", [])
                examinations = self._examination_context(case_state)
                memory_notes = await self._memory_notes(case_state, phase="examination")
                case_state["memory_notes"] = memory_notes

                # 检查步骤 1：先让模型从所有标准检查类别中选一个类别。
                category_prompt = format_prompt(
                    EXAM_CATEGORY_PROMPT,
                    {
                        "memory_notes": memory_notes,
                        "chat_history": chat_history,
                        "examinations": examinations,
                        "exam_categories": self.exam_categories,
                    },
                )
                available_categories = [name for name, items in self.examination_catalog.items()
                    if any(item not in case_state["ordered_examinations"] for item in items)]
                if not available_categories:
                    decision_limit_reached = False
                    break
                category_decision = await self._call_llm(
                    prompt=category_prompt,
                    default={"category": available_categories[0] if available_categories else "", "reason": "JSON recovery exhausted; deterministic catalog fallback."},
                    prompt_name="exam_category",
                    patient_id=patient_id,
                )
                category = match_standard_name(category_decision.get("category"), self.exam_category_map)
                if category not in available_categories:
                    category = available_categories[0]
                    for candidate, names in self.examination_catalog.items():
                        if any(name not in case_state["ordered_examinations"] for name in names):
                            category = candidate
                            break

                # 检查步骤 2：只给该类别下的检查名称，让模型选择具体要开的检查。
                item_prompt = format_prompt(
                    EXAM_ITEM_PROMPT,
                    {
                        "memory_notes": memory_notes,
                        "chat_history": chat_history,
                        "examinations": examinations,
                        "category": category,
                        "exam_items": self.examination_catalog.get(category, []),
                    },
                )
                available_items = [item for item in self.examination_catalog.get(category, [])
                    if item not in case_state["ordered_examinations"]]
                item_decision = await self._call_llm(
                    prompt=item_prompt,
                    default={"examinations": available_items[:1], "reason": "JSON recovery exhausted; deterministic catalog fallback."},
                    prompt_name="exam_item",
                    patient_id=patient_id,
                    allowed_examinations=self.examination_catalog.get(category, []),
                    already_ordered=case_state["ordered_examinations"],
                )
                allowed_exam_map = build_name_map(self.examination_catalog.get(category, []))
                already_ordered = set(as_text_list(case_state["ordered_examinations"]))
                examinations = []
                for item in as_text_list(item_decision.get("examinations")):
                    standard_name = match_standard_name(item, allowed_exam_map)
                    if not standard_name or standard_name in already_ordered or standard_name in examinations:
                        continue
                    examinations.append(standard_name)
                exam_plan = {
                    "category": category,
                    "examinations": examinations,
                    "reason": clean_text(item_decision.get("reason") or category_decision.get("reason")),
                }
                case_state["exam_decision_trace"].append(exam_plan)
                if not examinations:
                    decision_limit_reached = False
                    break

                exam_response = await self.actions.order_examination(
                    patient_id,
                    examinations,
                    reason=exam_plan.get("reason", ""),
                )
                self._timeline.append({"kind": "tool_result", "role": "user",
                                       "text": json.dumps(exam_response, ensure_ascii=False)})
                case_state["ordered_examinations"].extend(
                    as_text_list(exam_response.get("normalized_items"))
                )
                case_state["invalid_examinations"].extend(
                    as_text_list(exam_response.get("invalid_items"))
                )
                results = exam_response.get("results") or {}
                if isinstance(results, dict):
                    case_state["examination_results"].update(results)
                continue

            decision_limit_reached = False
            break

        if decision_limit_reached and self.logger is not None:
            self.logger.write_event(
                event_type="DECISION_LIMIT_REACHED",
                agent_id=getattr(self.runtime_settings, "team_id", "doctor"),
                patient_id=patient_id,
                payload={"max_decision_steps": max_decision_steps},
            )

        # 步骤 3：进入诊断。先选科室，再从该科室疾病列表里选一个疾病并生成治疗方案。
        chat_history = case_state.get("chat_history", [])
        examinations = self._examination_context(case_state)
        memory_notes = await self._memory_notes(case_state, phase="diagnosis_and_treatment")
        case_state["memory_notes"] = memory_notes

        # 诊断步骤 1：先从标准科室中选择一个最相关科室。
        department_prompt = format_prompt(
            DEPARTMENT_PROMPT,
            {
                "memory_notes": memory_notes,
                "chat_history": chat_history,
                "examinations": examinations,
                "departments": self.departments,
            },
        )
        department_decision = await self._call_llm(
            prompt=department_prompt,
            default={"department": self.departments[0] if self.departments else "", "reason": ""},
            prompt_name="department",
            patient_id=patient_id,
        )
        department = match_standard_name(department_decision.get("department"), self.department_map)
        if not department:
            department = self.departments[0] if self.departments else ""

        # 诊断步骤 2：只给该科室下的标准疾病名称，让模型选一个疾病并给出治疗方案。
        disease_prompt = format_prompt(
            DISEASE_AND_TREATMENT_PROMPT,
            {
                "memory_notes": memory_notes,
                "chat_history": chat_history,
                "examinations": examinations,
                "department": department,
                "diseases": self.disease_catalog.get(department, []),
            },
        )
        diseases = self.disease_catalog.get(department, [])
        default_diagnosis = diseases[0] if diseases else "未明确诊断"
        final_plan = await self._call_llm(
            prompt=disease_prompt,
            default={
                "diagnosis": default_diagnosis,
                "treatment_plan": "当前信息不足以制定特异性治疗方案；建议补充病史、体格检查和必要辅助检查后再决策。",
                "reasoning": "LLM 未返回可解析的最终方案，使用保守兜底结果。",
            },
            prompt_name="disease_and_treatment",
            patient_id=patient_id,
        )
        diagnosis = match_standard_name(
            final_plan.get("diagnosis"),
            build_name_map(self.disease_catalog.get(department, [])),
        ) or default_diagnosis
        treatment_plan = clean_text(final_plan.get("treatment_plan"))
        if not treatment_plan:
            treatment_plan = "当前信息不足以制定特异性治疗方案；建议补充关键病史、体格检查和必要辅助检查后再决策。"
        reasoning = clean_text(final_plan.get("reasoning"))
        if not reasoning:
            reasoning = clean_text(department_decision.get("reason")) or "基于问诊和检查结果形成诊疗方案。"

        final_plan = {
            "department": department,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
        }
        case_state["final_plan"] = final_plan

        final_result = await self.actions.prescribe_treatment(
            patient_id=patient_id,
            diagnosis=[diagnosis],
            treatment_plan=treatment_plan,
            reasoning=reasoning,
        )

        # 步骤 4：训练模式下先评估，再基于评估做一次反思，最后把反思写入 memory。
        if mode == "train" and hasattr(self.memory, "record_execution"):
            await self.memory.record_execution(patient_id=patient_id, case_state=case_state, final_result=final_result)
        evaluation_report = None
        evaluation_reflection = None
        if mode == "train":
            evaluation_report = await self.actions.evaluation(
                patient_id=patient_id,
                final_result=final_result,
            )
            case_state["evaluation_report"] = evaluation_report
            if hasattr(self.memory, "record_score"):
                await self.memory.record_score(patient_id=patient_id, case_state=case_state,
                    final_result=final_result, evaluation_report=evaluation_report)

            reflection_prompt = format_prompt(
                EVALUATION_REFLECTION_PROMPT,
                {
                    "chat_history": case_state.get("chat_history", []),
                    "evaluation_details": self._evaluation_details(evaluation_report or {}),
                },
            )
            evaluation_reflection = await self._call_llm(
                prompt=reflection_prompt,
                default={"reflection": {"profile": "", "future_strategy": ""}},
                prompt_name="evaluation_reflection",
                patient_id=patient_id,
            )
            case_state["evaluation_reflection"] = evaluation_reflection

        if mode == "train" and self.memory:
            result = self.memory.append_case_reflection(
                patient_id=patient_id,
                evaluation_reflection=evaluation_reflection,
                case_state=case_state,
                evaluation_report=evaluation_report,
                final_result=final_result,
            )
            if inspect.isawaitable(result):
                await result

        return final_result

    def _decision_error(self, parsed, default, prompt_name, allowed_examinations, already_ordered):
        if not isinstance(parsed, dict):
            return "Expected a JSON object"
        schema = default
        if prompt_name == "evaluation_reflection":
            schema = {"reflection": {name: "" for name in ("profile", "diagnosis_reflection",
                "examination_reflection", "treatment_reflection", "future_strategy")}}
        def validate(value, model, path):
            if type(value) is not type(model):
                return path + ": expected " + type(model).__name__
            if isinstance(model, dict):
                for key, child in model.items():
                    if key not in value:
                        return path + ": missing " + key
                    error = validate(value[key], child, path + "." + key)
                    if error:
                        return error
            if isinstance(model, list) and any(not isinstance(item, str) for item in value):
                return path + ": expected string array"
            return None
        error = validate(parsed, schema, "decision")
        if error:
            return error
        if prompt_name == "next_action" and parsed.get("action") not in {
                "ask_patient", "order_examination", "final_diagnosis"}:
            return "Unknown action"
        if prompt_name == "exam_category" and not match_standard_name(parsed.get("category"), self.exam_category_map):
            return "Category is outside the examination catalog"
        if prompt_name == "exam_item" and allowed_examinations is not None:
            names = build_name_map(allowed_examinations)
            chosen = [match_standard_name(item, names) for item in parsed.get("examinations", [])]
            if any(not item or item in already_ordered for item in chosen):
                return "Examination is outside the selected catalog or already ordered"
            if len(set(chosen)) != len(chosen):
                return "Duplicate examination names"
        return None

    async def _call_llm(self, *, prompt, default, prompt_name="", patient_id="",
                        allowed_examinations=None, already_ordered=()):
        json_options = {"extra_body": {"response_format": {"type": "json_object"}}}
        response = await self._llm_text(prompt, system_prompt=DOCTOR_SYSTEM_PROMPT,
            temperature=0.2, prompt_name=prompt_name, patient_id=patient_id,
            **(json_options if self.config.get("json_exam_fallback") else {}))
        original = response
        errors = []
        fallback = False
        for attempt in range(4):
            parsed = parse_json_object(response)
            error = self._decision_error(parsed, default, prompt_name, allowed_examinations, already_ordered)
            if error is None:
                break
            errors.append({"attempt": attempt, "error": error, "response": response})
            if self.logger is not None:
                self.logger.write_event(event_type="DOCTOR_JSON_INVALID", patient_id=patient_id,
                    agent_id=getattr(self.runtime_settings, "team_id", "doctor"),
                    payload={"prompt_name": prompt_name, **errors[-1]})
            if attempt == 3:
                # Missing explanatory text is absence, not a missing clinical
                # decision. Preserve the doctor's diagnosis and treatment byte
                # for byte; never substitute the default catalog diagnosis or
                # invent reasoning/therapy to make a failed response pass.
                if prompt_name == "disease_and_treatment":
                    candidate = parse_json_object(original)
                    if (isinstance(candidate, dict) and "reasoning" not in candidate
                            and all(isinstance(candidate.get(key), str) and candidate[key].strip()
                                    for key in ("diagnosis", "treatment_plan"))):
                        normalized = {**candidate, "reasoning":""}
                        if self._decision_error(normalized, default, prompt_name, allowed_examinations, already_ordered) is None:
                            parsed, fallback = normalized, True
                            if self.logger is not None:
                                self.logger.write_event(event_type="DOCTOR_JSON_FALLBACK", patient_id=patient_id,
                                    agent_id=getattr(self.runtime_settings, "team_id", "doctor"),
                                    payload={"prompt_name":prompt_name, "reason":"missing_reasoning_recorded_as_empty",
                                        "original_response":original, "errors":errors, "decision":parsed})
                            break
                if self.config.get("json_exam_fallback") and prompt_name in {"exam_item", "exam_category"}:
                    parsed = dict(default)
                    # Preserve every valid selection in any parseable response.
                    if prompt_name == "exam_item" and allowed_examinations is not None:
                        names = build_name_map(allowed_examinations)
                        valid = []
                        for item in errors:
                            partial = parse_json_object(item["response"])
                            values = partial.get("examinations", []) if isinstance(partial, dict) else []
                            if not isinstance(values, list):
                                continue
                            for value in values:
                                name = match_standard_name(value, names) if isinstance(value, str) else None
                                if name and name not in already_ordered and name not in valid:
                                    valid.append(name)
                        parsed["examinations"] = valid or [name for name in allowed_examinations if name not in already_ordered][:1]
                    fallback_error = self._decision_error(parsed, default, prompt_name, allowed_examinations, already_ordered)
                    if fallback_error:
                        raise DoctorOutputPaused("Catalog fallback is invalid: " + fallback_error)
                    fallback = True
                    if self.logger is not None:
                        self.logger.write_event(event_type="DOCTOR_JSON_FALLBACK", patient_id=patient_id,
                            agent_id=getattr(self.runtime_settings, "team_id", "doctor"),
                            payload={"prompt_name": prompt_name, "original_response": original,
                                "errors": errors, "decision": parsed})
                    break
                raise DoctorOutputPaused(f"Doctor output paused after 3 JSON repairs: {prompt_name}; responses and usage retained")
            repair_prompt = (
                "修复下面的医生决策 JSON 格式和字段，不重新运行诊疗。只输出一个合法 JSON 对象。"
                "保留原决策的医学含义，不添加临床事实；字符串内双引号必须转义。\n"
                "必须包含结构中列出的全部字段；原决策未给出的说明字段（reasoning/reason）填空字符串，不编造依据。\n"
                "必要字段结构：" + json.dumps(default, ensure_ascii=False)
                + "\n校验错误：" + error
                + ("\n合法检查名称：" + json.dumps(allowed_examinations, ensure_ascii=False)
                    + "\n已执行检查：" + json.dumps(already_ordered, ensure_ascii=False)
                    if allowed_examinations is not None else "")
                + "\n原始医生决策（只修复此内容）：\n" + original
                + ("\n上次无效修复：\n" + response if attempt else ""))
            response = await self._llm_text(repair_prompt, system_prompt=JSON_REPAIR_SYSTEM_PROMPT,
                temperature=0, prompt_name=f"{prompt_name}_json_repair_{attempt + 1}",
                patient_id=patient_id, **json_options)
        if hasattr(self, "_timeline") and prompt_name != "evaluation_reflection":
            self._timeline.append({"kind": "action", "role": "assistant",
                "text": json.dumps({"phase": prompt_name, "decision": parsed}, ensure_ascii=False),
                "json_repair_count": len(errors) - int(fallback), "json_fallback": fallback})
        return parsed

    async def _memory_notes(self, case_state: Dict[str, Any], *, phase: str) -> List[str]:
        if not self.memory or phase == "initial":
            return []
        examinations = self._examination_context(case_state)
        if hasattr(self.memory, "retrieve"):
            result = self.memory.retrieve(
                patient_id=str(case_state.get("patient_id", "")),
                phase=phase,
                chat_history=case_state.get("chat_history", []),
                examinations=examinations,
                mode=str(case_state.get("mode", "train")),
            )
        elif hasattr(self.memory, "load_notes"):
            result = self.memory.load_notes()
        else:
            return []
        if inspect.isawaitable(result):
            result = await result
        notes = [str(item) for item in (result or []) if str(item).strip()]
        if notes and hasattr(self.memory, "confirm_application"):
            self.memory.confirm_application(str(case_state["patient_id"]), phase=phase)
        return notes

    async def _llm_text(
        self,
        prompt: str,
        *,
        system_prompt: str,
        temperature: float,
        prompt_name: str,
        patient_id: str,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        counts = getattr(self, "_cache_prompt_counts", None)
        if counts is None:
            counts = self._cache_prompt_counts = {}
        key = (patient_id, prompt_name)
        counts[key] = counts.get(key, 0) + 1
        cache_prefix = self.config.get("prompt_cache") == "ephemeral"
        options = {"extra_body": extra_body} if extra_body is not None else {}
        response = await self.llm.chat(
            messages=llm_messages(system_prompt, prompt, cache_prefix=cache_prefix),
            temperature=temperature,
            **options,
        )
        # Write before parsing so malformed outputs and every repair survive.
        self._write_prompt_log(prompt_name=prompt_name, patient_id=patient_id,
                               system_prompt=system_prompt, user_prompt=prompt, response=response.content)
        if self.logger is not None:
            self.logger.write_event(
                event_type="DOCTOR_LLM_USAGE",
                agent_id=getattr(self.runtime_settings, "team_id", "doctor"),
                patient_id=patient_id,
                payload={
                    "prompt_name": prompt_name,
                    "model": getattr(self.llm, "model", "qwen3.5-flash"),
                    "usage": dict(response.usage),
                },
            )
        return response.content

    def _write_prompt_log(
        self,
        *,
        prompt_name: str,
        patient_id: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
    ) -> None:
        if not bool(self.config.get("log_llm_prompts", False)):
            return
        if self.logger is None:
            return
        output_dir = getattr(self.logger, "output_dir", None)
        if output_dir is None:
            return
        prompt_dir = Path(output_dir) / "llm_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        filename = "%s_%s.txt" % (prompt_name, patient_id)
        path = prompt_dir / filename
        content = "\n".join(
            [
                "timestamp: %s" % datetime.now(timezone.utc).astimezone().isoformat(),
                "prompt_name: %s" % prompt_name,
                "patient_id: %s" % patient_id,
                "",
                "system_prompt:",
                system_prompt,
                "",
                "user_prompt:",
                user_prompt,
                "",
                "response:",
                response,
                "",
                "=" * 80,
                "",
            ]
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(content)

    def _examination_context(self, case_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ordered_examinations": unique_preserve_order(
                case_state.get("ordered_examinations", [])
            ),
            "examination_results": case_state.get("examination_results", {}),
        }

    def _evaluation_details(self, evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "diagnosisDetail": evaluation_report.get("diagnosisDetail", {}),
            "examinationDetail": evaluation_report.get("examinationDetail", {}),
            "treatmentDetail": evaluation_report.get("treatmentDetail", {}),
        }


def build_name_map(names: Iterable[str]) -> Dict[str, str]:
    result = {}
    for name in names:
        normalized = normalize_name(name)
        if normalized and normalized not in result:
            result[normalized] = name
    return result


def load_examination_catalog() -> Dict[str, List[str]]:
    data = json.loads((REF_DATA_DIR / "examinations_catalog.json").read_text(encoding="utf-8"))
    catalog: Dict[str, List[str]] = {}
    for category, items in data.get("examinations", {}).items():
        names = []
        for item in items if isinstance(items, list) else []:
            name = item.get("name") if isinstance(item, dict) else item
            if str(name or "").strip():
                names.append(str(name).strip())
        if names:
            catalog[str(category)] = names
    return catalog


def load_disease_catalog() -> Dict[str, List[str]]:
    data = json.loads((REF_DATA_DIR / "diseases_catalog.json").read_text(encoding="utf-8"))
    catalog: Dict[str, List[str]] = {}
    for department, names in data.get("diseases", {}).items():
        clean_names = [str(name).strip() for name in names if str(name).strip()]
        if clean_names:
            catalog[str(department)] = clean_names
    return catalog


def match_standard_name(value: Any, name_map: Dict[str, str]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return name_map.get(normalize_name(text), "")


def as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        value = value.values()
    if not isinstance(value, Iterable):
        return [str(value).strip()] if str(value).strip() else []
    items = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in as_text_list(values):
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = text.strip("`").strip()

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    # Extract one complete object from explanatory text. Do not select between
    # conflicting objects or recover a nested fragment from a malformed object.
    start = text.find("{")
    if start < 0:
        return None
    try:
        parsed, end = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or "{" in text[end:]:
        return None
    return parsed


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[\s\-_/，,。.;；:：、（）()\[\]【】]+", "", text)


def clean_text(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    config = load_config("config.yaml")
    memory = build_memory(config)
    agent = MyDoctorAgent(config=config, memory=memory)
    AgentBuilder(agent).start()
