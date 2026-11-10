"""Doctor-visible actions for OpenHospital contest agents."""

from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .event_logger import EventLogger
from .protocols import EvaluationClientProtocol, ExamClientProtocol, PatientClientProtocol


class DoctorActions:
    """Action facade used by participant-defined doctor logic."""

    def __init__(
        self,
        patient_client: PatientClientProtocol,
        exam_client: ExamClientProtocol,
        evaluate_client: EvaluationClientProtocol,
        event_logger: EventLogger,
        team_id: str,
    ):
        self.patient_client = patient_client
        self.exam_client = exam_client
        self.evaluate_client = evaluate_client
        self.event_logger = event_logger
        self.team_id = team_id
        self.question_count_by_patient: Dict[str, int] = defaultdict(int)
        self.ordered_exams_by_patient: Dict[str, List[str]] = defaultdict(list)

    async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        """Ask the patient through /invoke using a free-form input object."""
        if not isinstance(input_data, dict):
            raise TypeError("ask_patient input_data must be a JSON object/dict.")

        self.event_logger.write_event(
            event_type="SEND_MESSAGE",
            agent_id=self.team_id,
            patient_id=patient_id,
            payload={
                "patient_id": patient_id,
                "target": patient_id,
                "message_type": "doctor_to_patient",
                "input": input_data,
            },
        )
        answer = await self.patient_client.invoke(patient_id=patient_id, input_data=input_data)
        self.question_count_by_patient[patient_id] += 1
        self.event_logger.write_event(
            event_type="PATIENT_REPLY",
            agent_id=patient_id,
            patient_id=patient_id,
            payload={
                "patient_id": patient_id,
                "target": self.team_id,
                "content": answer,
                "message_type": "patient_to_doctor",
            },
        )
        return answer

    async def order_examination(
        self,
        patient_id: str,
        items: Iterable[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        """Order examinations and return configured examination results."""
        item_list = [str(item).strip() for item in items if str(item).strip()]
        self.event_logger.write_event(
            event_type="SCHEDULE_EXAMINATION",
            agent_id=self.team_id,
            patient_id=patient_id,
            payload={
                "patient_id": patient_id,
                "items": item_list,
                "reason": reason,
            },
        )

        raw_exam_response = self.exam_client.get_results(patient_id=patient_id, items=item_list)
        if inspect.isawaitable(raw_exam_response):
            exam_response = await raw_exam_response
        else:
            exam_response = raw_exam_response

        results = exam_response["results"]
        normalized_items = [str(item) for item, row in results.items() if row.get("status") != "invalid"]
        invalid_items = [str(item) for item, row in results.items() if row.get("status") == "invalid"]
        self.ordered_exams_by_patient[patient_id].extend(normalized_items)

        if invalid_items:
            self.event_logger.write_event(
                event_type="INVALID_EXAMINATION",
                agent_id=self.team_id,
                patient_id=patient_id,
                status="failed",
                payload={
                    "patient_id": patient_id,
                    "items": invalid_items,
                    "reason": "无效检查",
                },
            )

        self.event_logger.write_event(
            event_type="DO_EXAMINATION",
            agent_id=patient_id,
            patient_id=patient_id,
            payload={
                "patient_id": patient_id,
                "items": normalized_items,
                "invalid_items": invalid_items,
                "results_count": len(results),
                "results": results,
            },
        )
        return exam_response

    async def prescribe_treatment(
        self,
        patient_id: str,
        diagnosis: Any,
        treatment_plan: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """Record treatment. This action finishes the current patient encounter."""
        ordered_exams = self._unique_preserve_order(self.ordered_exams_by_patient[patient_id])
        conversation_rounds = self.question_count_by_patient[patient_id]

        final_result = {
            "patient_id": patient_id,
            "team_id": self.team_id,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
            "ordered_examinations": ordered_exams,
            "conversation_rounds": conversation_rounds,
            "finished": True,
        }
        self.event_logger.write_event(
            event_type="PRESCRIBE_TREATMENT",
            agent_id=self.team_id,
            patient_id=patient_id,
            payload=final_result,
        )
        self.event_logger.write_final_result(final_result)
        return final_result

    async def evaluation(
        self,
        patient_id: str,
        final_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate one completed patient encounter.

        This is intended for train-mode reflection after `prescribe_treatment`.
        The contest service enforces evaluation-token access, so deployed
        silicon test tokens cannot call this action successfully.
        """
        if final_result is None:
            final_result = self.event_logger.latest_final_result(patient_id)
        if not isinstance(final_result, dict):
            raise TypeError("evaluation final_result must be a JSON object/dict.")
        if str(final_result.get("patient_id") or "").strip() != str(patient_id):
            final_result = {**final_result, "patient_id": patient_id}

        self.event_logger.write_event(
            event_type="EVALUATION_REQUEST",
            agent_id=self.team_id,
            patient_id=patient_id,
            payload={
                "patient_id": patient_id,
                "final_result": final_result,
            },
        )
        raw_report = self.evaluate_client.evaluate_case(final_result)
        if inspect.isawaitable(raw_report):
            report = await raw_report
        else:
            report = raw_report
        if not isinstance(report, dict):
            report = {"response": report}

        self.event_logger.write_event(
            event_type="EVALUATION_RESULT",
            agent_id=self.team_id,
            patient_id=patient_id,
            payload={
                "patient_id": patient_id,
                "report": report,
            },
        )
        self.event_logger.write_evaluation_result(patient_id=patient_id, report=report)
        return report

    async def batch_evaluation(self, output_log_path: str | Path | None = None) -> Dict[str, Any]:
        """Evaluate a batch of final results from a test/train output directory or JSONL file."""
        if output_log_path is None:
            output_log_path = self.event_logger.final_results_path

        self.event_logger.write_event(
            event_type="BATCH_EVALUATION_REQUEST",
            agent_id=self.team_id,
            patient_id="SYSTEM",
            payload={
                "output_log_path": str(output_log_path),
            },
        )
        raw_report = self.evaluate_client.evaluate(output_log_path)
        if inspect.isawaitable(raw_report):
            report = await raw_report
        else:
            report = raw_report
        if not isinstance(report, dict):
            report = {"response": report}

        self.event_logger.write_event(
            event_type="BATCH_EVALUATION_RESULT",
            agent_id=self.team_id,
            patient_id="SYSTEM",
            payload={
                "output_log_path": str(output_log_path),
                "report": report,
            },
        )
        return report

    @staticmethod
    def _unique_preserve_order(values: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
