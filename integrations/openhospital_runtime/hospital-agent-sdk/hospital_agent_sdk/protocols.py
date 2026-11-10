"""Protocols used by doctor actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Protocol, Union


class PatientClientProtocol(Protocol):
    async def invoke(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        """Ask a patient service for a response."""


class ExamClientProtocol(Protocol):
    def get_results(self, patient_id: str, items: Iterable[str]) -> Any:
        """Fetch examination results for a patient."""


class EvaluationClientProtocol(Protocol):
    def evaluate(self, output_log_path: Union[str, Path]) -> Any:
        """Evaluate a final_results.jsonl file or a directory containing it."""

    def evaluate_case(self, final_result: Dict[str, Any]) -> Any:
        """Evaluate one completed final result."""
