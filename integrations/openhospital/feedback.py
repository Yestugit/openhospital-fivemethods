"""Explicit clinical reward mapping and feedback validity."""
from dataclasses import dataclass
import math
from typing import Mapping


CORE_SCORES = ("diagnosisAccuracy", "examinationPrecision", "treatmentOverallScore")


def valid_score(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


@dataclass(frozen=True)
class FeedbackPolicy:
    threshold: float = 0.5
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        if not valid_score(self.threshold) or len(self.weights) != 3 or any(
                not isinstance(w, (float, int)) or isinstance(w, bool) or not math.isfinite(w) or w < 0 for w in self.weights
        ) or sum(self.weights) <= 0:
            raise ValueError("Feedback threshold/weights are invalid")

    def evaluate(self, report: Mapping):
        report = report.get("report", report)
        status = report.get("evaluationStatus", report.get("status", "evaluated"))
        if status != "evaluated" or any(not valid_score(report.get(k)) for k in CORE_SCORES):
            raise ValueError("Incomplete or failed clinical feedback; learning was not applied")
        reward = sum(report[k] * w for k, w in zip(CORE_SCORES, self.weights)) / sum(self.weights)
        return reward, reward >= self.threshold

    def to_dict(self):
        return {"rule": "weighted_complete_clinical_scores", "weights": self.weights, "threshold": self.threshold}
