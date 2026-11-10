"""Conservative clinical assessment for recorded examination values."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

Assessment = Literal["normal", "abnormal"]

SUPPLEMENTAL_DESCRIPTION_KEY = "补充描述"
INSIGNIFICANT_TERMINAL_PUNCTUATION = "。.!！?？;；"
REFERENCE_MARKER = re.compile(r"(?:参考(?:范围|区间|值)?|正常(?:范围|值)?)[：:]?")
NUMBER = r"(?<![A-Za-z0-9_^.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
NUMBER_PATTERN = re.compile(NUMBER)
CLOSED_RANGE_PATTERN = re.compile(
    rf"(?P<lower>{NUMBER})\s*(?:-|~|～|—|至)\s*(?P<upper>{NUMBER})"
)
BOUND_PATTERN = re.compile(rf"(?P<operator><=|>=|<|>|≤|≥)\s*(?P<bound>{NUMBER})")
UNIT_PATTERN = re.compile(r"[%a-zA-Zμµ/·^0-9\u4e00-\u9fff]+")
SEMIQUANTITATIVE_POSITIVE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+])(?:[1-4]\+|\+{1,4})(?![A-Za-z0-9+])"
)
TEXT_COMPARISON_NOISE = re.compile(r"[\s，,。.;；:：()（）\[\]【】]|可能|通常|一般")
OPPOSITE_TEXT_DIRECTIONS = (
    (("阳性",), ("阴性",)),
    (("异常",), ("正常",)),
    (("增多", "增加", "增高", "升高", "增大", "延长"),
     ("减少", "减低", "降低", "减小", "缩短", "减退")),
)
NEGATED_FINDING = re.compile(r"(?:未(?:见|发现|检出|检测到|引出|触及|提示)|无)")
PRESENT_FINDING = re.compile(r"(?:可见|发现|检出|检测到|引出|触及|提示|存在)")

NORMAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?<!革兰)阴性",
        r"未见(?:明显)?异常",
        r"未见异常浸润或发育异常",
        r"未(?:检出|检测到|发现|提示)(?:异常|阳性|病原|致病|肿瘤|恶性|病毒|细菌|真菌|信号|成分|抗原|抗体)?",
        r"未分离出(?:致病|病原)?(?:菌|微生物)",
        r"低于(?:检出|检测)限",
        r"无(?:明显|局灶性|明确|急性)?异常",
        r"(?:结果|表现|范围|状态)?正常",
        r"未见(?:积液|占位|结节|肿块|病变|异物|出血|梗阻|狭窄|骨折|脱位)",
    )
)
ABNORMAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"阳性",
        r"异常",
        r"(?:明显|显著)?(?:增高|升高|降低|减低)",
        r"(?:明显|显著)?(?:增多|减少)",
        r"(?:高于|低于|超过|超出)(?:正常|参考)?(?:范围|上限|下限|值)?",
        r"(?<!未)(?<!未曾)(?:检出|检测到|发现|提示)(?:致病|病原|异常|阳性|肿瘤|恶性|病毒|细菌|真菌)",
        r"检出(?!限)|检测到",
        r"关节病",
    )
)


@dataclass(frozen=True)
class ExaminationAssessment:
    status: Literal["normal", "abnormal"]
    abnormal_indicators: tuple[str, ...]
    indicator_statuses: dict[str, Assessment]


@dataclass(frozen=True)
class NumericReference:
    lower: float | None
    upper: float | None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    unit: str = ""

    def contains(self, value: float) -> bool:
        if self.lower is not None:
            if value < self.lower or (value == self.lower and not self.lower_inclusive):
                return False
        if self.upper is not None:
            if value > self.upper or (value == self.upper and not self.upper_inclusive):
                return False
        return True


def assess_indicator_value(
    value: Any,
    normal_value: Any = None,
    has_reference: bool = False,
) -> Assessment:
    if has_reference and comparison_key(value) == comparison_key(normal_value):
        return "normal"

    text = _normalize_text(value)
    if not text:
        return "abnormal"
    if has_reference and _is_similar_to_normal_reference(text, normal_value):
        return "normal"

    embedded = _split_embedded_reference(text)
    if embedded is not None:
        measured_text, reference_text = embedded
        polarity = _assess_polarity_against_reference(measured_text, reference_text)
        if polarity is not None:
            return polarity
        qualitative = _assess_qualitative(measured_text)
        if qualitative is not None:
            return qualitative
        if SEMIQUANTITATIVE_POSITIVE_PATTERN.search(measured_text):
            return "abnormal"
        if _equivalent_one_sided_bounds(measured_text, reference_text):
            return "normal"
        numeric = _assess_numeric(measured_text, reference_text, allow_missing_unit=True)
        if numeric is not None:
            return numeric
        return "abnormal"

    qualitative = _assess_qualitative(text)
    if qualitative is not None:
        return qualitative
    if has_reference:
        numeric = _assess_numeric(text, _normalize_text(normal_value), allow_missing_unit=False)
        return numeric if numeric is not None else "abnormal"
    return "abnormal"


def _is_similar_to_normal_reference(value_text: str, normal_value: Any) -> bool:
    reference_text = _normalize_text(normal_value)
    if not reference_text or NUMBER_PATTERN.search(value_text) or NUMBER_PATTERN.search(reference_text):
        return False
    if _has_opposite_direction(value_text, reference_text):
        return False
    value_key = TEXT_COMPARISON_NOISE.sub("", value_text)
    reference_key = TEXT_COMPARISON_NOISE.sub("", reference_text)
    if min(len(value_key), len(reference_key)) < 4:
        return False
    return SequenceMatcher(None, value_key, reference_key).ratio() >= 0.62


def _has_opposite_direction(value_text: str, reference_text: str) -> bool:
    value_presence = _finding_presence(value_text)
    reference_presence = _finding_presence(reference_text)
    if value_presence and reference_presence and value_presence != reference_presence:
        return True
    for left_terms, right_terms in OPPOSITE_TEXT_DIRECTIONS:
        value_left = any(term in value_text for term in left_terms)
        value_right = any(term in value_text for term in right_terms)
        reference_left = any(term in reference_text for term in left_terms)
        reference_right = any(term in reference_text for term in right_terms)
        if (value_left and reference_right) or (value_right and reference_left):
            return True
    return False


def _finding_presence(text: str) -> int:
    if NEGATED_FINDING.search(text):
        return -1
    return 1 if PRESENT_FINDING.search(text) else 0


def assess_examination_result(result: Any, normal_result: Any) -> ExaminationAssessment:
    if not isinstance(result, dict) or not result:
        return ExaminationAssessment(status="abnormal", abnormal_indicators=(), indicator_statuses={})

    normal_fields = normal_result if isinstance(normal_result, dict) else {}
    assessments: list[Assessment] = []
    abnormal_indicators: list[str] = []
    indicator_statuses: dict[str, Assessment] = {}
    for raw_name, value in result.items():
        name = str(raw_name)
        has_reference = name in normal_fields
        assessment = assess_indicator_value(
            value,
            normal_value=normal_fields.get(name),
            has_reference=has_reference,
        )
        assessments.append(assessment)
        if name != SUPPLEMENTAL_DESCRIPTION_KEY:
            indicator_statuses[name] = assessment
        if assessment == "abnormal" and name != SUPPLEMENTAL_DESCRIPTION_KEY:
            abnormal_indicators.append(name)

    status: Literal["normal", "abnormal"] = (
        "normal" if assessments and all(item == "normal" for item in assessments) else "abnormal"
    )
    return ExaminationAssessment(
        status=status,
        abnormal_indicators=tuple(abnormal_indicators),
        indicator_statuses=indicator_statuses,
    )


def comparison_key(value: Any) -> str:
    def normalize(current: Any) -> Any:
        if isinstance(current, dict):
            return {
                str(key): normalize(nested)
                for key, nested in sorted(current.items(), key=lambda item: str(item[0]))
            }
        if isinstance(current, list):
            return [normalize(item) for item in current]
        return _normalize_text(current).rstrip(INSIGNIFICANT_TERMINAL_PUNCTUATION).rstrip()

    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _assess_qualitative(text: str) -> Assessment | None:
    normal_matches = [
        match
        for pattern in NORMAL_PATTERNS
        for match in pattern.finditer(text)
    ]
    if not normal_matches:
        return "abnormal" if _matches_any(text, ABNORMAL_PATTERNS) else None

    clauses = [clause.strip() for clause in re.split(r"[;；]", text) if clause.strip()]
    if len(clauses) > 1:
        unmatched_clauses = [clause for clause in clauses if not _matches_any(clause, NORMAL_PATTERNS)]
        if any(_matches_any(clause, ABNORMAL_PATTERNS) for clause in unmatched_clauses):
            return "abnormal"
        if unmatched_clauses:
            return None

    unnegated = text
    for match in sorted(normal_matches, key=lambda item: item.start(), reverse=True):
        unnegated = unnegated[: match.start()] + (" " * (match.end() - match.start())) + unnegated[match.end() :]
    if _matches_any(unnegated, ABNORMAL_PATTERNS):
        return "abnormal"
    return "normal"


def _normalize_text(value: Any) -> str:
    return (
        unicodedata.normalize("NFKC", str(value if value is not None else ""))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def _split_embedded_reference(text: str) -> tuple[str, str] | None:
    marker = REFERENCE_MARKER.search(text)
    if marker is None:
        return None
    measured = text[: marker.start()].rstrip(" ([【:：")
    reference = text[marker.end() :].lstrip(" :：")
    return (measured, reference) if measured and reference else None


def _assess_numeric(
    measured_text: str,
    reference_text: str,
    *,
    allow_missing_unit: bool,
) -> Assessment | None:
    measured = _parse_single_measurement(measured_text)
    reference = _parse_reference(reference_text)
    if measured is None or reference is None:
        return None
    value, measured_unit = measured
    if measured_unit and reference.unit and measured_unit != reference.unit:
        return None
    if not allow_missing_unit and bool(measured_unit) != bool(reference.unit):
        return None
    within_reference = reference.contains(value)
    if _reference_describes_abnormal_threshold(reference_text):
        return "abnormal" if within_reference else "normal"
    return "normal" if within_reference else "abnormal"


def _reference_describes_abnormal_threshold(reference_text: str) -> bool:
    if re.search(r"(?:为|提示|属于|视为)异常(?:阈值|范围|界值)?", reference_text):
        return True
    return re.search(r"(?:无|未见).{0,40}(?:>=|>|≥)", reference_text) is not None


def _assess_polarity_against_reference(
    measured_text: str,
    reference_text: str,
) -> Assessment | None:
    measured_negative = re.search(r"(?<!革兰)阴性", measured_text) is not None
    measured_positive = "阳性" in measured_text
    reference_negative = re.search(r"(?<!革兰)阴性", reference_text) is not None
    reference_positive = "阳性" in reference_text
    if measured_negative == measured_positive or not (reference_negative or reference_positive):
        return None
    if reference_negative and reference_positive:
        return "normal"
    if measured_negative:
        return "normal" if reference_negative else "abnormal"
    return "normal" if reference_positive else "abnormal"


def _equivalent_one_sided_bounds(measured_text: str, reference_text: str) -> bool:
    measured = BOUND_PATTERN.search(measured_text)
    reference = BOUND_PATTERN.search(reference_text)
    if measured is None or reference is None:
        return False
    if _has_unrelated_numbers(measured_text, measured.span()) or _has_unrelated_numbers(reference_text, reference.span()):
        return False
    measured_operator = measured.group("operator").replace("≤", "<=").replace("≥", ">=")
    reference_operator = reference.group("operator").replace("≤", "<=").replace("≥", ">=")
    if measured_operator != reference_operator or float(measured.group("bound")) != float(reference.group("bound")):
        return False
    measured_unit = _unit_after(measured_text, measured.end())
    reference_unit = _unit_after(reference_text, reference.end())
    return not measured_unit or not reference_unit or measured_unit == reference_unit


def _parse_single_measurement(text: str) -> tuple[float, str] | None:
    matches = list(NUMBER_PATTERN.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    try:
        value = float(match.group())
    except ValueError:
        return None
    return value, _unit_after(text, match.end())


def _parse_reference(text: str) -> NumericReference | None:
    closed = CLOSED_RANGE_PATTERN.search(text)
    if closed is not None:
        if _has_unrelated_numbers(text, closed.span()):
            return None
        lower = float(closed.group("lower"))
        upper = float(closed.group("upper"))
        if lower > upper:
            return None
        return NumericReference(lower=lower, upper=upper, unit=_unit_after(text, closed.end()))

    bound = BOUND_PATTERN.search(text)
    if bound is None or _has_unrelated_numbers(text, bound.span()):
        return None
    number = float(bound.group("bound"))
    operator = bound.group("operator")
    if operator in {"<", "<=", "≤"}:
        return NumericReference(
            lower=None,
            upper=number,
            upper_inclusive=operator in {"<=", "≤"},
            unit=_unit_after(text, bound.end()),
        )
    return NumericReference(
        lower=number,
        upper=None,
        lower_inclusive=operator in {">=", "≥"},
        unit=_unit_after(text, bound.end()),
    )


def _has_unrelated_numbers(text: str, used_span: tuple[int, int]) -> bool:
    return any(
        match.end() <= used_span[0] or match.start() >= used_span[1]
        for match in NUMBER_PATTERN.finditer(text)
    )


def _unit_after(text: str, offset: int) -> str:
    suffix = text[offset:].lstrip()
    match = UNIT_PATTERN.match(suffix)
    if match is None:
        return ""
    unit = match.group().casefold().replace("µ", "μ")
    return unit.rstrip("。.,;:)]}）】")
