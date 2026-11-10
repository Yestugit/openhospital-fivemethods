"""Seeded OpenHospital streams for replay, within-department, and transfer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import file_sha256
from mematlas.evaluation.streams import _ExperienceStreamPlan
from .cohort import OpenHospitalCohort, SPLITS


STREAM_CONDITIONS = ("repeated", "novel_same_department", "novel_departments")


@dataclass(frozen=True)
class OpenHospitalStream:
    plan: _ExperienceStreamPlan
    patient_departments: Mapping[str, str]
    cohorts: tuple[OpenHospitalCohort, ...]

    def freeze(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        hashes: dict[str, str] = {}
        for cohort in self.cohorts:
            hashes[str(cohort.mapping_path)] = file_sha256(cohort.mapping_path)
            if cohort.patients_data_dir:
                for split in SPLITS:
                    for name in ("profiles.jsonl", "patient_departments.json", "ground_truth.json"):
                        source = cohort.patients_data_dir / split / name
                        if source.is_file():
                            hashes[str(source)] = file_sha256(source)
        atomic_write_json(output, {
            "schema_version": 2,
            "selection_rule": "department label list must equal [department]",
            "departments": [cohort.department for cohort in self.cohorts],
            "splits": {"train": self.plan.train_ids, "primary_test": self.plan.primary_test_ids,
                       "advanced_test": self.plan.advanced_test_ids},
            "training_batches": self.plan.training_batches,
            "train_unique_count": len(self.plan.train_ids),
            "train_exposure_count": self.plan.training_count,
            "checkpoints": self.plan.effective_checkpoints(),
            "feedback_boundaries": self.plan.feedback_boundaries,
            "patient_departments": dict(self.patient_departments),
            "stream": dict(self.plan.stream_metadata),
            "source_hashes": hashes,
        })
        return output


def build_openhospital_stream(
    cohorts: Sequence[OpenHospitalCohort], *, condition: str,
    cases_per_block: int = 50, blocks: int = 5,
    primary_per_department: int = 10, advanced_per_department: int = 10,
    seed: int = 0,
    evaluation_cohorts: Sequence[OpenHospitalCohort] | None = None,
    training_only: bool = False,
) -> OpenHospitalStream:
    if condition not in STREAM_CONDITIONS:
        raise ValueError(f"Unknown OpenHospital stream condition: {condition}")
    counts = (cases_per_block, blocks, primary_per_department, advanced_per_department)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in counts):
        raise ValueError("Stream counts must be integers")
    if training_only and (primary_per_department or advanced_per_department):
        raise ValueError("training_only requires empty evaluation splits")
    if min(cases_per_block, blocks) < 1 or primary_per_department < (0 if training_only else 1) or advanced_per_department < 0:
        raise ValueError("Training and primary counts must be positive; advanced count may be zero")
    expected_departments = blocks if condition == "novel_departments" else 1
    if len(cohorts) != expected_departments or len({c.department for c in cohorts}) != len(cohorts):
        raise ValueError(f"{condition} requires {expected_departments} distinct department cohorts")
    panel = tuple(cohorts if evaluation_cohorts is None else evaluation_cohorts)
    if (not panel and not training_only) or len({c.department for c in panel}) != len(panel):
        raise ValueError("Evaluation panel must contain distinct department cohorts")
    all_cohorts = {cohort.department: cohort for cohort in cohorts}
    for cohort in panel:
        existing = all_cohorts.get(cohort.department)
        if existing and any(existing.ids(split) != cohort.ids(split) for split in SPLITS):
            raise ValueError("Training and evaluation disagree on department source rosters")
        all_cohorts[cohort.department] = cohort
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Stream seed must be an integer")
    rng = Random(seed)
    schedule: list[tuple[str, ...]] = []
    train_ids: list[str] = []
    primary_ids: list[str] = []
    advanced_ids: list[str] = []
    labels: dict[str, str] = {}

    def select(cohort: OpenHospitalCohort, split: str, count: int) -> tuple[str, ...]:
        available = sorted(cohort.ids(split))
        if len(available) < count:
            raise ValueError(f"{cohort.department}/{split} has {len(available)} cases; {count} required")
        # Stable by department/split, independent of training condition and
        # panel size. Equal seeds produce exactly the same held-out patients.
        Random(f"{seed}:{cohort.department}:{split}").shuffle(available)
        return tuple(available[:count])

    for cohort in cohorts:
        needed = cases_per_block * blocks if condition == "novel_same_department" else cases_per_block
        chosen = select(cohort, "train", needed)
        for patient_id in chosen:
            if patient_id in labels:
                raise ValueError(f"Patient {patient_id!r} appears in multiple department cohorts/splits")
            labels[patient_id] = cohort.department
        train_ids.extend(chosen)
        if condition == "repeated":
            schedule.extend(tuple(rng.sample(chosen, len(chosen))) for _ in range(blocks))
        elif condition == "novel_same_department":
            schedule.extend(chosen[start:start + cases_per_block] for start in range(0, len(chosen), cases_per_block))
        else:
            schedule.append(chosen)
    for cohort in panel:
        primary = select(cohort, "primary_test", primary_per_department)
        advanced = select(cohort, "advanced_test", advanced_per_department)
        for patient_id in (*primary, *advanced):
            if patient_id in labels:
                raise ValueError(f"Patient {patient_id!r} appears in multiple department cohorts/splits")
            labels[patient_id] = cohort.department
        primary_ids.extend(primary)
        advanced_ids.extend(advanced)
    plan = _ExperienceStreamPlan(
        training_only=training_only,
        train_ids=tuple(train_ids), primary_test_ids=tuple(primary_ids), advanced_test_ids=tuple(advanced_ids),
        batch_count=len(schedule), checkpoints=() if training_only else tuple(index * cases_per_block for index in range(blocks + 1)),
        training_batches=tuple(schedule),
        stream_metadata={"condition": condition, "seed": seed, "cases_per_block": cases_per_block,
                         "blocks": blocks, "departments": [cohort.department for cohort in cohorts],
                         "evaluation_departments": [cohort.department for cohort in panel],
                         "feedback_timing": "after_each_completed_case",
                         "primary_per_department": primary_per_department,
                         "advanced_per_department": advanced_per_department},
    )
    return OpenHospitalStream(plan, labels, tuple(all_cohorts.values()))
