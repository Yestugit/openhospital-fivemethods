from __future__ import annotations

from hospital_service.service.repository import ServiceDataRepository
from hospital_service.service.server import select_patient_ids


def test_balanced_random_selection_spreads_patients_across_departments():
    patient_ids = [
        "cardiology-1",
        "cardiology-2",
        "cardiology-3",
        "neurology-1",
        "neurology-2",
        "neurology-3",
        "dermatology-1",
        "dermatology-2",
        "dermatology-3",
    ]
    patient_departments = {
        patient_id: [patient_id.split("-", 1)[0]]
        for patient_id in patient_ids
    }

    selected = select_patient_ids(
        patient_ids,
        patient_count=6,
        random_seed=7,
        selection="balanced_random",
        patients_by_department={
            "cardiology": ["cardiology-1", "cardiology-2", "cardiology-3"],
            "neurology": ["neurology-1", "neurology-2", "neurology-3"],
            "dermatology": ["dermatology-1", "dermatology-2", "dermatology-3"],
        },
    )

    department_counts: dict[str, int] = {}
    for patient_id in selected:
        department = patient_departments[patient_id][0]
        department_counts[department] = department_counts.get(department, 0) + 1

    assert len(selected) == 6
    assert set(department_counts) == {"cardiology", "neurology", "dermatology"}
    assert max(department_counts.values()) - min(department_counts.values()) <= 1


def test_balanced_random_selection_is_seeded():
    patient_ids = [f"patient-{index}" for index in range(12)]

    first = select_patient_ids(
        patient_ids,
        patient_count=6,
        random_seed=11,
        selection="balanced_random",
        patients_by_department={
            "department-0": ["patient-0", "patient-3", "patient-6", "patient-9"],
            "department-1": ["patient-1", "patient-4", "patient-7", "patient-10"],
            "department-2": ["patient-2", "patient-5", "patient-8", "patient-11"],
        },
    )
    second = select_patient_ids(
        patient_ids,
        patient_count=6,
        random_seed=11,
        selection="balanced_random",
        patients_by_department={
            "department-0": ["patient-0", "patient-3", "patient-6", "patient-9"],
            "department-1": ["patient-1", "patient-4", "patient-7", "patient-10"],
            "department-2": ["patient-2", "patient-5", "patient-8", "patient-11"],
        },
    )

    assert second == first


def test_balanced_random_selection_assigns_comorbid_patients_to_one_bucket():
    patient_ids = [
        "shared-1",
        "shared-2",
        "cardiology-1",
        "cardiology-2",
        "neurology-1",
        "neurology-2",
    ]
    selected = select_patient_ids(
        patient_ids,
        patient_count=6,
        random_seed=19,
        selection="balanced_random",
        patients_by_department={
            "cardiology": ["shared-1", "cardiology-1", "cardiology-2"],
            "neurology": ["shared-2", "neurology-1", "neurology-2"],
        },
    )

    assert len(selected) == 6
    assert len(set(selected)) == 6


def test_repository_builds_department_buckets_once_and_assigns_comorbid_patients_once():
    summaries = [
        {"patientId": "shared-1", "departments": ["cardiology", "neurology"]},
        {"patientId": "cardiology-1", "departments": ["cardiology"]},
        {"patientId": "neurology-1", "departments": ["neurology"]},
        {"patientId": "unknown-1", "departments": []},
    ]

    buckets, unbucketed = ServiceDataRepository._build_patient_department_buckets(
        summaries,
        dataset_key="primary_test",
    )

    bucketed_patients = [
        patient_id
        for patient_ids in buckets.values()
        for patient_id in patient_ids
    ]

    assert "unknown-1" in unbucketed
    assert bucketed_patients.count("shared-1") == 1
    assert set(bucketed_patients) == {"shared-1", "cardiology-1", "neurology-1"}
