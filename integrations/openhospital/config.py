"""Hospital scene settings carried by ExperimentConfig.benchmark."""
from dataclasses import dataclass, fields
from pathlib import Path

from mematlas.evaluation.feedback import FeedbackSchedule


OPENHOSPITAL_ROOT = Path(__file__).resolve().parents[1] / "openhospital_runtime"
SERVICE_URLS = {split: f"http://127.0.0.1:{port}" for split, port in
                (("train", 8001), ("primary_test", 8002), ("advanced_test", 8003))}


@dataclass(frozen=True)
class HospitalOptions:
    condition: str = "repeated"
    departments: tuple[str, ...] = ("心内科",)
    evaluation_departments: tuple[str, ...] = ("心内科", "呼吸内科", "消化内科", "内分泌科", "神经内科")
    cases_per_block: int = 50
    blocks: int = 5
    heldout_per_department: int = 10
    advanced_per_department: int = 10
    doctor_config: str | None = None
    hospital_root: str = str(OPENHOSPITAL_ROOT)
    path: str | None = None
    world_dir: str = ".mematlas/hospital-world"
    feedback_timing: str = "immediate"
    feedback_delay: int = 0
    feedback_every: int | None = None
    pilot_training_count: int | None = None
    training_only: bool = False
    snapshot_keep_recent: int | None = None
    snapshot_keep_every: int = 50
    memory_top_k: int | None = None
    service_port: int | None = None

    @property
    def service_urls(self):
        if self.service_port is None:
            return dict(SERVICE_URLS)
        return {split: f"http://127.0.0.1:{self.service_port + index}" for index, split in enumerate(SERVICE_URLS)}

    def __post_init__(self):
        if self.memory_top_k is not None and (type(self.memory_top_k) is not int or self.memory_top_k < 1):
            raise ValueError("memory_top_k must be a positive integer or None")

    @classmethod
    def from_experiment(cls, config):
        values = dict(config.benchmark)
        values.pop("kind", None)
        unknown = values.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown hospital options: {sorted(unknown)}")
        for key in ("departments", "evaluation_departments"):
            if key in values:
                if isinstance(values[key], str):
                    raise ValueError(f"{key} must be a sequence of department names")
                values[key] = tuple(values[key])
        result = cls(**values)
        if result.pilot_training_count is not None and (result.condition != "repeated" or result.pilot_training_count != 20):
            raise ValueError("pilot_training_count supports the 20-case repeated pilot only")
        if type(result.training_only) is not bool:
            raise ValueError("training_only must be a boolean")
        for name in ("snapshot_keep_recent", "snapshot_keep_every"):
            value = getattr(result, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if result.training_only and (result.heldout_per_department or result.advanced_per_department):
            raise ValueError("training_only requires zero heldout and advanced counts")
        FeedbackSchedule(result.feedback_timing, result.feedback_delay)
        if result.feedback_every is not None and (
                type(result.feedback_every) is not int or result.feedback_every < 1 or result.feedback_timing != "batch"):
            raise ValueError("feedback_every requires batch feedback and a positive integer")
        return result

    @property
    def root(self):
        return Path(self.hospital_root).expanduser().resolve()

    @property
    def roster(self):
        return Path(self.path).expanduser().resolve() if self.path else self.root / "department_patient_ids.json"

    @property
    def doctor(self):
        return Path(self.doctor_config).expanduser().resolve() if self.doctor_config else self.root / "hospital_agent_example/config.yaml"

    @property
    def patients_data(self):
        return self.root / "hospital_service/hospital_service/service/data/patients_data"

    @property
    def splits(self):
        if self.training_only:
            return ("train",)
        return ("train", "primary_test", "advanced_test") if self.advanced_per_department else ("train", "primary_test")
