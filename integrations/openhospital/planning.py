"""Pure hospital roster and schedule assembly; no services or Memory allocation."""
from dataclasses import dataclass, replace
from pathlib import Path
from .cohort import OpenHospitalCohort
from .streams import build_openhospital_stream
from .feedback import FeedbackPolicy

@dataclass
class PreparedHospitalPlan:
    plan: object
    labels: dict
    stream: object

    def freeze(self, output_dir):
        target = Path(output_dir) / "frozen_cohort.json"
        return self.stream.freeze(target)

    def summary(self):
        return {"train_unique_count": len(self.plan.train_ids),
                "train_exposure_count": self.plan.training_count,
                "primary_test_count": len(self.plan.primary_test_ids),
                "advanced_test_count": len(self.plan.advanced_test_ids),
                "checkpoints": list(self.plan.effective_checkpoints())}


def build_plan(options, *, checkpoints=(), seed=0):
    def cohort(department):
        return OpenHospitalCohort(options.roster, department=department, patients_data_dir=options.patients_data)

    cohorts = [cohort(department) for department in options.departments]
    stream = build_openhospital_stream(
        cohorts, condition=options.condition, cases_per_block=options.cases_per_block,
        blocks=options.blocks, primary_per_department=options.heldout_per_department,
        advanced_per_department=options.advanced_per_department, seed=seed,
        evaluation_cohorts=[] if options.training_only else [cohort(department) for department in options.evaluation_departments],
        training_only=options.training_only,
    )
    if options.pilot_training_count:
        exposures = tuple(patient for batch in stream.plan.batches() for patient in batch)[:options.pilot_training_count]
        plan = replace(stream.plan, train_ids=tuple(dict.fromkeys(exposures)),
            training_batches=(exposures[:10], exposures[10:]), batch_count=2,
            checkpoints=(20,), feedback_boundaries=(10, 20),
            stream_metadata={**stream.plan.stream_metadata, "pilot": True})
        stream = replace(stream, plan=plan)
    plan = replace(stream.plan, stream_metadata={**stream.plan.stream_metadata,
        "feedback_timing":options.feedback_timing, "feedback_delay":options.feedback_delay,
        "reward_policy":FeedbackPolicy().to_dict()})
    boundaries = None
    if options.feedback_every:
        boundaries = tuple(range(options.feedback_every, plan.training_count, options.feedback_every)) + (plan.training_count,)
    if checkpoints or boundaries:
        plan = plan.with_checkpoints(tuple(checkpoints) if checkpoints else plan.checkpoints,
                                     feedback_boundaries=boundaries)
    stream = replace(stream, plan=plan)
    plan.batches()

    return PreparedHospitalPlan(plan, dict(stream.patient_departments), stream)
