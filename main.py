import pandas as pd
from datetime import datetime, timedelta
import random

from agents.dataset_builder import (
    DatasetBuilderAgent,
    DatasetReportBuilder,
    DatasetValidator,
    FeatureEvaluator,
)
from agents.explainability import (
    ExplainabilityAgent,
    ExplainabilityPipeline,
)
from agents.model_registry import (
    ModelRegistryAgent,
    ModelSaver,
)
from agents.model_training import (
    CandidateModelGenerator,
    DatasetProfiler,
    ModelEvaluator,
    ModelSelector,
    ModelTrainingAgent,
    TrainingPipeline,
)
from agents.task_discovery import TaskDiscoveryAgent
from core.config import (
    DATASET_PROFILE_FILE,
    DATASET_SUMMARY_FILE,
    DIAGNOSIS_DISTRIBUTION_FILE,
    PATIENTS_FILE,
    ADMISSIONS_FILE,
    MASTER_DATASET_FILE,
)
from core.dataset_construction import DatasetConstructor
from core.feature_evaluation_config import FeatureEvaluationConfig
from core.feature_selector import FeatureSelector
from core.preprocessing import MasterDatasetPreprocessor
from core.state import WorkflowState
from core.constants import WorkflowStage
from core.utils import assert_file_exists, log_info
from core.utils import ensure_directory_exists
from human.reviewer import HumanReviewAgent
from workflow.workflow_engine import WorkflowEngine

from agents.model_training.criteria.dataset_size_criterion import (
    DatasetSizeCriterion,
)
from agents.model_training.criteria.missing_value_criterion import (
    MissingValueCriterion,
)


def _prepare_master_artifacts() -> None:
    """
    Ensure the master dataset and preprocessing artifacts exist.
    """

    required_files = [
        MASTER_DATASET_FILE,
        DIAGNOSIS_DISTRIBUTION_FILE,
        DATASET_PROFILE_FILE,
        DATASET_SUMMARY_FILE,
    ]

    if all(path.exists() for path in required_files):
        return

    _bootstrap_demo_raw_data()
    preprocessor = MasterDatasetPreprocessor()
    preprocessor.run_pipeline()


def _bootstrap_demo_raw_data() -> None:
    """
    Create a small synthetic MIMIC-like dataset when raw inputs are missing.
    This keeps the end-to-end workflow runnable in a clean workspace.
    """

    if PATIENTS_FILE.exists() and ADMISSIONS_FILE.exists():
        return

    ensure_directory_exists(PATIENTS_FILE.parent)

    rng = random.Random(42)
    n_patients = 300
    n_admissions = 1200

    patient_ids = list(range(10000, 10000 + n_patients))

    patients = []
    for subject_id in patient_ids:
        gender = rng.choice(["M", "F"])
        dob_year = rng.randint(1930, 1990)
        dob_month = rng.randint(1, 12)
        dob_day = rng.randint(1, 28)
        patients.append(
            {
                "subject_id": subject_id,
                "gender": gender,
                "dob": datetime(dob_year, dob_month, dob_day).strftime("%Y-%m-%d"),
            }
        )

    diagnoses = [
        ("Sepsis", 500),
        ("Pneumonia", 400),
        ("Heart Failure", 300),
    ]

    diagnosis_pool = []
    for diagnosis, count in diagnoses:
        diagnosis_pool.extend([diagnosis] * count)

    admissions = []
    for hadm_id, diagnosis in enumerate(diagnosis_pool, start=200000):
        subject_id = rng.choice(patient_ids)
        admit_year = rng.randint(2010, 2014)
        admit_month = rng.randint(1, 12)
        admit_day = rng.randint(1, 28)
        admit_hour = rng.randint(0, 23)
        admit_minute = rng.randint(0, 59)
        admit_time = datetime(
            admit_year,
            admit_month,
            admit_day,
            admit_hour,
            admit_minute,
        )
        length_of_stay_days = rng.randint(1, 10)
        discharge_time = admit_time + timedelta(days=length_of_stay_days)

        admissions.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "admittime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "dischtime": discharge_time.strftime("%Y-%m-%d %H:%M:%S"),
                "admission_type": rng.choice(["EMERGENCY", "ELECTIVE", "URGENT"]),
                "admission_location": rng.choice(
                    ["EMERGENCY ROOM", "CLINIC REFERRAL", "TRANSFER"]
                ),
                "discharge_location": rng.choice(
                    ["HOME", "SNF", "REHAB", "EXPIRED"]
                ),
                "insurance": rng.choice(["Medicare", "Private", "Medicaid"]),
                "language": rng.choice(["EN", "ES", "PT"]),
                "religion": rng.choice(["CATHOLIC", "PROTESTANT", "JEWISH", "NONE"]),
                "marital_status": rng.choice(["MARRIED", "SINGLE", "DIVORCED"]),
                "ethnicity": rng.choice(
                    ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]
                ),
                "diagnosis": diagnosis,
                "hospital_expire_flag": rng.choice([0, 1]),
            }
        )

    pd.DataFrame(patients).to_csv(PATIENTS_FILE, index=False)
    pd.DataFrame(admissions).to_csv(ADMISSIONS_FILE, index=False)
    log_info(
        f"Bootstrapped synthetic raw data: {len(patients)} patients, {len(admissions)} admissions."
    )


def build_workflow_state() -> WorkflowState:
    state = WorkflowState()
    state.master_dataset_path = str(MASTER_DATASET_FILE)
    state.diagnosis_distribution_path = str(DIAGNOSIS_DISTRIBUTION_FILE)
    state.dataset_profile_path = str(DATASET_PROFILE_FILE)
    state.dataset_summary_path = str(DATASET_SUMMARY_FILE)
    state.current_stage = WorkflowStage.TASK_DISCOVERY.value
    state.raw_dataset = pd.read_csv(MASTER_DATASET_FILE)
    return state


def build_workflow_engine() -> WorkflowEngine:
    log_info("initializing feature evaluator")
    feature_evaluator = FeatureEvaluator(config=FeatureEvaluationConfig())
    log_info("initializing dataset builder agent")
    dataset_builder_agent = DatasetBuilderAgent(
        feature_evaluator=feature_evaluator,
        feature_selector=FeatureSelector(),
        dataset_constructor=DatasetConstructor(),
        dataset_validator=DatasetValidator(),
        report_builder=DatasetReportBuilder(),
    )
    log_info("initializing model evaluator")
    model_evaluator = ModelEvaluator(
        criteria=[
            DatasetSizeCriterion(),
            MissingValueCriterion(),
        ]
    )
    log_info("initializing model training agent")
    model_training_agent = ModelTrainingAgent(
        profiler=DatasetProfiler(),
        model_generator=CandidateModelGenerator(),
        evaluator=model_evaluator,
        selector=ModelSelector(),
        training_pipeline=TrainingPipeline(),
    )
    log_info("initializing explainability agent")
    explainability_agent = ExplainabilityAgent(
        pipeline=ExplainabilityPipeline()
    )
    log_info("initializing fregistry agent")
    registry_agent = ModelRegistryAgent(
        model_saver=ModelSaver()
    )
    log_info("initializing completed")

    return WorkflowEngine(
        [
            TaskDiscoveryAgent(),
            HumanReviewAgent(),
            dataset_builder_agent,
            model_training_agent,
            explainability_agent,
            registry_agent,
        ]
    )


def main() -> WorkflowState:
    _prepare_master_artifacts()

    for path in [
        MASTER_DATASET_FILE,
        DIAGNOSIS_DISTRIBUTION_FILE,
        # DATASET_PROFILE_FILE,
        DATASET_SUMMARY_FILE,
    ]:
        assert_file_exists(path)

    print("Building the workflow state")
    state = build_workflow_state()
    print("Building the workflow engine")
    engine = build_workflow_engine()
    print("Running the engine")
    final_state = engine.run(state)

    log_info(
        f"Pipeline completed successfully. Final stage: {final_state.current_stage}"
    )
    return final_state


if __name__ == "__main__":
    main()
