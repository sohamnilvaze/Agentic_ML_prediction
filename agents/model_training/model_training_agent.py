from agents.base_agent import BaseAgent
from core.dataset_artifact import DatasetArtifact
from core.state import WorkflowState
from core.training_artifact import TrainingArtifact

from .dataset_profiler import DatasetProfiler
from .candidate_model_generator import CandidateModelGenerator
from .model_evaluator import ModelEvaluator
from .model_selector import ModelSelector
from .training_pipeline import TrainingPipeline

# Import your implemented criteria
from .criteria.dataset_size_criterion import DatasetSizeCriterion
from .criteria.missing_value_criterion import MissingValueCriterion


class ModelTrainingAgent(BaseAgent):
    """
    Responsible for selecting, training and evaluating
    machine learning models.

    Workflow
    --------
    DatasetArtifact
            ↓
    DatasetProfiler
            ↓
    CandidateModelGenerator
            ↓
    ModelEvaluator
            ↓
    ModelSelector
            ↓
    TrainingPipeline
            ↓
    TrainingArtifact
    """

    # ---------------------------------------------------------

    def __init__(
        self,
        profiler,
        model_generator,
        evaluator,
        selector,
        training_pipeline
    ):

        self.profiler = profiler

        self.model_generator = model_generator

        self.model_evaluator = evaluator

        self.model_selector = selector

        self.training_pipeline = training_pipeline

    # =========================================================
    # Public API
    # =========================================================

    def run(

        self,

        state: WorkflowState

    ) -> WorkflowState:

        dataset_artifact = state.dataset_artifact

        if dataset_artifact is None:
            raise ValueError("WorkflowState.dataset_artifact is required.")

        dataset_profile = self.profiler.profile(

            dataset_artifact.dataset,

            dataset_artifact.task.target_column

        )

        candidate_models = (

            self.model_generator.generate_candidate_models(

                dataset_profile

            )

        )

        evaluation_results = (

            self.model_evaluator.evaluate_models(

                candidate_models,

                dataset_profile,

                dataset_artifact

            )

        )

        selection_result = (

            self.model_selector.select_models(

                evaluation_results

            )

        )

        training_artifact = (

            self.training_pipeline.train_models(

                selection_result,

                dataset_artifact

            )

        )

        state.training_artifact = training_artifact
        state.current_stage = "MODEL_TRAINING_COMPLETED"
        state.add_history(
            f"Model training completed with {len(training_artifact.training_results)} trained models."
        )

        return state
