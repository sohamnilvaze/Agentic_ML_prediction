"""
Workflow State

Shared state passed between all agents.

Every agent reads from it,
updates it,
and returns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from core.artifacts import DataArtifact
from core.dataset_artifact import DatasetArtifact
from core.dataset_profile import DatasetProfile
from core.feature_evaluation_result import FeatureEvaluationResult
from core.feature_selection import FeatureSelectionResult
from core.candidate_feature import CandidateFeature
from core.registry_artifact import RegistryArtifact 
from core.training_artifact import TrainingArtifact
from core.explainability_artifact import ExplainabilityArtifact

@dataclass
class WorkflowState:
    """
    Shared workflow state passed between all agents.
    """

    # ============================================================
    # Input Artifacts
    # ============================================================

    master_dataset_path: str = ""

    diagnosis_distribution_path: str = ""

    dataset_profile_path: str = ""

    dataset_summary_path: str = ""

    # ============================================================
    # Task Discovery
    # ============================================================

    candidate_tasks: List[PredictionTask] = field(default_factory=list)

    selected_prediction_task: Optional[PredictionTask] = None

    # ============================================================
    # Dataset Builder
    # ============================================================

    selected_features: List[str] = field(default_factory=list)

    target_column: str = ""

    train_dataset_path: str = ""

    validation_dataset_path: str = ""

    test_dataset_path: str = ""

    # ============================================================
    # Model Strategy
    # ============================================================

    selected_model: str = ""

    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # Training
    # ============================================================

    trained_model_path: str = ""

    training_metrics: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # Evaluation
    # ============================================================

    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)

    evaluation_comments: str = ""

    success: bool = False

    # ============================================================
    # Human Review
    # ============================================================

    human_feedback: str = ""

    # ============================================================
    # Registry
    # ============================================================

    artifacts: List[DataArtifact] = field(default_factory=list)

    # ============================================================
    # Workflow Control
    # ============================================================

    current_stage: str = ""

    dataset_iteration: int = 0

    model_iteration: int = 0

    total_iteration: int = 0

    history: List[str] = field(default_factory=list)

    # ============================================================
    # Dataset Builder
    # ============================================================

    dataset_profile: Optional[DatasetProfile] = None

    feature_evaluations: List[FeatureEvaluationResult] = field(
        default_factory=list
    )

    feature_selection_history: List[
        FeatureSelectionResult
    ] = field(
        default_factory=list
    )

    candidate_features: List[
        CandidateFeature
    ] = field(
        default_factory=list
    )

    dataset_artifact: DatasetArtifact | None = None

    training_artifact: TrainingArtifact | None = None

    explainability_artifact: ExplainabilityArtifact | None = None

    registry_artifact: RegistryArtifact | None = None

    raw_dataset: pd.DataFrame | None = None

    # ============================================================
    # Helper Methods
    # ============================================================

    def clear_feature_evaluations(
        self
    ):

        self.feature_evaluations.clear()

    def add_candidate_feature(
        self,
        feature: CandidateFeature
    ):

        self.candidate_features.append(feature)

    def add_artifact(
        self,
        artifact: DataArtifact
    ):

        self.artifacts.append(artifact)

    def add_history(
        self,
        message: str
    ):

        self.history.append(message)

    def increment_dataset_iteration(self):

        self.dataset_iteration += 1

    def increment_model_iteration(self):

        self.model_iteration += 1

    def increment_total_iteration(self):

        self.total_iteration += 1
    
    def add_feature_selection(
        self,
        selection: FeatureSelectionResult
    ):

        self.feature_selection_history.append(selection)


    def add_feature_evaluation(
        self,
        evaluation: FeatureEvaluationResult
    ):

        self.feature_evaluations.append(evaluation)


@dataclass
class PredictionTask:
    """
    Represents a prediction task discovered
    by the Task Discovery Agent.
    """

    # -----------------------------
    # Identity
    # -----------------------------

    task_name: str

    target_column: str

    target_value: str

    # -----------------------------
    # Statistics
    # -----------------------------

    positive_samples: int

    negative_samples: int

    prevalence: float

    # -----------------------------
    # Suitability Analysis
    # -----------------------------

    suitability_score: float

    criterion_scores: Dict[str, float] = field(default_factory=dict)

    criterion_explanations: Dict[str, str] = field(default_factory=dict)

    reasoning: str = ""

    # -----------------------------
    # Metadata
    # -----------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def disease_name(self) -> str:
        return self.target_value

    @property
    def task_id(self) -> str:
        return self.metadata.get(
            "task_id",
            self.target_value
        )

    @property
    def positive_label(self) -> str:
        return self.target_value

    @property
    def negative_label(self) -> str:
        return self.metadata.get(
            "negative_label",
            f"not_{self.target_value}"
        )

    def to_dict(self):

        return {

            "task_name": self.task_name,

            "target_column": self.target_column,

            "target_value": self.target_value,

            "positive_samples": self.positive_samples,

            "negative_samples": self.negative_samples,

            "prevalence": self.prevalence,

            "suitability_score": self.suitability_score,

            "criterion_scores": self.criterion_scores,

            "criterion_explanations": self.criterion_explanations,

            "reasoning": self.reasoning,

            "metadata": self.metadata

        }
