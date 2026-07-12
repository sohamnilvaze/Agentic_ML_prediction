from dataclasses import dataclass, field
from typing import Dict

from core.dataset_artifact import DatasetArtifact
from core.dataset_profile import DatasetProfile
from core.model_selection_result import ModelSelectionResult
from core.training_result import TrainingResult


@dataclass
class ModelArtifact:
    """
    Final output of the Model Training Agent.
    """

    dataset_artifact: DatasetArtifact

    dataset_profile: DatasetProfile

    model_selection: ModelSelectionResult

    best_training_result: TrainingResult

    metadata: Dict = field(default_factory=dict)