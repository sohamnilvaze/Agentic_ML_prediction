from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.dataset_artifact import DatasetArtifact
from core.model_selection_result import ModelSelectionResult
from core.training_result import TrainingResult


@dataclass
class TrainingArtifact:
    """
    Artifact produced by the complete training stage.
    """

    dataset_artifact: Optional[DatasetArtifact] = None

    model_selection_result: Optional[
        ModelSelectionResult
    ] = None

    training_results: List[
        TrainingResult
    ] = field(default_factory=list)

    best_training_result: Optional[
        TrainingResult
    ] = None

    training_summary: Dict = field(
        default_factory=dict
    )

    metadata: Dict = field(
        default_factory=dict
    )