from dataclasses import dataclass, field
from typing import Dict, List

from core.training_artifact import TrainingArtifact


@dataclass
class RegistryArtifact:
    """
    Stores information about the saved models.
    """

    training_artifact: TrainingArtifact

    saved_models: List[Dict] = field(
        default_factory=list
    )

    registry_summary: Dict = field(
        default_factory=dict
    )

    metadata: Dict = field(
        default_factory=dict
    )