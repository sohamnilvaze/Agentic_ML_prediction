from dataclasses import dataclass, field
from typing import Dict

from core.training_artifact import TrainingArtifact


@dataclass
class ExplainabilityArtifact:
    """
    Output of the Explainability Agent.
    """

    training_artifact: TrainingArtifact

    explanations: Dict = field(
        default_factory=dict
    )

    metadata: Dict = field(
        default_factory=dict
    )