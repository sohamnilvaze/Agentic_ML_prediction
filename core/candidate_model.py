from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CandidateModel:
    """
    Represents one candidate machine learning algorithm.
    """

    model_name: str

    algorithm: str

    task_type: str

    interpretable: bool

    supports_categorical: bool

    supports_missing_values: bool

    requires_feature_scaling: bool

    supports_multiclass: bool

    produces_probabilities: bool

    training_complexity: str

    default_parameters: Dict

    metadata: Dict = field(default_factory=dict)