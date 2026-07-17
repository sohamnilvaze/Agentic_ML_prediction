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

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "task_type": self.task_type,
            "interpretable": self.interpretable,
            "supports_categorical": self.supports_categorical,
            "supports_missing_values": self.supports_missing_values,
            "requires_feature_scaling": self.requires_feature_scaling,
            "supports_multiclass": self.supports_multiclass,
            "produces_probabilities": self.produces_probabilities,
            "training_complexity": self.training_complexity,
            "default_parameters": self.default_parameters,
            "metadata": self.metadata,
        }
