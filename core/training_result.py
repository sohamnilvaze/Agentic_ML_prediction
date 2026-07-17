from dataclasses import dataclass, field
from typing import Any, Dict

from core.candidate_model import CandidateModel


@dataclass
class TrainingResult:

    model_name: str

    candidate_model: CandidateModel

    trained_model: Any

    evaluation_metrics: Dict

    cross_validation_metrics: Dict

    training_time_seconds: float

    preprocessing_summary: Dict = field(
        default_factory=dict
    )

    metadata: Dict = field(
        default_factory=dict
    )

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "evaluation_metrics": self.evaluation_metrics,
            "cross_validation_metrics": self.cross_validation_metrics,
            "training_time_seconds": self.training_time_seconds,
            "preprocessing_summary": self.preprocessing_summary,
            "metadata": self.metadata,
            "candidate_model": self.candidate_model.__dict__,
        }
