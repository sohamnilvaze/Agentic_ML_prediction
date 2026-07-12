from dataclasses import dataclass, field
from typing import Any, Dict

from core.candidate_model import CandidateModel


@dataclass
class TrainingResult:

    model_name: str

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

    candidate_model: CandidateModel