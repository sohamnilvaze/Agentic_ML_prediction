from dataclasses import dataclass
from typing import Dict

from core.criterion_score import CriterionScore
from core.candidate_model import CandidateModel


@dataclass
class ModelEvaluationResult:
    """
    Overall suitability assessment of one candidate model.
    """

    model_name: str

    overall_score: float

    passed: bool

    criterion_scores: Dict[
        str,
        CriterionScore
    ]

    strongest_criterion: str

    weakest_criterion: str

    summary: str

    candidate_model: CandidateModel