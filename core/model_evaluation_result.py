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

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "criterion_scores": {
                name: score.to_dict()
                for name, score in self.criterion_scores.items()
            },
            "strongest_criterion": self.strongest_criterion,
            "weakest_criterion": self.weakest_criterion,
            "summary": self.summary,
            "candidate_model": self.candidate_model.to_dict()
            if hasattr(self.candidate_model, "to_dict")
            else self.candidate_model.__dict__,
        }
