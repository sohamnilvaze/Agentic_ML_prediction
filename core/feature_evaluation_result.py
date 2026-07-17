from dataclasses import dataclass
from typing import Dict

from core.criterion_score import CriterionScore


@dataclass
class FeatureEvaluationResult:

    feature_name: str

    overall_score: float

    passed: bool

    criterion_scores: Dict[
        str,
        CriterionScore
    ]

    strongest_criterion: str

    weakest_criterion: str

    summary: str

    def to_dict(self):
        return {
            "feature_name": self.feature_name,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "criterion_scores": {
                name: score.to_dict()
                for name, score in self.criterion_scores.items()
            },
            "strongest_criterion": self.strongest_criterion,
            "weakest_criterion": self.weakest_criterion,
            "summary": self.summary,
        }
