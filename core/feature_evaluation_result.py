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