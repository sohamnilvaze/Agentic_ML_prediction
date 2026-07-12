from typing import Dict

from core.criterion_score import CriterionScore
from core.feature_evaluation_result import FeatureEvaluationResult


class FeatureScoringPolicy:

    DEFAULT_WEIGHTS = {

        "MissingValueCriterion": 0.20,

        "InformationContentCriterion": 0.25,

        "DistributionCriterion": 0.20,

        "CardinalityCriterion": 0.15,

        "DuplicateLeakageCriterion": 0.20

    }

    PASS_THRESHOLD = 0.70

    # -----------------------------------------------------

    def compute_final_score(

        self,

        feature_name: str,

        criterion_scores: Dict[
            str,
            CriterionScore
        ]

    ) -> FeatureEvaluationResult:

        weighted_sum = 0.0
        total_weight = 0.0

        reasoning = []

        for criterion_name, result in criterion_scores.items():

            weight = self.DEFAULT_WEIGHTS.get(
                criterion_name,
                0.0
            )

            weighted_sum += weight * result.score
            total_weight += weight

            reasoning.append(

                f"{criterion_name}: {result.score:.2f}"

            )

        if total_weight == 0:

            overall_score = 0.0

        else:

            overall_score = (

                weighted_sum

                /

                total_weight

            )

        strongest = max(

            criterion_scores,

            key=lambda x: criterion_scores[x].score

        )

        weakest = min(

            criterion_scores,

            key=lambda x: criterion_scores[x].score

        )

        return FeatureEvaluationResult(

            feature_name=feature_name,

            overall_score=round(
                overall_score,
                3
            ),

            passed=(
                overall_score
                >=
                self.PASS_THRESHOLD
            ),

            criterion_scores=criterion_scores,

            strongest_criterion=strongest,

            weakest_criterion=weakest,

            summary="; ".join(reasoning)

        )