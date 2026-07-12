from typing import Dict

from core.criterion_score import CriterionScore
from core.model_evaluation_result import (
    ModelEvaluationResult
)


class ModelScoringPolicy:
    """
    Aggregates the outputs of all Model Evaluation
    Criteria into a single suitability score.
    """

    DEFAULT_WEIGHTS = {

        "DatasetSizeCriterion": 0.15,

        "MissingValueCriterion": 0.15,

        "FeatureTypeCriterion": 0.20,

        "InterpretabilityCriterion": 0.10,

        "ComplexityCriterion": 0.15,

        "ClassImbalanceCriterion": 0.15,

        "ScalabilityCriterion": 0.10

    }

    PASS_THRESHOLD = 0.70

    # -----------------------------------------------------

    def compute_final_score(

        self,

        candidate_model,

        criterion_scores

    ):

        weighted_sum = 0

        total_weight = 0

        reasoning = []

        strongest = None

        weakest = None

        max_score = -1

        min_score = 2

        for name, score in criterion_scores.items():

            weight = self.DEFAULT_WEIGHTS.get(

                name,

                0

            )

            weighted_sum += weight * score.score

            total_weight += weight

            reasoning.append(

                f"{name}: {score.score:.2f}"

            )

            if score.score > max_score:

                max_score = score.score

                strongest = name

            if score.score < min_score:

                min_score = score.score

                weakest = name

        final_score = (

            weighted_sum / total_weight

            if total_weight

            else 0

        )

        return ModelEvaluationResult(

            candidate_model=candidate_model,

            model_name=candidate_model.model_name,

            overall_score=round(

                final_score,

                3

            ),

            passed=(

                final_score >= self.PASS_THRESHOLD

            ),

            criterion_scores=criterion_scores,

            strongest_criterion=strongest,

            weakest_criterion=weakest,

            summary="; ".join(reasoning)

        )