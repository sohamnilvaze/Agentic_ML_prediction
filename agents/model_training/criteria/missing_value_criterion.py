from core.criterion_score import CriterionScore

from .base_model_criterion import BaseModelCriterion


class MissingValueCriterion(BaseModelCriterion):

    """
    Evaluates model suitability based on
    dataset missing values.
    """

    def compute_score(self, context):

        ratio = context.dataset_profile.overall_missing_ratio

        model = context.candidate_model

        if ratio == 0:

            score = 1.0

            reason = "No missing values."

        elif ratio < 0.05:

            score = 0.90

            reason = "Very few missing values."

        elif ratio < 0.20:

            score = 0.75

            reason = "Moderate missing values."

        elif ratio < 0.40:

            score = 0.55

            reason = "High missing values."

        else:

            score = 0.30

            reason = "Extensive missing values."

        if model.supports_missing_values:

            score = min(

                score + 0.15,

                1.0

            )

            reason += " Model naturally handles missing values."

        return CriterionScore(

            score=round(score, 3),

            passed=(score >= 0.60),

            reasoning=reason

        )