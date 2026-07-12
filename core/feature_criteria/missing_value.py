from core.criterion_score import CriterionScore
from core.feature_criteria.base import FeatureCriterion


class MissingValueCriterion(FeatureCriterion):

    @property
    def description(self):
        return (
            "Evaluates feature completeness based on the "
            "percentage of missing values."
        )

    def compute_score(self, context):

        stats = context.statistics

        missing = stats.missing_percentage * 100
        threshold = self.config.missing_threshold

        if missing == 0:
            score = 1.0

        elif missing <= threshold:
            score = max(
                0.70,
                1.0 - (missing / threshold) * 0.30
            )

        else:
            overflow = missing - threshold

            score = max(
                0.0,
                0.70 - (
                    overflow / (100 - threshold)
                ) * 0.70
            )

        return CriterionScore(

            score=round(score, 3),

            passed=(missing <= threshold),

            reasoning=(
                f"{missing:.2f}% values are missing "
                f"(threshold={threshold:.1f}%)."
            ),

            details={
                "missing_percentage": missing,
                "threshold": threshold
            }

        )