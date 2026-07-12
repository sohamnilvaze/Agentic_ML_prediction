from core.criterion_score import CriterionScore
from core.feature_criteria.base import FeatureCriterion


class InformationContentCriterion(FeatureCriterion):

    @property
    def description(self):
        return (
            "Evaluates the intrinsic information richness "
            "of the feature using normalized entropy."
        )

    def compute_score(self, context):

        stats = context.statistics

        score = round(stats.entropy, 3)

        passed = score >= 0.40

        reasoning = (
            f"Normalized entropy = {score:.3f}."
        )

        return CriterionScore(

            score=score,

            passed=passed,

            reasoning=reasoning,

            details={
                "normalized_entropy": stats.entropy,
                "unique_count": stats.unique_count,
                "dominant_value_percentage": (
                    stats.dominant_value_percentage
                )
            }

        )