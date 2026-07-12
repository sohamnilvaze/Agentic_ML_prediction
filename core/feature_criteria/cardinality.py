from core.criterion_score import CriterionScore
from core.feature_criteria.base import FeatureCriterion


class CardinalityCriterion(FeatureCriterion):

    """
    Evaluates whether a feature has a suitable number of
    unique values for efficient ML encoding.
    """

    PASS_THRESHOLD = 0.60

    VERY_LOW_RATIO = 0.10
    LOW_RATIO = 0.30
    MEDIUM_RATIO = 0.50
    HIGH_RATIO = 0.80

    @property
    def description(self):

        return (
            "Evaluates encoding feasibility based on "
            "feature cardinality."
        )

    # --------------------------------------------------------

    def compute_score(self, context):

        stats = context.statistics

        # Numeric features are not penalized here.
        if stats.is_numeric:

            return CriterionScore(

                score=1.0,

                passed=True,

                reasoning=(
                    "Continuous numeric feature."
                ),

                details={
                    "is_numeric": True
                }

            )

        unique_ratio = (

            stats.unique_count

            /

            max(
                stats.non_missing_count,
                1
            )

        )

        score = self._score(unique_ratio)

        return CriterionScore(

            score=score,

            passed=(
                score >= self.PASS_THRESHOLD
            ),

            reasoning=(

                f"Unique ratio = "

                f"{unique_ratio:.3f}"

            ),

            details={

                "unique_count":
                    stats.unique_count,

                "unique_ratio":
                    unique_ratio

            }

        )

    # --------------------------------------------------------

    def _score(
        self,
        ratio
    ):

        if ratio <= self.VERY_LOW_RATIO:

            return 1.0

        elif ratio <= self.LOW_RATIO:

            return 0.80

        elif ratio <= self.MEDIUM_RATIO:

            return 0.60

        elif ratio <= self.HIGH_RATIO:

            return 0.40

        else:

            return 0.20