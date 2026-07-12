from core.criterion_score import CriterionScore

from core.feature_criteria.base import (
    FeatureCriterion
)


class DistributionCriterion(
    FeatureCriterion
):

    """
    Scoring Policy
    --------------
    These constants define how much each distribution issue
    contributes to the final score deduction.
    """

    # -----------------------------------------------------
    # Pass threshold
    # -----------------------------------------------------

    PASS_THRESHOLD = 0.50

    # -----------------------------------------------------
    # Penalty Weights
    # -----------------------------------------------------

    SINGLETON_RATIO_WEIGHT = 0.40

    RARE_CATEGORY_RATIO_WEIGHT = 0.30

    DOMINANT_VALUE_RATIO_WEIGHT = 0.30

    OUTLIER_RATIO_WEIGHT = 0.30

    ZERO_IQR_PENALTY = 0.50

    @property
    def description(self):

        return (

            "Evaluates whether the feature has a "
            "statistically learnable distribution."

        )

    # --------------------------------------------------------

    def compute_score(
        self,
        context
    ):

        stats = context.statistics

        if stats.is_numeric:

            score = self._numeric_score(stats)

            details = {

                "iqr": stats.iqr,

                "outlier_percentage":
                    stats.outlier_percentage

            }

        else:

            score = self._categorical_score(stats)

            score -= (
                stats.singleton_ratio
                * self.SINGLETON_RATIO_WEIGHT
            )

            score -= (
                stats.rare_category_ratio
                * self.RARE_CATEGORY_RATIO_WEIGHT
            )

            score -= (
                stats.dominant_value_percentage
                * self.DOMINANT_VALUE_RATIO_WEIGHT
            )

            if stats.iqr == 0:
                score -= self.ZERO_IQR_PENALTY

            score -= (
                stats.outlier_percentage
                * self.OUTLIER_RATIO_WEIGHT
            )

            details = {

                "singleton_ratio":
                    stats.singleton_ratio,

                "rare_category_ratio":
                    stats.rare_category_ratio,

                "dominant_value_percentage":
                    stats.dominant_value_percentage

            }

        passed = score >= 0.50

        reasoning = (

            f"Distribution quality score = "

            f"{score:.3f}"

        )

        return CriterionScore(

            score=score,

            passed=passed,

            reasoning=reasoning,

            details=details

        )

    # --------------------------------------------------------

    def _numeric_score(
        self,
        stats
    ):

        score = 1.0

        if stats.iqr == 0:

            score -= 0.50

        score -= (

            stats.outlier_percentage

            * 0.30

        )

        return round(

            max(
                0.0,
                score
            ),

            3

        )

    # --------------------------------------------------------

    def _categorical_score(
        self,
        stats
    ):

        score = 1.0

        score -= (

            stats.singleton_ratio

            * 0.40

        )

        score -= (

            stats.rare_category_ratio

            * 0.30

        )

        score -= (

            stats.dominant_value_percentage

            * 0.30

        )

        return round(

            max(
                0.0,
                score
            ),

            3

        )