from core.criterion_score import CriterionScore
from core.feature_criteria.base import FeatureCriterion


class LeakageCriterion(
    FeatureCriterion
):

    PASS_THRESHOLD = 0.80

    DUPLICATE_THRESHOLD = 0.99

    VERY_HIGH_SIMILARITY = 0.95

    HIGH_SIMILARITY = 0.90

    MODERATE_SIMILARITY = 0.80

    @property
    def description(self):

        return (

            "Detects potential target leakage by "
            "comparing the candidate feature with "
            "the prediction target."

        )

    # ----------------------------------------------------

    def compute_score(
        self,
        context
    ):

        feature = context.feature_series
        target = context.target_series

        # ----------------------------------------------

        if (

            context.feature.feature_name

            ==

            context.target_column

        ):

            return CriterionScore(

                score=0.0,

                passed=False,

                reasoning=(
                    "Candidate feature is the target."
                ),

                details={

                    "agreement":1.0

                }

            )

        # ----------------------------------------------

        agreement = (

            feature.fillna("__NA__")

            ==

            target.fillna("__NA__")

        ).mean()

        score = self._score_from_agreement(
            agreement
        )

        return CriterionScore(

            score=score,

            passed=(
                score
                >=
                self.PASS_THRESHOLD
            ),

            reasoning=(

                f"Feature-target agreement = "

                f"{agreement:.3f}"

            ),

            details={

                "agreement":agreement

            }

        )

    # ----------------------------------------------------

    def _score_from_agreement(
        self,
        agreement
    ):

        if agreement >= self.DUPLICATE_THRESHOLD:

            return 0.0

        elif agreement >= self.VERY_HIGH_SIMILARITY:

            return 0.20

        elif agreement >= self.HIGH_SIMILARITY:

            return 0.40

        elif agreement >= self.MODERATE_SIMILARITY:

            return 0.70

        return 1.0