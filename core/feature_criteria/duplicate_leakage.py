import pandas as pd

from core.criterion_score import CriterionScore
from core.feature_criteria.base import (
    FeatureCriterion
)


class DuplicateLeakageCriterion(
    FeatureCriterion
):
    """
    Detects direct or near-duplicate leakage between
    the candidate feature and the prediction target.

    NOTE
    ----
    This criterion intentionally detects only
    statistical duplication.

    Temporal or causal leakage is outside the
    scope of the current POC.
    """

    PASS_THRESHOLD = 0.80

    EXACT_DUPLICATE_THRESHOLD = 0.99

    VERY_HIGH_SIMILARITY = 0.95

    HIGH_SIMILARITY = 0.90

    MODERATE_SIMILARITY = 0.80

    @property
    def description(self):

        return (
            "Evaluates whether the feature is a "
            "duplicate or near-duplicate of the "
            "prediction target."
        )

    # ---------------------------------------------------------

    def compute_score(
        self,
        context
    ):

        feature = context.feature_series

        target = context.target_series

        # ------------------------------------------
        # Case 1
        # Same column
        # ------------------------------------------

        if (
            context.feature.feature_name
            ==
            context.target_column
        ):

            return CriterionScore(

                score=0.0,

                passed=False,

                reasoning=(
                    "Feature is identical to the target column."
                ),

                details={
                    "agreement":1.0,
                    "leakage_type":"same_column"
                }

            )

        # ------------------------------------------
        # Case 2
        # Compare values
        # ------------------------------------------

        agreement = self.compute_agreement(

            feature,

            target

        )

        score = self.compute_score_from_agreement(

            agreement

        )

        return CriterionScore(

            score=score,

            passed=(
                score >= self.PASS_THRESHOLD
            ),

            reasoning=(

                f"Feature-target agreement "

                f"= {agreement:.3f}"

            ),

            details={

                "agreement":agreement,

                "leakage_type":"duplicate_check"

            }

        )

    # ---------------------------------------------------------

    def compute_agreement(
        self,
        feature: pd.Series,
        target: pd.Series
    ) -> float:
        """
        Computes row-wise agreement after normalizing
        both series to a common representation.
        """

        feature = feature.copy()
        target = target.copy()

        # ------------------------------------------
        # Normalize missing values
        # ------------------------------------------

        feature = feature.fillna("__NULL__")
        target = target.fillna("__NULL__")

        # ------------------------------------------
        # Convert to common representation
        # ------------------------------------------

        feature = feature.astype(str).str.strip()

        target = target.astype(str).str.strip()

        # ------------------------------------------
        # Compare
        # ------------------------------------------

        agreement = (

            feature
            ==
            target

        ).mean()

        return float(agreement)
    # ---------------------------------------------------------

    def compute_score_from_agreement(

        self,

        agreement

    ):

        if agreement >= self.EXACT_DUPLICATE_THRESHOLD:

            return 0.0

        elif agreement >= self.VERY_HIGH_SIMILARITY:

            return 0.20

        elif agreement >= self.HIGH_SIMILARITY:

            return 0.40

        elif agreement >= self.MODERATE_SIMILARITY:

            return 0.70

        return 1.0