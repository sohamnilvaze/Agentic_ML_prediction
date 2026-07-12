from typing import List

from core.feature_evaluation_result import (
    FeatureEvaluationResult
)

from core.feature_ranking import (
    FeatureRanking
)

from core.feature_selection_result import (
    FeatureSelectionResult
)


class FeatureSelector:
    """
    Selects the final feature set from evaluated features.
    """

    DEFAULT_THRESHOLD = 0.70

    def __init__(

        self,

        threshold: float = DEFAULT_THRESHOLD

    ):

        self.threshold = threshold

    # -----------------------------------------------------

    def select_features(

        self,

        feature_results: List[
            FeatureEvaluationResult
        ]

    ) -> FeatureSelectionResult:

        """
        Performs ranking and threshold-based selection.
        """

        ranking = self.build_ranking(
            feature_results
        )

        selected = []
        rejected = []

        for feature_name in ranking.ranked_feature_names:

            result = ranking.feature_results[
                feature_name
            ]

            if result.overall_score >= self.threshold:

                selected.append(
                    feature_name
                )

            else:

                rejected.append(
                    feature_name
                )

        summary = self.build_summary(

            ranking,

            selected,

            rejected

        )

        return FeatureSelectionResult(

            feature_ranking=ranking,

            selected_feature_names=selected,

            rejected_feature_names=rejected,

            selection_threshold=self.threshold,

            overall_summary=summary

        )

    # -----------------------------------------------------

    def build_ranking(

        self,

        feature_results: List[
            FeatureEvaluationResult
        ]

    ) -> FeatureRanking:

        """
        Sorts all evaluated features by descending score.
        """

        ordered = sorted(

            feature_results,

            key=lambda x: x.overall_score,

            reverse=True

        )

        ranked_names = [

            result.feature_name

            for result in ordered

        ]

        result_lookup = {

            result.feature_name: result

            for result in ordered

        }

        return FeatureRanking(

            ranked_feature_names=ranked_names,

            feature_results=result_lookup

        )

    # -----------------------------------------------------

    def build_summary(

        self,

        ranking,

        selected,

        rejected

    ) -> str:

        total = len(
            ranking.ranked_feature_names
        )

        average = sum(

            ranking.feature_results[name].overall_score

            for name in ranking.ranked_feature_names

        ) / max(total, 1)

        return (

            f"{total} candidate features evaluated. "

            f"{len(selected)} selected. "

            f"{len(rejected)} rejected. "

            f"Average score = {average:.3f}. "

            f"Selection threshold = {self.threshold:.2f}."

        )