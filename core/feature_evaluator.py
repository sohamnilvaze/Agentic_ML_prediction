from typing import Dict, List

import pandas as pd

from core.candidate_feature import CandidateFeature
from core.criterion_score import CriterionScore
from core.feature_evaluation_context import FeatureEvaluationContext
from core.feature_evaluation_result import FeatureEvaluationResult
from core.feature_scoring_policy import FeatureScoringPolicy
from core.feature_statistics_builder import FeatureStatisticsBuilder


class FeatureEvaluator:
    """
    Evaluates a single candidate feature.

    Responsibilities
    ----------------
    1. Build feature statistics.
    2. Create evaluation context.
    3. Execute every evaluation criterion.
    4. Aggregate criterion scores.
    """

    def __init__(
        self,
        criteria: List
    ):

        self.criteria = criteria

        self.statistics_builder = FeatureStatisticsBuilder()

        self.scoring_policy = FeatureScoringPolicy()

    # ---------------------------------------------------------

    def evaluate(
        self,
        feature: CandidateFeature,
        dataframe: pd.DataFrame,
        target_column: str
    ) -> FeatureEvaluationResult:

        # -----------------------------------------------------
        # Build statistics
        # -----------------------------------------------------

        statistics = self.statistics_builder.build(

            feature_name=feature.feature_name,

            series=dataframe[
                feature.feature_name
            ]

        )

        # -----------------------------------------------------
        # Create evaluation context
        # -----------------------------------------------------

        context = FeatureEvaluationContext(

            feature=feature,

            dataframe=dataframe,

            feature_series=dataframe[
                feature.feature_name
            ],

            target_series=dataframe[
                target_column
            ],

            target_column=target_column,

            statistics=statistics

        )

        # -----------------------------------------------------
        # Run every criterion
        # -----------------------------------------------------

        criterion_scores = self.run_all_criteria(
            context
        )

        # -----------------------------------------------------
        # Aggregate
        # -----------------------------------------------------

        return self.scoring_policy.compute_final_score(

            feature_name=feature.feature_name,

            criterion_scores=criterion_scores

        )

    # ---------------------------------------------------------

    def run_all_criteria(

        self,

        context: FeatureEvaluationContext

    ) -> Dict[str, CriterionScore]:

        """
        Executes all feature evaluation criteria.
        """

        results = {}

        for criterion in self.criteria:

            score = criterion.compute_score(
                context
            )

            results[
                criterion.__class__.__name__
            ] = score

        return results