from typing import Dict

import pandas as pd

from core.candidate_feature import CandidateFeature
from core.feature_statistics_builder import (
    FeatureStatisticsBuilder
)
from core.feature_evaluation_context import (
    FeatureEvaluationContext
)
from core.feature_evaluation_result import (
    FeatureEvaluationResult
)
from core.feature_scoring_policy import (
    FeatureScoringPolicy
)
from core.feature_evaluation_config import FeatureEvaluationConfig
from core.criterion_score import CriterionScore

from core.feature_criteria.missing_value import (
    MissingValueCriterion
)
from core.feature_criteria.information_content import (
    InformationContentCriterion
)
from core.feature_criteria.distribution import (
    DistributionCriterion
)
from core.feature_criteria.cardinality import (
    CardinalityCriterion
)
from core.feature_criteria.duplicate_leakage import (
    DuplicateLeakageCriterion
)


class FeatureEvaluator:
    """
    Orchestrates the complete feature evaluation process.

    Responsibilities
    ----------------
    1. Build feature statistics.
    2. Create FeatureEvaluationContext.
    3. Execute every evaluation criterion.
    4. Aggregate criterion scores.
    5. Return a FeatureEvaluationResult.
    """

    def __init__(self, config=None):

        self.config = config or FeatureEvaluationConfig()

        self.statistics_builder = (
            FeatureStatisticsBuilder()
        )

        self.scoring_policy = (
            FeatureScoringPolicy()
        )

        self.criteria = [

            MissingValueCriterion(config),

            InformationContentCriterion(config),

            DistributionCriterion(config),

            CardinalityCriterion(config),

            DuplicateLeakageCriterion(config)

        ]

    # ---------------------------------------------------------

    def evaluate(

        self,

        feature: CandidateFeature,

        dataframe: pd.DataFrame,

        target_column: str

    ) -> FeatureEvaluationResult:

        """
        Evaluates a single candidate feature.
        """

        # -------------------------------------------------
        # Build statistics
        # -------------------------------------------------

        statistics = self.statistics_builder.build(

            feature_name=feature.feature_name,

            series=dataframe[
                feature.feature_name
            ]

        )

        # -------------------------------------------------
        # Create evaluation context
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Execute all criteria
        # -------------------------------------------------

        criterion_scores = self.run_all_criteria(
            context
        )

        # -------------------------------------------------
        # Aggregate
        # -------------------------------------------------

        evaluation = self.scoring_policy.compute_final_score(

            feature_name=feature.feature_name,

            criterion_scores=criterion_scores

        )

        return evaluation

    # ---------------------------------------------------------

    def run_all_criteria(

        self,

        context: FeatureEvaluationContext

    ) -> Dict[str, CriterionScore]:

        """
        Executes every Feature Criterion.
        """

        criterion_scores = {}

        for criterion in self.criteria:

            score = criterion.compute_score(
                context
            )

            criterion_scores[
                criterion.__class__.__name__
            ] = score

        return criterion_scores
