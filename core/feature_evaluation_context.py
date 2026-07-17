from dataclasses import dataclass
import pandas as pd

from core.candidate_feature import CandidateFeature
from core.feature_statistics import FeatureStatistics


@dataclass
class FeatureEvaluationContext:
    """
    Context shared with every feature evaluation criterion.
    """

    feature: CandidateFeature

    dataframe: pd.DataFrame

    feature_series: pd.Series

    target_series: pd.Series

    target_column: str

    statistics: FeatureStatistics

    @property
    def statisttics(self) -> FeatureStatistics:
        """
        Backward-compatible alias for the old misspelled field name.
        """

        return self.statistics
