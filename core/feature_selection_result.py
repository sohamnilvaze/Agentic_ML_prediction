from dataclasses import dataclass, field
from typing import Dict, List

from core.feature_ranking import (
    FeatureRanking
)


@dataclass
class FeatureSelectionResult:
    """
    Represents the final outcome of feature selection.
    """

    feature_ranking: FeatureRanking

    selected_feature_names: List[str]

    rejected_feature_names: List[str]

    selection_threshold: float

    overall_summary: str

    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "feature_ranking": {
                "ranked_feature_names": self.feature_ranking.ranked_feature_names,
                "feature_results": {
                    name: result.to_dict()
                    for name, result in self.feature_ranking.feature_results.items()
                },
                "metadata": self.feature_ranking.metadata,
            },
            "selected_feature_names": self.selected_feature_names,
            "rejected_feature_names": self.rejected_feature_names,
            "selection_threshold": self.selection_threshold,
            "overall_summary": self.overall_summary,
            "metadata": self.metadata,
        }
