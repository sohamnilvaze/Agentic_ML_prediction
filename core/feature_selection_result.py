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