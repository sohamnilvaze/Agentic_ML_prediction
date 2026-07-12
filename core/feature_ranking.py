from dataclasses import dataclass, field
from typing import Dict, List

from core.feature_evaluation_result import (
    FeatureEvaluationResult
)


@dataclass
class FeatureRanking:
    """
    Stores the complete ranking of all evaluated features.

    This class is the single source of truth for all
    FeatureEvaluationResult objects.
    """

    ranked_feature_names: List[str]

    feature_results: Dict[
        str,
        FeatureEvaluationResult
    ]

    metadata: Dict = field(default_factory=dict)