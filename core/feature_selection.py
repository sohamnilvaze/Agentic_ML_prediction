from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FeatureSelectionResult:
    """
    Stores one feature-selection iteration.
    """

    iteration: int

    selected_features: List[str] = field(default_factory=list)

    rejected_features: List[str] = field(default_factory=list)

    feature_scores: Dict[str, float] = field(default_factory=dict)

    feature_rankings: List[str] = field(default_factory=list)

    overall_reasoning: str = ""

    metadata: Dict = field(default_factory=dict)

    def to_dict(self):

        return {

            "iteration": self.iteration,
            "selected_features": self.selected_features,
            "rejected_features": self.rejected_features,
            "feature_scores": self.feature_scores,
            "feature_rankings": self.feature_rankings,
            "overall_reasoning": self.overall_reasoning,
            "metadata": self.metadata

        }