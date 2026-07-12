from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SuitabilityResult:
    """
    Result returned after evaluating
    one prediction task.
    """

    final_score: float

    criterion_scores: Dict[str, float] = field(default_factory=dict)

    criterion_explanations: Dict[str, str] = field(default_factory=dict)

    reasoning: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):

        return {
            "final_score": self.final_score,
            "criterion_scores": self.criterion_scores,
            "criterion_explanations": self.criterion_explanations,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }