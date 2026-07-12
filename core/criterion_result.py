from dataclasses import dataclass, field
from typing import Dict

@dataclass
class CriterionResult:
    """
    Stores the evaluation result of a single criterion.
    """

    criterion_name: str

    score: float

    passed: bool

    reasoning: str

    weight: float = 1.0

    details: Dict = field(default_factory=dict)

    # ---------------------------------------------------------

    def to_dict(self):

        return {

            "criterion_name": self.criterion_name,

            "score": self.score,

            "passed": self.passed,

            "reasoning": self.reasoning,

            "weight": self.weight,

            "details": self.details

        }