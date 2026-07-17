from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CriterionScore:
    """
    Represents the raw output produced by one evaluation criterion.

    Notes
    -----
    score
        Normalized quality score in [0,1].

    passed
        Whether this feature satisfies this criterion's policy.

    reasoning
        Human-readable explanation.

    details
        Machine-readable statistics used by downstream agents.
    """

    score: float

    passed: bool

    reasoning: str

    details: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "details": self.details,
        }

