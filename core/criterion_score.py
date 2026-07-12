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

    def evaluate(
        self,
        context
    ):

        score = self.compute_score(context)

        return self.build_result(

            score.score,
            score.passed,
            score.reasoning,
            score.details

        )

