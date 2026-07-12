from abc import ABC, abstractmethod

from core.criterion_result import CriterionResult
from core.feature_evaluation_context import FeatureEvaluationContext
from core.feature_evaluation_config import FeatureEvaluationConfig




class FeatureCriterion(ABC):

    """
    ================================================================================
    Feature Evaluation Contract
    ================================================================================

    Every FeatureCriterion evaluates exactly ONE aspect of a feature's quality.

    Responsibilities
    ----------------
    A criterion should ONLY evaluate its own responsibility.

    Examples:
        MissingValueCriterion
            -> evaluates missing values only

        CardinalityCriterion
            -> evaluates cardinality only

        LeakageCriterion
            -> evaluates leakage only

    A criterion MUST NOT consider the outputs of any other criterion.

    ================================================================================
    Scoring Contract
    ================================================================================

    Every criterion must return a CriterionScore satisfying the following rules.

    1. Score Range

        score ∈ [0.0, 1.0]

        1.0
            Excellent feature quality

        0.75
            Good quality

        0.50
            Acceptable but not ideal

        0.25
            Poor quality

        0.0
            Unusable feature

    No criterion should ever return values outside [0,1].

    ================================================================================
    Pass / Fail Contract
    ================================================================================

    passed is NOT determined by the score.

    Instead,

    passed indicates whether the feature satisfies this criterion's policy.

    Example

    Missing threshold = 30%

    Missing = 28%

    score = 0.72

    passed = True

    Missing = 32%

    score = 0.68

    passed = False

    Therefore

    score represents feature quality.

    passed represents policy compliance.

    ================================================================================
    Reasoning Contract
    ================================================================================

    reasoning should explain

    WHY

    the score was assigned.

    Example

    Good:

        "Missing percentage is 12.4%, which is below the configured threshold."

    Bad:

        "Feature is good."

    ================================================================================
    Details Contract
    ================================================================================

    details should contain structured metadata that other agents can inspect.

    Example

    details = {

        "missing_percentage":12.4,

        "threshold":30

    }

    Do NOT duplicate reasoning here.

    details should contain machine-readable values.

    ================================================================================
    Statelessness
    ================================================================================

    Criteria MUST be stateless.

    They should never modify

        dataframe

        candidate feature

        workflow state

    Their only responsibility is evaluation.

    ================================================================================
    Normalization
    ================================================================================

    Higher score always means better feature quality.

    All criteria should normalize their score into [0,1].

    ================================================================================
    Determinism
    ================================================================================

    Given identical inputs,

    compute_score()

    must always return identical outputs.

    No randomness should be used.

    ================================================================================
    """

    def __init__(self,
                 config,
                 weight=1.0):

        self.config = config
        self.weight = weight

    @property
    def name(self):

        return self.__class__.__name__

    @property
    def description(self):

        return "No description."

    def evaluate(
        self,
        context
    ):

        score, passed, reasoning, details = \
            self.compute_score(context)

        return self.build_result(

            score,
            passed,
            reasoning,
            details

        )

    @abstractmethod
    def compute_score(
        self,
        context
    ):
        """
        Returns

        score,
        passed,
        reasoning,
        details
        """
        pass

    def build_result(

        self,

        score,

        passed,

        reasoning,

        details

    ):

        return CriterionResult(

            criterion_name=self.name,

            score=score,

            passed=passed,

            reasoning=reasoning,

            weight=self.weight,

            details=details

        )