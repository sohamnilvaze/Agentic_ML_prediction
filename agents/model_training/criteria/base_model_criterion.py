from abc import ABC, abstractmethod

from core.criterion_score import CriterionScore
from core.model_evaluation_context import ModelEvaluationContext


class BaseModelCriterion(ABC):
    """
    Base class for all Model Evaluation Criteria.

    Each criterion evaluates one aspect of the suitability
    of a candidate model for a dataset.
    """

    @abstractmethod
    def compute_score(
        self,
        context: ModelEvaluationContext
    ) -> CriterionScore:
        """
        Computes the suitability score for this criterion.
        """
        pass