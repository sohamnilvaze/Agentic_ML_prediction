from abc import ABC, abstractmethod


class SuitabilityCriterion(ABC):
    """
    Base class for every suitability criterion.
    """

    name = ""

    weight = 0.0

    @abstractmethod
    def evaluate(
        self,
        diagnosis,
        diagnosis_stats,
        total_samples,
        context=None
    ) -> float:
        """
        Returns a score between 0 and 1.
        """
        pass