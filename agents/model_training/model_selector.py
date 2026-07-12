from typing import List

from core.model_evaluation_result import ModelEvaluationResult
from core.model_selection_result import ModelSelectionResult


class ModelSelector:
    """
    Selects the best candidate models
    for the training phase.
    """

    DEFAULT_SCORE_THRESHOLD = 0.70

    DEFAULT_MAX_MODELS = 3

    # -----------------------------------------------------

    def __init__(

        self,

        score_threshold=None,

        max_models=None

    ):

        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else self.DEFAULT_SCORE_THRESHOLD
        )

        self.max_models = (
            max_models
            if max_models is not None
            else self.DEFAULT_MAX_MODELS
        )

    # -----------------------------------------------------

    def select_models(

        self,

        evaluation_results: List[ModelEvaluationResult]

    ) -> ModelSelectionResult:

        sorted_results = self._sort_models(
            evaluation_results
        )

        selected = []
        rejected = []

        for result in sorted_results:

            if (
                result.passed
                and
                result.overall_score >= self.score_threshold
            ):
                selected.append(result)
            else:
                rejected.append(result)

        selected = selected[: self.max_models]

        result = ModelSelectionResult()

        result.selected_models = selected
        result.rejected_models = rejected

        result.total_models = len(evaluation_results)
        result.selected_count = len(selected)
        result.rejected_count = len(rejected)

        if selected:
            result.best_model = selected[0]

        result.selection_summary = (
            f"{result.selected_count} of "
            f"{result.total_models} models selected."
        )

        return result

    # -----------------------------------------------------

    def _sort_models(

        self,

        results

    ):

        return sorted(

            results,

            key=lambda x: x.overall_score,

            reverse=True

        )