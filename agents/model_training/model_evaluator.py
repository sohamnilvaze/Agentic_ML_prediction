from typing import Dict, List

from core.candidate_model import CandidateModel
from core.criterion_score import CriterionScore
from core.dataset_artifact import DatasetArtifact
from core.dataset_profile import DatasetProfile
from core.model_evaluation_context import (
    ModelEvaluationContext
)
from core.model_evaluation_result import (
    ModelEvaluationResult
)
from core.model_scoring_policy import (
    ModelScoringPolicy
)


class ModelEvaluator:
    """
    Evaluates the suitability of one candidate model
    for the given dataset.

    Responsibilities
    ----------------
    1. Build evaluation context.
    2. Execute every model evaluation criterion.
    3. Aggregate criterion scores.
    """

    # -----------------------------------------------------

    def __init__(

        self,

        criteria: List

    ):

        self.criteria = criteria

        self.scoring_policy = ModelScoringPolicy()

    # =====================================================
    # Public API
    # =====================================================

    def evaluate_models(

        self,

        candidate_models,

        dataset_profile,

        dataset_artifact

    ):

        results = []

        for model in candidate_models:

            results.append(

                self.evaluate(

                    model,

                    dataset_profile,

                    dataset_artifact

                )

            )

        return results

    # =====================================================
    # Context Builder
    # =====================================================

    def _build_context(

        self,

        candidate_model,

        dataset_profile,

        dataset_artifact

    ) -> ModelEvaluationContext:

        return ModelEvaluationContext(

            candidate_model=candidate_model,

            dataset_profile=dataset_profile,

            dataset_artifact=dataset_artifact

        )

    # =====================================================
    # Criterion Execution
    # =====================================================

    def _run_all_criteria(

        self,

        context: ModelEvaluationContext

    ) -> Dict[str, CriterionScore]:

        scores = {}

        for criterion in self.criteria:

            result = criterion.compute_score(

                context

            )

            scores[

                criterion.__class__.__name__

            ] = result

        return scores