from core.feature_evaluation_config import FeatureEvaluationConfig
from core.feature_evaluation_result import (
    FeatureEvaluationResult
)


class FeatureScorer:
    """
    Combines criterion results into one final decision.
    """

    def __init__(self,config:FeatureEvaluationConfig):

        self.config = config

    # -----------------------------------------------------

    def score_feature(

        self,

        feature_name,

        criterion_results

    ):

        evaluation = FeatureEvaluationResult(

            feature_name=feature_name

        )

        evaluation.criterion_results = criterion_results

        total_weight = sum(

            result.weight

            for result

            in criterion_results

        )

        if total_weight == 0:

            evaluation.overall_score = 0

        else:

            evaluation.overall_score = (

                sum(

                    result.score * result.weight

                    for result

                    in criterion_results

                )

                / total_weight

            )

        evaluation.confidence = evaluation.overall_score

        evaluation.selected = (

            evaluation.overall_score

            >=

            self.config.feature_selection_threshold

        )

        evaluation.reasoning = self.generate_reasoning(

            evaluation

        )

        return evaluation

    # -----------------------------------------------------

    def generate_reasoning(
        self,
        evaluation
    ):

        passed = sum(

            result.passed

            for result

            in evaluation.criterion_results

        )

        total = len(

            evaluation.criterion_results

        )

        return (

            f"{passed}/{total} criteria passed. "

            f"Overall Score = "

            f"{evaluation.overall_score:.3f}"

        )