from core.criterion_score import CriterionScore

from .base_model_criterion import BaseModelCriterion


class DatasetSizeCriterion(BaseModelCriterion):

    """
    Evaluates whether a model is appropriate
    for the dataset size.
    """

    def compute_score(self, context):

        n = context.dataset_profile.n_samples

        model = context.candidate_model.algorithm

        score = 1.0

        reasoning = ""

        if model in ["knn", "svm"]:

            if n > 10000:

                score = 0.40
                reasoning = "Large datasets reduce efficiency."

            elif n > 5000:

                score = 0.70
                reasoning = "Moderately suitable."

            else:

                score = 1.0
                reasoning = "Very suitable."

        elif model in [

            "random_forest",

            "gradient_boosting"

        ]:

            if n < 500:

                score = 0.70
                reasoning = "Limited training samples."

            else:

                score = 1.0
                reasoning = "Well suited."

        else:

            score = 1.0
            reasoning = "Dataset size acceptable."

        return CriterionScore(

            score=score,

            passed=(score >= 0.60),

            reasoning=reasoning

        )