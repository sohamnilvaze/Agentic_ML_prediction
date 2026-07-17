from typing import List

from core.candidate_model import CandidateModel


class ModelCatalog:
    """
    Repository of all supported machine learning algorithms.

    This class is intentionally static.
    It simply describes the algorithms that the framework
    knows about.

    No dataset-specific logic belongs here.
    """

    def __init__(self):

        self.models = self._build_catalog()

    # -----------------------------------------------------

    def get_all_models(self) -> List[CandidateModel]:

        return list(self.models)

    # -----------------------------------------------------

    def _build_catalog(self):

        return [

            CandidateModel(

                model_name="Logistic Regression",

                algorithm="logistic_regression",

                task_type="classification",

                interpretable=True,

                supports_categorical=False,

                supports_missing_values=False,

                requires_feature_scaling=True,

                supports_multiclass=True,

                produces_probabilities=True,

                training_complexity="low",

                default_parameters={

                    "max_iter": 1000

                }

            ),

            CandidateModel(

                model_name="Decision Tree",

                algorithm="decision_tree",

                task_type="classification",

                interpretable=True,

                supports_categorical=False,

                supports_missing_values=False,

                requires_feature_scaling=False,

                supports_multiclass=True,

                produces_probabilities=True,

                training_complexity="low",

                default_parameters={}

            ),

            CandidateModel(

                model_name="Random Forest",

                algorithm="random_forest",

                task_type="classification",

                interpretable=False,

                supports_categorical=False,

                supports_missing_values=False,

                requires_feature_scaling=False,

                supports_multiclass=True,

                produces_probabilities=True,

                training_complexity="medium",

                default_parameters={

                    "n_estimators": 200

                }

            ),

            CandidateModel(

                model_name="Gradient Boosting",

                algorithm="gradient_boosting",

                task_type="classification",

                interpretable=False,

                supports_categorical=False,

                supports_missing_values=False,

                requires_feature_scaling=False,

                supports_multiclass=True,

                produces_probabilities=True,

                training_complexity="high",

                default_parameters={}

            ),

            CandidateModel(

                model_name="Support Vector Machine",

                algorithm="svm",

                task_type="classification",

                interpretable=False,

                supports_categorical=False,

                supports_missing_values=False,

                requires_feature_scaling=True,

                supports_multiclass=True,

                produces_probabilities=True,

                training_complexity="high",

                default_parameters={

                    "probability": True

                }

            ),

            CandidateModel(

                model_name="K Nearest Neighbors",

                algorithm="knn",

                task_type="classification",

                interpretable=False,

                supports_categorical=False,

                supports_missing_values=False,

                requires_feature_scaling=True,

                supports_multiclass=True,

                produces_probabilities=True,

                training_complexity="medium",

                default_parameters={}

            ),

            CandidateModel(

                model_name="Naive Bayes",

                algorithm="naive_bayes",

                task_type="classification",

                interpretable=True,

                supports_categorical=False,

                supports_missing_values=False,

                requires_feature_scaling=False,

                supports_multiclass=True,

                produces_probabilities=True,

                training_complexity="low",

                default_parameters={}

            )

        ]
