import pandas as pd

from core.dataset_artifact import DatasetArtifact
from core.model_selection_result import ModelSelectionResult
from core.training_artifact import TrainingArtifact
from core.training_result import TrainingResult

from .best_model_selector import BestModelSelector
from .cross_validation_runner import CrossValidationRunner
from .data_preprocessor import DataPreprocessor
from .model_factory import ModelFactory
from .model_trainer import ModelTrainer
from .performance_analyzer import PerformanceAnalyzer


class TrainingPipeline:
    """
    Executes the complete training workflow
    for all selected models.
    """

    # -----------------------------------------------------

    def __init__(self):

        self.factory = ModelFactory()

        self.preprocessor = DataPreprocessor()

        self.trainer = ModelTrainer()

        self.cv_runner = CrossValidationRunner()

        self.performance = PerformanceAnalyzer()

        self.best_selector = BestModelSelector()

    # =====================================================
    # Public API
    # =====================================================

    def train_models(

        self,

        model_selection_result: ModelSelectionResult,

        dataset_artifact: DatasetArtifact

    ) -> TrainingArtifact:

        dataframe = dataset_artifact.dataset

        target_column = dataset_artifact.task.target_column

        training_results = []

        for evaluation_result in model_selection_result.selected_models:

            result = self._train_single_model(

                candidate_model=evaluation_result.candidate_model,

                dataframe=dataframe,

                target_column=target_column

            )

            training_results.append(result)

        best_result = self.best_selector.select(

            training_results

        )

        return TrainingArtifact(

            dataset_artifact=dataset_artifact,

            model_selection_result=model_selection_result,

            training_results=training_results,

            best_training_result=best_result,

            training_summary=self._build_summary(

                training_results,

                best_result

            )

        )

    # =====================================================
    # Private Helpers
    # =====================================================

    def _train_single_model(

        self,

        candidate_model,

        dataframe,

        target_column

    ) -> TrainingResult:

        (
            X_train,
            X_test,
            y_train,
            y_test

        ) = self.preprocessor.prepare(

            dataframe,

            target_column

        )

        estimator = self.factory.create(

            candidate_model

        )

        trained_model, training_time = (

            self.trainer.train(

                estimator,

                X_train,

                y_train

            )

        )

        evaluation_metrics = (

            self.performance.analyze(

                trained_model,

                X_test,

                y_test

            )

        )

        cv_metrics = self.cv_runner.run(

            estimator,
            pd.concat([X_train, X_test], axis=0),
            pd.concat([y_train, y_test], axis=0)

        )

        preprocessing_summary = {

            "feature_names": list(

                X_train.columns

            )

        }

        return TrainingResult(
            candidate_model=candidate_model,    

            model_name=candidate_model.model_name,

            trained_model=trained_model,

            evaluation_metrics=evaluation_metrics,

            cross_validation_metrics=cv_metrics,

            training_time_seconds=training_time,

            preprocessing_summary=preprocessing_summary

        )

    # =====================================================

    def _build_summary(

        self,

        training_results,

        best_result

    ):

        if best_result is None:

            return {

                "n_models_trained": 0,

                "best_model": None

            }

        return {

            "n_models_trained":

                len(training_results),

            "best_model":

                best_result.model_name,

            "best_f1":

                best_result.evaluation_metrics.get(

                    "f1",

                    0

                )

        }
