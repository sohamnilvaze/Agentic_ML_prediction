from agents.base_agent import BaseAgent
from core.dataset_artifact import DatasetArtifact
from core.state import WorkflowState

from .dataset_report_builder import DatasetReportBuilder
from .dataset_validator import DatasetValidator
from .feature_evaluator import FeatureEvaluator
from .feature_generator import FeatureGenerator


class DatasetBuilderAgent(BaseAgent):
    """
    Builds a machine-learning-ready dataset for the selected task.
    """

    def __init__(
        self,
        feature_generator=None,
        feature_evaluator=None,
        feature_selector=None,
        dataset_constructor=None,
        dataset_validator=None,
        report_builder=None,
    ):
        self.feature_generator = feature_generator or FeatureGenerator()
        self.feature_evaluator = feature_evaluator
        self.feature_selector = feature_selector
        self.dataset_constructor = dataset_constructor
        self.dataset_validator = dataset_validator or DatasetValidator()
        self.report_builder = report_builder or DatasetReportBuilder()

    def run(self, state: WorkflowState) -> WorkflowState:
        if state.raw_dataset is None:
            raise ValueError("WorkflowState.raw_dataset is required.")

        if state.selected_prediction_task is None:
            raise ValueError("WorkflowState.selected_prediction_task is required.")

        master_dataframe = state.raw_dataset
        task = state.selected_prediction_task

        candidate_features = self.feature_generator.generate(
            master_dataframe,
            task,
        )

        candidate_features = [
            feature
            for feature in candidate_features
            if feature.candidate
        ]

        feature_results = []
        for feature in candidate_features:
            result = self.feature_evaluator.evaluate(
                feature,
                master_dataframe,
                task.target_column,
            )
            feature_results.append(result)

        feature_selection = self.feature_selector.select_features(
            feature_results
        )

        dataset = self.dataset_constructor.construct_dataset(
            master_dataframe,
            task,
            feature_selection,
        )

        validation_report = self.dataset_validator.validate_dataset(
            dataset,
            task.target_column,
        )

        report = self.report_builder.build_report(
            task,
            dataset,
            feature_selection,
            validation_report,
        )

        artifact = DatasetArtifact(
            task=task,
            dataset=dataset,
            feature_selection=feature_selection,
            validation_report=validation_report,
            report=report,
            metadata={"builder": "DatasetBuilderAgent"},
        )

        state.dataset_artifact = artifact
        state.selected_features = list(feature_selection.selected_feature_names)
        state.current_stage = "DATASET_BUILDER_COMPLETED"
        state.add_history(
            f"Dataset built with {len(state.selected_features)} selected features."
        )
        return state
