from agents.base_agent import BaseAgent
from core.dataset_artifact import DatasetArtifact


class DatasetBuilderAgent(BaseAgent):

    def __init__(

        self,

        feature_generator,

        feature_evaluator,

        feature_selector,

        dataset_constructor,

        dataset_validator,

        report_builder

    ):

        self.feature_generator = feature_generator

        self.feature_evaluator = feature_evaluator

        self.feature_selector = feature_selector

        self.dataset_constructor = dataset_constructor

        self.dataset_validator = dataset_validator

        self.report_builder = report_builder

    # ---------------------------------------------------------

    def run(

        self,

        task,

        master_dataframe

    ) -> DatasetArtifact:

        # -------------------------------------------------
        # Generate candidate features
        # -------------------------------------------------

        candidate_features = self.feature_generator.generate(

            master_dataframe,

            task

        )

        # -------------------------------------------------
        # Evaluate every feature
        # -------------------------------------------------

        feature_results = []

        for feature in candidate_features:

            result = self.feature_evaluator.evaluate(

                feature,

                master_dataframe,

                task.target_column

            )

            feature_results.append(

                result

            )

        # -------------------------------------------------
        # Select best features
        # -------------------------------------------------

        feature_selection = (

            self.feature_selector.select_features(

                feature_results

            )

        )

        # -------------------------------------------------
        # Construct dataset
        # -------------------------------------------------

        dataset = (

            self.dataset_constructor.construct_dataset(

                master_dataframe,

                task,

                feature_selection

            )

        )

        # -------------------------------------------------
        # Validate dataset
        # -------------------------------------------------

        validation_report = (

            self.dataset_validator.validate_dataset(

                dataset,

                task.target_column

            )

        )

        # -------------------------------------------------
        # Build report
        # -------------------------------------------------

        report = (

            self.report_builder.build_report(

                task,

                dataset,

                feature_selection,

                validation_report

            )

        )

        # -------------------------------------------------
        # Package artifact
        # -------------------------------------------------

        return DatasetArtifact(

            task=task,

            dataset=dataset,

            feature_selection=feature_selection,

            validation_report=validation_report,

            report=report,

            metadata={

                "builder": "DatasetBuilderAgent"

            }

        )