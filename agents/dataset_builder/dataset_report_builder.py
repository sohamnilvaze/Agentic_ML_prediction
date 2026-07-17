from typing import TYPE_CHECKING

import pandas as pd

from core.dataset_report import DatasetReport
from core.dataset_validation_report import DatasetValidationReport
from core.feature_selection_result import FeatureSelectionResult

if TYPE_CHECKING:
    from core.state import PredictionTask


class DatasetReportBuilder:
    """
    Builds the final report describing the
    constructed training dataset.
    """

    # ---------------------------------------------------------

    def build_report(

        self,

        task: "PredictionTask",

        training_dataframe: pd.DataFrame,

        feature_selection: FeatureSelectionResult,

        validation_report: DatasetValidationReport

    ) -> DatasetReport:

        statistics = {

            "rows": len(training_dataframe),

            "columns": len(training_dataframe.columns),

            "feature_count": len(

                feature_selection.selected_feature_names

            ),

            "target_distribution": (

                training_dataframe[
                    task.target_column
                ]

                .value_counts()

                .to_dict()

            )

        }

        return DatasetReport(

            task_name=task.disease_name,

            target_column=task.target_column,

            dataset_statistics=statistics,

            selected_features=(

                feature_selection

                .selected_feature_names

            ),

            rejected_features=(

                feature_selection

                .rejected_feature_names

            ),

            validation_summary=(

                validation_report.summary

            ),

            feature_selection_summary=(

                feature_selection.overall_summary

            )

        )
