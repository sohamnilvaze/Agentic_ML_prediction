import pandas as pd

from core.candidate_task import CandidateTask
from core.feature_selection_result import FeatureSelectionResult


class DatasetConstructor:
    """
    Builds the final training dataset from the
    master dataset and selected features.
    """

    # -----------------------------------------------------

    def construct_dataset(

        self,

        master_dataframe: pd.DataFrame,

        task: CandidateTask,

        feature_selection: FeatureSelectionResult

    ) -> pd.DataFrame:

        selected_columns = list(

            feature_selection.selected_feature_names

        )

        target_column = task.target_column

        # -------------------------------------------------
        # Ensure target is included
        # -------------------------------------------------

        if target_column not in selected_columns:

            selected_columns.append(
                target_column
            )

        # -------------------------------------------------
        # Verify columns exist
        # -------------------------------------------------

        missing_columns = [

            column

            for column in selected_columns

            if column not in master_dataframe.columns

        ]

        if missing_columns:

            raise ValueError(

                f"Missing columns: {missing_columns}"

            )

        # -------------------------------------------------
        # Construct dataset
        # -------------------------------------------------

        dataset = master_dataframe[

            selected_columns

        ].copy()

        # -------------------------------------------------
        # Remove rows with missing target
        # -------------------------------------------------

        dataset = dataset.dropna(

            subset=[target_column]

        )

        # -------------------------------------------------
        # Remove duplicate rows
        # -------------------------------------------------

        dataset = dataset.drop_duplicates()

        # -------------------------------------------------
        # Reset index
        # -------------------------------------------------

        dataset = dataset.reset_index(

            drop=True

        )

        return dataset