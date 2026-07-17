from typing import TYPE_CHECKING

import pandas as pd

from core.feature_selection_result import FeatureSelectionResult

if TYPE_CHECKING:
    from core.state import PredictionTask


class DatasetConstructor:
    """
    Builds the final training dataset from the
    master dataset and selected features.
    """

    # -----------------------------------------------------

    def construct_dataset(

        self,

        master_dataframe: pd.DataFrame,

        task: "PredictionTask",

        feature_selection: FeatureSelectionResult

    ) -> pd.DataFrame:

        selected_columns = list(

            feature_selection.selected_feature_names

        )

        source_target_column = task.target_column
        binary_target_column = "target"

        missing_columns = [
            column
            for column in selected_columns
            if column not in master_dataframe.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        # -------------------------------------------------
        # Build a binary target for the selected diagnosis
        # -------------------------------------------------

        if source_target_column not in master_dataframe.columns:

            raise ValueError(
                f"Source target column '{source_target_column}' not found."
            )

        dataset = master_dataframe[
            selected_columns + [source_target_column]
        ].dropna(
            subset=[source_target_column]
        ).copy()

        # -------------------------------------------------
        # Convert the selected diagnosis into a binary label
        # -------------------------------------------------

        dataset[binary_target_column] = (
            dataset[source_target_column]
            == task.target_value
        ).astype(int)

        dataset = dataset.drop(
            columns=[source_target_column]
        )

        task.target_column = binary_target_column

        # -------------------------------------------------
        # Construct dataset
        # -------------------------------------------------

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
