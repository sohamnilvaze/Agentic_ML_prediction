import pandas as pd

from core.dataset_validation_report import (
    DatasetValidationReport
)


class DatasetValidator:
    """
    Performs quality validation on the constructed
    training dataset.
    """

    MINIMUM_ROWS = 100

    MINIMUM_CLASS_COUNT = 20

    MAX_DUPLICATE_ROW_RATIO = 0.10

    MAX_FEATURE_MISSING_RATIO = 0.50

    # ---------------------------------------------------------

    def validate_dataset(

        self,

        dataset: pd.DataFrame,

        target_column: str

    ) -> DatasetValidationReport:

        errors = []

        warnings = []

        statistics = {}

        # --------------------------------------------

        self._validate_row_count(

            dataset,

            errors,

            statistics

        )

        self._validate_target(

            dataset,

            target_column,

            errors,

            statistics

        )

        self._validate_class_balance(

            dataset,

            target_column,

            warnings,

            statistics

        )

        self._validate_duplicate_rows(

            dataset,

            warnings,

            statistics

        )

        self._validate_duplicate_columns(

            dataset,

            errors,

            statistics

        )

        self._validate_missing_features(

            dataset,

            target_column,

            warnings,

            statistics

        )

        summary = self._build_summary(

            errors,

            warnings,

            statistics

        )

        return DatasetValidationReport(

            passed=(len(errors) == 0),

            errors=errors,

            warnings=warnings,

            statistics=statistics,

            summary=summary

        )

    # ==========================================================
    # Validation Helpers
    # ==========================================================

    def _validate_row_count(

        self,

        dataset,

        errors,

        statistics

    ):

        row_count = len(dataset)

        statistics["row_count"] = row_count

        if row_count < self.MINIMUM_ROWS:

            errors.append(

                f"Dataset contains only {row_count} rows."

            )

    # ----------------------------------------------------------

    def _validate_target(

        self,

        dataset,

        target_column,

        errors,

        statistics

    ):

        if target_column not in dataset.columns:

            errors.append(

                f"Target column '{target_column}' not found."

            )

            return

        missing = dataset[target_column].isna().sum()

        statistics["missing_target"] = int(missing)

        if missing > 0:

            errors.append(

                f"Target column contains {missing} missing values."

            )

        unique_classes = dataset[target_column].nunique()

        statistics["target_classes"] = int(unique_classes)

        if unique_classes < 2:

            errors.append(

                "Target column contains fewer than two classes."

            )

    # ----------------------------------------------------------

    def _validate_class_balance(

        self,

        dataset,

        target_column,

        warnings,

        statistics

    ):

        if target_column not in dataset.columns:

            return

        class_counts = dataset[target_column].value_counts()

        statistics["class_distribution"] = (

            class_counts.to_dict()

        )

        minimum = class_counts.min()

        statistics["minority_class_count"] = int(minimum)

        if minimum < self.MINIMUM_CLASS_COUNT:

            warnings.append(

                "Minority class contains very few samples."

            )

    # ----------------------------------------------------------

    def _validate_duplicate_rows(

        self,

        dataset,

        warnings,

        statistics

    ):

        duplicates = dataset.duplicated().sum()

        ratio = duplicates / max(len(dataset), 1)

        statistics["duplicate_rows"] = int(duplicates)

        statistics["duplicate_row_ratio"] = round(ratio, 3)

        if ratio > self.MAX_DUPLICATE_ROW_RATIO:

            warnings.append(

                f"{duplicates} duplicate rows detected."

            )

    # ----------------------------------------------------------

    def _validate_duplicate_columns(

        self,

        dataset,

        errors,

        statistics

    ):

        duplicate_columns = dataset.columns[

            dataset.columns.duplicated()

        ].tolist()

        statistics["duplicate_columns"] = duplicate_columns

        if duplicate_columns:

            errors.append(

                f"Duplicate columns detected: {duplicate_columns}"

            )

    # ----------------------------------------------------------

    def _validate_missing_features(

        self,

        dataset,

        target_column,

        warnings,

        statistics

    ):

        missing_ratios = {}

        for column in dataset.columns:

            if column == target_column:

                continue

            ratio = dataset[column].isna().mean()

            missing_ratios[column] = round(ratio, 3)

            if ratio > self.MAX_FEATURE_MISSING_RATIO:

                warnings.append(

                    f"Feature '{column}' has "

                    f"{ratio:.1%} missing values."

                )

        statistics["feature_missing_ratio"] = (

            missing_ratios

        )

    # ----------------------------------------------------------

    def _build_summary(

        self,

        errors,

        warnings,

        statistics

    ):

        return (

            f"Rows={statistics.get('row_count',0)}, "

            f"Errors={len(errors)}, "

            f"Warnings={len(warnings)}"

        )