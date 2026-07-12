import pandas as pd

from core.dataset_artifact import DatasetArtifact
from core.dataset_profile import DatasetProfile


class DatasetProfiler:
    """
    Generates a comprehensive profile describing
    the constructed training dataset.

    The profile is later consumed by the
    Model Selection Agent.
    """

    # ---------------------------------------------------------

    def profile_dataset(

        self,

        dataset_artifact: DatasetArtifact

    ) -> DatasetProfile:

        dataframe = dataset_artifact.dataset

        target_column = dataset_artifact.task.target_column

        profile = DatasetProfile()

        profile.target_column = target_column

        # -------------------------------------------------

        self._compute_basic_statistics(

            dataframe,

            profile

        )

        self._compute_column_statistics(

            dataframe,

            profile,

            target_column

        )

        self._compute_missing_statistics(

            dataframe,

            profile

        )

        self._compute_cardinality_statistics(

            dataframe,

            profile,

            target_column

        )

        self._compute_target_statistics(

            dataframe,

            profile,

            target_column

        )

        self._compute_memory_statistics(

            dataframe,

            profile

        )

        return profile

    # =========================================================
    # Basic Statistics
    # =========================================================

    def _compute_basic_statistics(

        self,

        dataframe,

        profile

    ):

        profile.n_samples = len(dataframe)

        profile.n_features = len(dataframe.columns) - 1

        profile.feature_names = list(dataframe.columns)

        profile.duplicate_rows = int(

            dataframe.duplicated().sum()

        )

    # =========================================================
    # Column Statistics
    # =========================================================

    def _compute_column_statistics(

        self,

        dataframe,

        profile,

        target_column

    ):

        profile.feature_types = {}

        profile.numeric_columns = []

        profile.categorical_columns = []

        profile.metadata_columns = []

        for column in dataframe.columns:

            if column == target_column:

                continue

            dtype = str(

                dataframe[column].dtype

            )

            profile.feature_types[column] = dtype

            if pd.api.types.is_numeric_dtype(

                dataframe[column]

            ):

                profile.numeric_columns.append(

                    column

                )

            else:

                profile.categorical_columns.append(

                    column

                )

    # =========================================================
    # Missing Value Statistics
    # =========================================================

    def _compute_missing_statistics(

        self,

        dataframe,

        profile

    ):

        total_missing = 0

        total_cells = dataframe.shape[0] * dataframe.shape[1]

        for column in dataframe.columns:

            missing = int(

                dataframe[column].isna().sum()

            )

            percentage = (

                missing / len(dataframe)

            )

            profile.missing_counts[

                column

            ] = missing

            profile.missing_percentages[

                column

            ] = round(

                percentage,

                4

            )

            total_missing += missing

        profile.overall_missing_ratio = round(

            total_missing / total_cells,

            4

        )

    # =========================================================
    # Cardinality Statistics
    # =========================================================

    def _compute_cardinality_statistics(

        self,

        dataframe,

        profile,

        target_column

    ):

        HIGH_CARDINALITY_THRESHOLD = 100

        for column in dataframe.columns:

            if column == target_column:

                continue

            unique = int(

                dataframe[column].nunique(

                    dropna=True

                )

            )

            profile.unique_counts[

                column

            ] = unique

            if unique == 1:

                profile.constant_columns.append(

                    column

                )

            if unique > HIGH_CARDINALITY_THRESHOLD:

                profile.high_cardinality_columns.append(

                    column

                )

    # =========================================================
    # Target Statistics
    # =========================================================

    def _compute_target_statistics(

        self,

        dataframe,

        profile,

        target_column

    ):

        distribution = dataframe[

            target_column

        ].value_counts()

        profile.target_distribution = (

            distribution.to_dict()

        )

        if len(distribution) == 2:

            counts = distribution.tolist()

            majority = max(counts)

            minority = min(counts)

            profile.positive_samples = majority

            profile.negative_samples = minority

            if minority > 0:

                profile.imbalance_ratio = round(

                    majority / minority,

                    3

                )

        # profile.task_type = "classification"

        unique_classes = dataframe[target_column].nunique()

        if dataframe[target_column].dtype == object:

            profile.task_type = "classification"

        elif unique_classes <= 20:

            profile.task_type = "classification"

        else:

            profile.task_type = "regression"
    # =========================================================
    # Memory Statistics
    # =========================================================

    def _compute_memory_statistics(

        self,

        dataframe,

        profile

    ):

        profile.memory_usage_mb = round(

            dataframe.memory_usage(

                deep=True

            ).sum()

            / (1024 * 1024),

            2

        )