import pandas as pd

from core.dataset_profile import DatasetProfile


class DatasetProfiler:

    """
    Builds a reusable profile of a dataset.
    """

    # ---------------------------------------------------------

    def build_profile(
        self,
        df: pd.DataFrame,
        target_column: str = None
    ) -> DatasetProfile:

        profile = DatasetProfile()

        profile.n_samples = len(df)

        profile.n_features = len(df.columns)

        profile.feature_names = list(df.columns)

        profile.memory_usage_mb = (
            df.memory_usage(deep=True).sum()
            / (1024 ** 2)
        )

        self._identify_feature_types(
            df,
            profile
        )

        self._analyze_missing_values(
            df,
            profile
        )

        self._analyze_cardinality(
            df,
            profile
        )

        self._detect_constant_columns(
            profile
        )

        self._detect_duplicate_rows(
            df,
            profile
        )

        self._identify_metadata_columns(
            profile
        )

        self._generate_warnings(
            profile
        )

        self._generate_recommendations(
            profile
        )

        if target_column is not None:

            self._analyze_target(
                df,
                target_column,
                profile
            )

        return profile

    # ---------------------------------------------------------

    def _identify_feature_types(
        self,
        df,
        profile
    ):

        for column in df.columns:

            dtype = str(df[column].dtype)

            profile.feature_types[column] = dtype

            if pd.api.types.is_numeric_dtype(df[column]):

                profile.numeric_columns.append(column)

            else:

                profile.categorical_columns.append(column)

    # ---------------------------------------------------------

    def _analyze_missing_values(
        self,
        df,
        profile
    ):

        for column in df.columns:

            missing = int(df[column].isna().sum())

            percent = (
                missing / len(df)
            ) * 100

            profile.missing_counts[column] = missing

            profile.missing_percentages[column] = round(
                percent,
                2
            )

    # ---------------------------------------------------------

    def _analyze_cardinality(
        self,
        df,
        profile
    ):

        for column in df.columns:

            unique = int(df[column].nunique())

            profile.unique_counts[column] = unique

            if unique == 1:

                profile.constant_columns.append(
                    column
                )

            elif unique > 100:

                profile.high_cardinality_columns.append(
                    column
                )

    # ---------------------------------------------------------

    def _detect_constant_columns(
        self,
        profile
    ):
        """
        Placeholder.
        Constant columns already identified
        during cardinality analysis.
        """
        pass

    # ---------------------------------------------------------

    def _detect_duplicate_rows(
        self,
        df,
        profile
    ):

        profile.duplicate_rows = int(
            df.duplicated().sum()
        )

    # ---------------------------------------------------------

    def _identify_metadata_columns(
        self,
        profile
    ):

        keywords = [

            "subject_id",

            "hadm_id",

            "row_id"

        ]

        for column in profile.feature_names:

            lower = column.lower()

            for keyword in keywords:

                if keyword in lower:

                    profile.metadata_columns.append(
                        column
                    )

                    break

    # ---------------------------------------------------------

    def _generate_warnings(
        self,
        profile
    ):

        for column, pct in profile.missing_percentages.items():

            if pct > 40:

                profile.warnings.append(

                    f"{column} has {pct:.1f}% missing values."

                )

        if profile.duplicate_rows > 0:

            profile.warnings.append(

                f"{profile.duplicate_rows} duplicate rows detected."

            )

    # ---------------------------------------------------------

    def _generate_recommendations(
        self,
        profile
    ):

        if profile.metadata_columns:

            profile.recommendations.append(

                "Metadata columns should not be used as predictive features."

            )

        if profile.high_cardinality_columns:

            profile.recommendations.append(

                "Review high-cardinality categorical columns."

            )

    # ---------------------------------------------------------

    def _analyze_target(
        self,
        df,
        target_column,
        profile
    ):

        if target_column not in df.columns:

            return

        distribution = (

            df[target_column]

            .value_counts()

            .to_dict()

        )

        profile.target_distribution = distribution