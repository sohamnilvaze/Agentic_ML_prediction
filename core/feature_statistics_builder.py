import numpy as np
import pandas as pd

from core.feature_statistics import FeatureStatistics


class FeatureStatisticsBuilder:

    """
    Computes FeatureStatistics from a pandas Series.
    """

    def build(

        self,

        feature_name: str,

        series: pd.Series

    ) -> FeatureStatistics:

        total = len(series)

        non_missing = int(series.notna().sum())

        missing = int(series.isna().sum())

        missing_percentage = (

            missing

            / total

            if total

            else 0.0

        )

        clean = series.dropna()

        unique = int(clean.nunique())

        unique_percentage = (

            unique

            / non_missing

            if non_missing

            else 0.0

        )

        singleton_count = 0
        singleton_ratio = 0.0
        rare_category_count = 0
        rare_category_ratio = 0.0
        q1 = None
        q3 = None
        iqr = None
        outlier_count = 0
        outlier_percentage = 0.0

        if non_missing:

            value_counts = clean.value_counts()

            # -----------------------------------------------------
            # Singleton Statistics
            # -----------------------------------------------------

            singleton_count = int(
                (value_counts == 1).sum()
            )

            singleton_ratio = (

                singleton_count
                /
                unique

                if unique > 0

                else 0.0

            )

            # -----------------------------------------------------
            # Rare Category Statistics
            #
            # Rare = frequency < 1% of observations
            # -----------------------------------------------------

            rare_threshold = max(
                1,
                int(0.01 * non_missing)
            )

            rare_category_count = int(
                (value_counts < rare_threshold).sum()
            )

            rare_category_ratio = (

                rare_category_count
                /
                unique

                if unique > 0

                else 0.0

            )

            dominant_value = value_counts.index[0]

            dominant_count = int(value_counts.iloc[0])

            dominant_percentage = (

                dominant_count

                / non_missing

            )

        else:

            value_counts = pd.Series(dtype=int)

            dominant_value = None

            dominant_count = 0

            dominant_percentage = 0.0

        entropy = self.compute_entropy(

            value_counts

        )

        is_numeric = pd.api.types.is_numeric_dtype(

            clean

        )

        if is_numeric and non_missing:

            mean = float(clean.mean())

            median = float(clean.median())

            std = float(clean.std())

            variance = float(clean.var())

            minimum = float(clean.min())

            maximum = float(clean.max())

            # ---------------------------------------------

            q1 = float(
                clean.quantile(0.25)
            )

            q3 = float(
                clean.quantile(0.75)
            )

            iqr = q3 - q1

            # ---------------------------------------------

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            outlier_mask = (

                (clean < lower)

                |

                (clean > upper)

            )

            outlier_count = int(
                outlier_mask.sum()
            )

            outlier_percentage = (

                outlier_count
                /
                non_missing

            )
        else:

            mean = None
            median = None
            std = None
            variance = None
            minimum = None
            maximum = None

        return FeatureStatistics(

            feature_name=feature_name,

            dtype=str(series.dtype),

            is_numeric=is_numeric,

            total_count=total,

            non_missing_count=non_missing,

            missing_count=missing,

            missing_percentage=missing_percentage,

            unique_count=unique,

            unique_percentage=unique_percentage,

            dominant_value=dominant_value,

            dominant_value_count=dominant_count,

            dominant_value_percentage=dominant_percentage,

            entropy=entropy,

            mean=mean,

            median=median,

            std=std,

            variance=variance,

            minimum=minimum,

            maximum=maximum,

            singleton_count=singleton_count,

            singleton_ratio=singleton_ratio,

            rare_category_count=rare_category_count,

            rare_category_ratio=rare_category_ratio,

            q1=q1,

            q3=q3,

            iqr=iqr,

            outlier_count=outlier_count,

            outlier_percentage=outlier_percentage,

        )

    # ----------------------------------------------------

    def compute_entropy(

        self,

        value_counts

    ):

        if len(value_counts) <= 1:

            return 0.0

        probabilities = (

            value_counts

            / value_counts.sum()

        )

        entropy = -np.sum(

            probabilities

            * np.log2(probabilities)

        )

        maximum = np.log2(

            len(probabilities)

        )

        return (

            float(entropy / maximum)

            if maximum > 0

            else 0.0

        )
