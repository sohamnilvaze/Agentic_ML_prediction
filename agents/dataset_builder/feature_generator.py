from typing import List

import pandas as pd

from core.candidate_feature import CandidateFeature, ExclusionType
from core.state import PredictionTask


class FeatureGenerator:
    """
    Builds initial candidate feature objects from a dataframe.
    """

    def generate(
        self,
        dataframe: pd.DataFrame,
        task: PredictionTask,
    ) -> List[CandidateFeature]:
        candidate_features = []

        for column in dataframe.columns:
            series = dataframe[column]
            missing_percentage = round(series.isna().mean(), 4)
            unique_count = int(series.nunique(dropna=True))
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_metadata = column.lower() in {
                "subject_id",
                "hadm_id",
                "row_id",
            }
            is_constant = unique_count <= 1
            is_high_cardinality = unique_count > 100

            exclusion_type = ExclusionType.NONE
            exclusion_reason = ""
            candidate = True

            if column == task.target_column:
                candidate = False
                exclusion_type = ExclusionType.HARD
                exclusion_reason = "Target column."
            elif is_metadata:
                candidate = False
                exclusion_type = ExclusionType.HARD
                exclusion_reason = "Metadata column."
            elif is_constant:
                candidate = False
                exclusion_type = ExclusionType.HARD
                exclusion_reason = "Constant column."

            candidate_features.append(
                CandidateFeature(
                    feature_name=column,
                    dtype=str(series.dtype),
                    missing_percentage=missing_percentage,
                    unique_count=unique_count,
                    is_numeric=is_numeric,
                    is_metadata=is_metadata,
                    is_constant=is_constant,
                    is_high_cardinality=is_high_cardinality,
                    exclusion_type=exclusion_type,
                    exclusion_reason=exclusion_reason,
                    candidate=candidate,
                )
            )

        return candidate_features

