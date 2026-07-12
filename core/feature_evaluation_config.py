from dataclasses import dataclass


@dataclass
class FeatureEvaluationConfig:
    """
    Configuration shared by all feature evaluation criteria.
    """

    # -------------------------------
    # Missing Value Criterion
    # -------------------------------

    missing_threshold: float = 30.0

    # -------------------------------
    # Cardinality Criterion
    # -------------------------------

    high_cardinality_threshold: int = 100

    # -------------------------------
    # Variability Criterion
    # -------------------------------

    minimum_unique_values: int = 2

    dominant_value_threshold: float = 0.95

    # -------------------------------
    # Data Quality Criterion
    # -------------------------------

    supported_dtypes = (
        "int64",
        "float64",
        "object",
        "bool",
        "category"
    )

    # -------------------------------
    # Scoring Threshold
    # -------------------------------

    feature_selection_threshold: float = 0.65