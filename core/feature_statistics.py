from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class FeatureStatistics:
    """
    Stores statistics describing a single feature.

    These statistics are computed exactly once and shared
    across all Feature Evaluation Criteria.
    """

    feature_name: str

    dtype: str

    is_numeric: bool

    total_count: int

    non_missing_count: int

    missing_count: int

    missing_percentage: float

    unique_count: int

    unique_percentage: float

    dominant_value: Any

    dominant_value_count: int

    dominant_value_percentage: float

    entropy: float

    mean: float | None = None

    median: float | None = None

    std: float | None = None

    variance: float | None = None

    minimum: float | None = None

    maximum: float | None = None

    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Distribution Statistics (Categorical)
    # ------------------------------------------------------------------

    singleton_count: int = 0

    singleton_ratio: float = 0.0

    rare_category_count: int = 0

    rare_category_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Distribution Statistics (Numeric)
    # ------------------------------------------------------------------

    q1: float | None = None

    q3: float | None = None

    iqr: float | None = None

    outlier_count: int = 0

    outlier_percentage: float = 0.0