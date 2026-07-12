from dataclasses import dataclass
from enum import Enum


class ExclusionType(Enum):
    """
    Specifies why a feature was excluded.
    """

    NONE = "none"

    HARD = "hard"

    SOFT = "soft"


@dataclass
class CandidateFeature:
    """
    Represents one feature before it is evaluated.
    """

    feature_name: str

    dtype: str

    missing_percentage: float

    unique_count: int

    is_numeric: bool

    is_metadata: bool

    is_constant: bool

    is_high_cardinality: bool

    exclusion_type: ExclusionType = ExclusionType.NONE

    exclusion_reason: str = ""

    candidate: bool = True

    # ------------------------------------------------------

    def to_dict(self):

        return {

            "feature_name": self.feature_name,

            "dtype": self.dtype,

            "missing_percentage": self.missing_percentage,

            "unique_count": self.unique_count,

            "is_numeric": self.is_numeric,

            "is_metadata": self.is_metadata,

            "is_constant": self.is_constant,

            "is_high_cardinality": self.is_high_cardinality,

            "candidate": self.candidate,

            "exclusion_type": self.exclusion_type.value,

            "exclusion_reason": self.exclusion_reason

        }