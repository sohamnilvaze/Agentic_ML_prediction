from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DatasetValidationReport:
    """
    Represents the outcome of dataset validation.
    """

    passed: bool

    errors: List[str]

    warnings: List[str]

    statistics: Dict

    summary: str

    metadata: Dict = field(default_factory=dict)