from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DatasetReport:
    """
    Human-readable and machine-readable summary
    of the Dataset Builder output.
    """

    task_name: str

    target_column: str

    dataset_statistics: Dict

    selected_features: List[str]

    rejected_features: List[str]

    validation_summary: str

    feature_selection_summary: str

    metadata: Dict = field(default_factory=dict)