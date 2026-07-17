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

    def to_dict(self):
        return {
            "task_name": self.task_name,
            "target_column": self.target_column,
            "dataset_statistics": self.dataset_statistics,
            "selected_features": self.selected_features,
            "rejected_features": self.rejected_features,
            "validation_summary": self.validation_summary,
            "feature_selection_summary": self.feature_selection_summary,
            "metadata": self.metadata,
        }
