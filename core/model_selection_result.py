from dataclasses import dataclass, field
from typing import List

from core.model_evaluation_result import ModelEvaluationResult


@dataclass
class ModelSelectionResult:
    """
    Stores the output of the Model Selection stage.
    """

    selected_models: List[ModelEvaluationResult] = field(
        default_factory=list
    )

    rejected_models: List[ModelEvaluationResult] = field(
        default_factory=list
    )

    total_models: int = 0

    selected_count: int = 0

    rejected_count: int = 0

    best_model: ModelEvaluationResult | None = None

    selection_summary: str = ""

    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "selected_models": [model.to_dict() for model in self.selected_models],
            "rejected_models": [model.to_dict() for model in self.rejected_models],
            "total_models": self.total_models,
            "selected_count": self.selected_count,
            "rejected_count": self.rejected_count,
            "best_model": self.best_model.to_dict() if self.best_model else None,
            "selection_summary": self.selection_summary,
            "metadata": self.metadata,
        }
