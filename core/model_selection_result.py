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