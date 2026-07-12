from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CandidateTask:
    """
    Represents a disease prediction task discovered
    by the TaskDiscoveryAgent.
    """

    # -------------------------------------------------
    # Task Identification
    # -------------------------------------------------

    task_id: str

    disease_name: str

    target_column: str

    positive_label: str

    negative_label: str

    # -------------------------------------------------
    # Dataset Statistics
    # -------------------------------------------------

    positive_samples: int

    negative_samples: int

    total_samples: int

    prevalence: float

    # -------------------------------------------------
    # Task Quality
    # -------------------------------------------------

    suitability_score: float

    ranking_score: float

    reasoning: str

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    metadata: Dict = field(default_factory=dict)