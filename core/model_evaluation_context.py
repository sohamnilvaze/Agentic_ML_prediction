from dataclasses import dataclass

from core.dataset_profile import DatasetProfile
from core.candidate_model import CandidateModel
from core.dataset_artifact import DatasetArtifact


@dataclass
class ModelEvaluationContext:
    """
    Context supplied to every Model Evaluation Criterion.
    """

    candidate_model: CandidateModel

    dataset_artifact: DatasetArtifact

    dataset_profile: DatasetProfile