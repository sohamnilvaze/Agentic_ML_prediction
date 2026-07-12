from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.candidate_task import CandidateTask
from core.dataset_report import DatasetReport
from core.dataset_validation_report import DatasetValidationReport
from core.feature_selection_result import FeatureSelectionResult


@dataclass
class DatasetArtifact:
    """
    Final output produced by the Dataset Builder Agent.

    This artifact contains everything required by downstream
    agents to understand and use the constructed dataset.
    """

    task: CandidateTask

    dataset: pd.DataFrame

    feature_selection: FeatureSelectionResult

    validation_report: DatasetValidationReport

    report: DatasetReport

    metadata: Optional[dict] = None