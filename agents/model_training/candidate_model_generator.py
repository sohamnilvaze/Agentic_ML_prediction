from typing import List

from core.candidate_model import CandidateModel
from core.dataset_profile import DatasetProfile

from .model_catalog import ModelCatalog


class CandidateModelGenerator:
    """
    Generates all candidate models applicable to the
    current prediction task.

    No ranking or suitability analysis is performed here.
    """

    def __init__(

        self,

        catalog=None

    ):

        self.catalog = catalog or ModelCatalog()

    # -----------------------------------------------------

    def generate_candidate_models(

        self,

        profile: DatasetProfile

    ) -> List[CandidateModel]:

        models = self.catalog.get_all_models()

        return self._filter_task_type(

            models,

            profile.task_type

        )

    # -----------------------------------------------------

    def _filter_task_type(

        self,

        models,

        task_type

    ):

        return [

            model

            for model in models

            if model.task_type == task_type

        ]