from agents.base_agent import BaseAgent

from .explainability_pipeline import (
    ExplainabilityPipeline
)


class ExplainabilityAgent(BaseAgent):
    """
    Generates explanations for trained models.
    """

    # -----------------------------------------------------

    def __init__(

        self,

        pipeline

    ):

        self.pipeline = pipeline

    # =====================================================

    def run(

        self,

        state

    ):

        explainability_artifact = self.pipeline.build(

            training_artifact=state.training_artifact,

            dataset_artifact=state.dataset_artifact

        )

        state.explainability_artifact = (

            explainability_artifact

        )

        return state