from agents.base_agent import BaseAgent

from core.registry_artifact import RegistryArtifact


class ModelRegistryAgent(BaseAgent):

    def __init__(

        self,

        model_saver

    ):

        self.model_saver = model_saver

    # -----------------------------------------------------

    def run(

        self,

        state

    ):

        saved_models = []

        training_artifact = (

            state.training_artifact

        )

        for training_result in (

            training_artifact.training_results

        ):

            saved_models.append(

                self.model_saver.save(

                    training_result

                )

            )

        registry_artifact = RegistryArtifact(

            training_artifact=training_artifact,

            saved_models=saved_models,

            registry_summary={

                "models_registered":

                    len(saved_models),

                "best_model":

                    training_artifact.best_training_result.model_name
                    if training_artifact.best_training_result
                    else None

            }

        )

        state.registry_artifact = registry_artifact
        state.current_stage = "REGISTRY_COMPLETED"
        state.add_history(
            f"Registered {len(saved_models)} trained models."
        )

        return state
