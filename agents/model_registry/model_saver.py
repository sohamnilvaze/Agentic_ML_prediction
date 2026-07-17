import json
import os
import joblib

from datetime import datetime

from core.config import REGISTRY_DIR


class ModelSaver:
    """
    Saves trained models and metadata.
    """

    def __init__(

        self,

        output_directory=REGISTRY_DIR

    ):

        self.output_directory = output_directory

        os.makedirs(

            self.output_directory,

            exist_ok=True

        )

    # -----------------------------------------------------

    def save(

        self,

        training_result

    ):

        model_name = training_result.model_name
        safe_model_name = model_name.replace(" ", "_")

        timestamp = datetime.now().strftime(

            "%Y%m%d_%H%M%S"

        )

        model_filename = (

            f"{safe_model_name}_{timestamp}.joblib"

        )

        metadata_filename = (

            f"{safe_model_name}_{timestamp}_metadata.json"

        )

        model_path = os.path.join(

            self.output_directory,

            model_filename

        )

        metadata_path = os.path.join(

            self.output_directory,

            metadata_filename

        )

        joblib.dump(

            training_result.trained_model,

            model_path

        )

        metadata = {

            "model_name":

                model_name,

            "evaluation_metrics":

                training_result.evaluation_metrics,

            "cross_validation_metrics":

                training_result.cross_validation_metrics,

            "training_time_seconds":

                training_result.training_time_seconds,

            "candidate_model":

                training_result.candidate_model.to_dict()

                if hasattr(
                    training_result.candidate_model,
                    "to_dict"
                )
                else {},

            "preprocessing_summary":

                training_result.preprocessing_summary

        }

        with open(

            metadata_path,

            "w"

        ) as f:

            json.dump(

                metadata,

                f,

                indent=4

            )

        return {

            "model_name": model_name,

            "timestamp": timestamp,

            "model_path": model_path,

            "metadata_path": metadata_path

        }
