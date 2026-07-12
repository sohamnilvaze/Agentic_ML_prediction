from core.explainability_artifact import ExplainabilityArtifact


class ExplainabilityPipeline:
    """
    Generates explainability outputs for all
    trained models.

    Responsibilities
    ----------------
    1. Extract feature importance.
    2. Build explanation reports.
    3. Package ExplainabilityArtifact.
    """

    # =====================================================

    def build(

        self,

        training_artifact,

        dataset_artifact

    ):

        explanations = {}

        for training_result in training_artifact.training_results:

            importance = self._extract_feature_importance(
                training_result
            )

            explanations[
                training_result.model_name
            ] = self._build_explanation(
                training_result,
                importance
            )

        return ExplainabilityArtifact(

            training_artifact=training_artifact,
            explanations=explanations
        )

    # =====================================================

    def _extract_feature_importance(

        self,

        training_result

    ):

        model = training_result.trained_model

        if hasattr(

            model,

            "feature_importances_"

        ):

            values = model.feature_importances_

        elif hasattr(

            model,

            "coef_"

        ):

            values = model.coef_

            if values.ndim > 1:

                values = values[0]

            values = abs(values)

        else:

            return {}

        feature_names = (

            training_result.preprocessing_summary.get(

                "feature_names",

                []

            )

        )

        if not feature_names:

            return {

                str(i): float(score)

                for i, score in enumerate(values)

            }

        return {

            feature: float(score)

            for feature, score in zip(

                feature_names,

                values

            )

        }

    # =====================================================

    def _build_explanation(

        self,

        training_result,

        importance

    ):

        metrics = training_result.evaluation_metrics

        top_features = sorted(

            importance.items(),

            key=lambda x: x[1],

            reverse=True

        )[:10]

        feature_list = [

            {

                "feature": feature,

                "importance": float(score)

            }

            for feature, score in top_features

        ]

        summary = (

            f"{training_result.model_name} achieved "

            f"an F1 score of "

            f"{metrics.get('f1',0):.3f}. "

            f"Top influential features: "

            +

            ", ".join(

                feature

                for feature, _ in top_features

            )

        )

        return {

            "metrics": metrics,

            "top_features": feature_list,

            "summary": summary

        }