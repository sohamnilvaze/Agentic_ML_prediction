class BestModelSelector:
    """
    Selects the best trained model based on
    evaluation metrics.
    """

    DEFAULT_METRIC = "f1"

    # -----------------------------------------------------

    def select(

        self,

        training_results

    ):

        if not training_results:

            return None

        return max(

            training_results,

            key=lambda result:

                result.evaluation_metrics.get(

                    self.DEFAULT_METRIC,

                    0.0

                )

        )