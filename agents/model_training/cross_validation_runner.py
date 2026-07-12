from sklearn.model_selection import cross_validate


class CrossValidationRunner:
    """
    Runs cross-validation for a trained model.
    """

    DEFAULT_SCORING = [

        "accuracy",

        "precision",

        "recall",

        "f1"

    ]

    def run(

        self,

        model,

        X,

        y,

        cv=5

    ):

        results = cross_validate(

            estimator=model,

            X=X,

            y=y,

            cv=cv,

            scoring=self.DEFAULT_SCORING,

            return_train_score=False

        )

        summary = {}

        for key, values in results.items():

            if key.startswith("test_"):

                summary[key] = float(

                    values.mean()

                )

        return summary