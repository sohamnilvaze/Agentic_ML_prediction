from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class TrainingEvaluator:

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        predictions = model.predict(X_test)

        return {

            "accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "precision":
                precision_score(
                    y_test,
                    predictions
                ),

            "recall":
                recall_score(
                    y_test,
                    predictions
                ),

            "f1":
                f1_score(
                    y_test,
                    predictions
                )

        }