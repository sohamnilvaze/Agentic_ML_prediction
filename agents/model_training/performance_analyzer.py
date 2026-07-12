from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report

)


class PerformanceAnalyzer:
    """
    Computes evaluation metrics
    for a trained model.
    """

    def analyze(

        self,

        model,

        X_test,

        y_test

    ):

        predictions = model.predict(

            X_test

        )

        metrics = {

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

                ),

            "confusion_matrix":

                confusion_matrix(

                    y_test,

                    predictions

                ),

            "classification_report":

                classification_report(

                    y_test,

                    predictions,

                    output_dict=True

                )

        }

        return metrics