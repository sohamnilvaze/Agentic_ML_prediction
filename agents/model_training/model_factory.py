from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class ModelFactory:
    """
    Creates ML model instances from CandidateModel definitions.
    """

    def create(self, candidate_model):

        algorithm = candidate_model.algorithm
        params = candidate_model.default_parameters

        if algorithm == "logistic_regression":
            return LogisticRegression(**params)

        if algorithm == "decision_tree":
            return DecisionTreeClassifier(**params)

        if algorithm == "random_forest":
            return RandomForestClassifier(**params)

        if algorithm == "gradient_boosting":
            return GradientBoostingClassifier(**params)

        if algorithm == "svm":
            return SVC(**params)

        if algorithm == "knn":
            return KNeighborsClassifier(**params)

        if algorithm == "naive_bayes":
            return GaussianNB(**params)

        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )