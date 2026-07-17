import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Performs dataset preparation before training.
    """

    def prepare(
        self,
        dataframe,
        target_column,
        test_size=0.2,
        random_state=42
    ):

        X = dataframe.drop(columns=[target_column]).copy()
        y = dataframe[target_column].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        numeric_columns = list(
            X_train.select_dtypes(include=["number", "bool"]).columns
        )
        categorical_columns = [
            column
            for column in X_train.columns
            if column not in numeric_columns
        ]

        for column in numeric_columns:
            median = X_train[column].median()
            if pd.isna(median):
                median = 0
            X_train[column] = X_train[column].fillna(median)
            X_test[column] = X_test[column].fillna(median)

        for column in categorical_columns:
            X_train[column] = X_train[column].fillna("Missing").astype(str)
            X_test[column] = X_test[column].fillna("Missing").astype(str)

        if categorical_columns:
            X_train = pd.get_dummies(
                X_train,
                columns=categorical_columns,
                drop_first=False,
            )
            X_test = pd.get_dummies(
                X_test,
                columns=categorical_columns,
                drop_first=False,
            )

            X_train, X_test = X_train.align(
                X_test,
                join="outer",
                axis=1,
                fill_value=0,
            )

        return X_train, X_test, y_train, y_test
