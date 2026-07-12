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

        X = dataframe.drop(columns=[target_column])

        y = dataframe[target_column]

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )