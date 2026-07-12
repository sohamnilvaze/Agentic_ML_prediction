import time


class ModelTrainer:

    def train(
        self,
        model,
        X_train,
        y_train
    ):

        start = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = time.time() - start

        return model, training_time