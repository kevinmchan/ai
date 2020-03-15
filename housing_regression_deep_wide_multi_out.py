from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os


def load_data():
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target
    )

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "train": {"X": (X_train[:, :5], X_train[:, 2:]), "y": y_train},
        "val": {"X": (X_val[:, :5], X_val[:, 2:]), "y": y_val},
        "test": {"X": (X_test[:, :5], X_test[:, 2:]), "y": y_test},
    }


class DeepAndWideRegressor(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        hidden1 = self.hidden1(input_deep)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_wide, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


if __name__ == "__main__":
    data = load_data()

    model = DeepAndWideRegressor()
    model.compile(
        loss=["mse", "mse"],
        loss_weights=[0.9, 0.1],
        optimizer=keras.optimizers.SGD(lr=1e-3),
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "data/wide_and_deep_multi_output.h5"
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    tensorboard_cb = keras.callbacks.TensorBoard(
        os.path.join("logs", time.strftime("%Y_%m_%d_%H_%M_%S"))
    )

    history = model.fit(
        data["train"]["X"],
        [data["train"]["y"], data["train"]["y"]],
        epochs=1000,
        validation_data=(data["val"]["X"], [data["val"]["y"], data["val"]["y"]]),
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
    )

    mse = model.evaluate(data["test"]["X"], (data["test"]["y"], data["test"]["y"]))
