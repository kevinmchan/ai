"""Build a "deep and wide" regression neural network to predict California housing
using the keras functional api.

    Model:
        - concat (30, m -> 30 + m)
            - layers (m -> 30)
                - input (m)
                - dense (m -> 30)
                - dense (30 -> 30)
            - input (m)
        - dense (30 + m -> 1)
"""
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target
    )

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
    X_val_wide, X_val_deep = X_val[:, :5], X_val[:, 2:]
    X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]

    input_wide = keras.layers.Input(shape=[5], name="wide_input")
    input_deep = keras.layers.Input(shape=[6], name="deep_input")
    hidden_1 = keras.layers.Dense(30, activation="relu", name="hidden1")(input_deep)
    hidden_2 = keras.layers.Dense(30, activation="relu", name="hidden2")(hidden_1)
    concat = keras.layers.concatenate([input_wide, hidden_2], name="concat")
    output = keras.layers.Dense(1, name="output")(concat)
    model = keras.Model(inputs=[input_wide, input_deep], outputs=[output])
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))

    history = model.fit(
        (X_train_wide, X_train_deep),
        y_train,
        epochs=30,
        validation_data=((X_val_wide, X_val_deep), y_val),
    )

    mse = model.evaluate((X_test_wide, X_test_deep), y_test)
