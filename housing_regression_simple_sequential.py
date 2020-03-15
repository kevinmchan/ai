"""Build a simple regression neural network to predict California housing

    Model:
        - dense (m -> 30, relu)
        - dense (30 -> 1)
"""
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = keras.models.Sequential(
    [
        keras.layers.Dense(30, input_shape=X_train.shape[1:], activation="relu"),
        keras.layers.Dense(1),
    ],
)
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

mse = model.evaluate(X_test, y_test)
