"""Builds an image classifier for fashion MNIST dataset using a simple sequential network

    Model (sequential):
        - flatten (28 x 28 -> 784)
        - dense (784 -> 300, relu)
        - dense (300 -> 100, relu)
        - dense (100 -> 10, softmax)
"""

from tensorflow import keras

if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    class_names = [
        "top",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle-boot",
    ]

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"],
    )

    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

    model.evaluate(X_test, y_test)
