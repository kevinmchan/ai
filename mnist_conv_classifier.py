from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_img, train_label), (test_img, test_label) = mnist.load_data()
train_img = train_img[:, :, :, None].astype("float32") / 255
test_img = test_img[:, :, :, None].astype("float32") / 255
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model.summary()
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.fit(
    train_img[:-10_000],
    train_label[:-10_000],
    epochs=5,
    batch_size=64,
    validation_data=(train_img[-10_000:], train_label[-10_000:]),
)

print(model.evaluate(test_img, test_label))
