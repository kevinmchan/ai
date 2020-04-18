from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.applications import VGG16
import numpy as np


def image_generators():
    train_dir = "./data/dogs-vs-cats/manning/training"
    val_dir = "./data/dogs-vs-cats/manning/validation"
    test_dir = "./data/dogs-vs-cats/manning/test"

    generator = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = generator.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary",
        shuffle=False,
    )

    val_generator = generator.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary",
        shuffle=False,
    )

    test_generator = generator.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary",
        shuffle=False,
    )
    return train_generator, val_generator, test_generator


def extract_features(model, generator, max_samples):
    features = None
    labels = None
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        features_batch = features_batch.reshape((features_batch.shape[0], -1))
        if features is not None:
            features = np.concatenate([features, features_batch])
            labels = np.concatenate([labels, labels_batch])
        else:
            features = features_batch
            labels = labels_batch
        if features.shape[0] >= max_samples:
            break
    return features, labels


def plot_training_and_validation_performance(history, output):
    import matplotlib.pyplot as plt

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.savefig(output)


conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

train_generator, val_generator, test_generator = image_generators()
train_features, train_labels = extract_features(conv_base, train_generator, 2000)
val_features, val_labels = extract_features(conv_base, val_generator, 1000)
test_features, test_labels = extract_features(conv_base, test_generator, 1000)

model = models.Sequential(
    [
        layers.Dense(256, activation="relu", input_dim=train_features.shape[1]),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=RMSprop(learning_rate=2e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_features,
    train_labels,
    batch_size=20,
    epochs=30,
    validation_data=(val_features, val_labels),
)

plot_training_and_validation_performance(history, "figures/cat_vs_dog_vgg_features.png")
