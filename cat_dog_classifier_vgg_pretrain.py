from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.applications import VGG16


def model_specification():
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    conv_base.trainable = False

    model = models.Sequential(
        layers=[
            conv_base,
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        RMSprop(learning_rate=2e-5), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def image_generators():
    train_dir = "./data/dogs-vs-cats/manning/training"
    val_dir = "./data/dogs-vs-cats/manning/validation"
    test_dir = "./data/dogs-vs-cats/manning/test"

    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_data_gen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
    )

    val_generator = test_data_gen.flow_from_directory(
        val_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
    )

    test_generator = test_data_gen.flow_from_directory(
        test_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
    )
    return train_generator, val_generator, test_generator


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


def main():
    model = model_specification()
    train_generator, val_generator, test_generator = image_generators()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=val_generator,
        validation_steps=50,
    )

    plot_training_and_validation_performance(
        history, "./figures/cat_vs_dog_vgg_pretrain_perf.png"
    )
    print(model.evaluate_generator(test_generator, steps=50))

    model.save("./data/cat_vs_dog_vgg_pretrain_conv.h5")


if __name__ == "__main__":
    main()
