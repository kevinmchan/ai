from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping


BATCH_SIZE = 32
TRAIN_SIZE = 20_000
VAL_SIZE = 3_000
TEST_SIZE = 2_000
START_FINE_TUNING_LAYER = 144

def model_specification():
    conv_base = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    print("Number of layers in the base model: ", len(conv_base.layers))
    conv_base.trainable = True
    for layer in conv_base.layers[:START_FINE_TUNING_LAYER]:
        layer.trainable =  False
    
    model = models.Sequential(
        layers=[
            conv_base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"],
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
        train_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
    )

    val_generator = test_data_gen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
    )

    test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
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
    model.summary()
    train_generator, val_generator, test_generator = image_generators()
    model_checkpoint = ModelCheckpoint(
        "./data/cat_dog_mobilenet_pooled_finetuned_best.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=30,
        validation_data=val_generator,
        validation_steps=100,
        callbacks=[model_checkpoint, es],
    )
    model.save("./data/cat_dog_mobilenet_pooled_finetuned_conv.h5")
    model.load_weights("./data/cat_dog_mobilenet_pooled_finetuned_best.h5")

    plot_training_and_validation_performance(
        history, "./figures/cat_dog_mobilenet_pooled_finetuned_perf.png"
    )
    print(model.evaluate_generator(test_generator, steps=TEST_SIZE // BATCH_SIZE))


if __name__ == "__main__":
    main()
