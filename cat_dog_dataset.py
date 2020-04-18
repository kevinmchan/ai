import os
import shutil


def copy_images(source, destination, files):
    for fname in files:
        source_file = os.path.join(source, fname)
        destination_file = os.path.join(destination, fname)
        shutil.copyfile(source_file, destination_file)


def create_image_set(source, destination, indices):
    for category in ("cat", "dog"):
        directory = os.path.join(destination, category)
        os.makedirs(directory, exist_ok=True)
        copy_images(source, directory, [f"{category}.{i}.jpg" for i in indices])


def main():
    original_dataset = "./data/dogs-vs-cats/train"
    new_dataset = "./data/dogs-vs-cats/manning"
    training_dataset = os.path.join(new_dataset, "training")
    validation_dataset = os.path.join(new_dataset, "validation")
    test_dataset = os.path.join(new_dataset, "test")

    train_size = 10_000
    val_size = 1_500
    test_size = 1_000

    create_image_set(original_dataset, training_dataset, range(train_size))
    create_image_set(
        original_dataset, validation_dataset, range(train_size, train_size + val_size)
    )
    create_image_set(
        original_dataset,
        test_dataset,
        range(train_size + val_size, train_size + val_size + test_size),
    )


if __name__ == "__main__":
    main()
