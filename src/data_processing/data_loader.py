import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, val_dir, test_dir, image_size, batch_size):
    """
    Creates and configures data generators for training, validation, and testing.

    This function sets up ImageDataGenerator for data augmentation on the training
    set and simple rescaling for the validation and test sets.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        test_dir (str): Path to the test data directory.
        image_size (tuple): The target dimensions for the images (height, width).
        batch_size (int): The number of images to process in each batch.

    Returns:
        tuple: A tuple containing the training, validation, and test generators.
    """

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )


    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = val_datagen.flow_from_directory(
        directory=test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator