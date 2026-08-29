import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = r"D:\Research\R26-IT-003"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "images",
    "train",
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "images",
    "val",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hull_validator.keras",
)

# IMPORTANT:
#
# 0 = hull image
# 1 = non-hull image
#
# "hull" means ANY of the four defect classes.
# "non_hull" means unrelated images.

HULL_CLASSES = [
    "biofouling",
    "corrosion",
    "cracks",
    "paint_damage",
]

VALIDATOR_CLASSES = [
    "hull",
    "non_hull",
]


print("\n==============================================")
print("TRAINING HULL / NON-HULL VALIDATOR")
print("==============================================")
print("0 = hull")
print("1 = non_hull")
print()


# ============================================================
# CREATE TEMPORARY VALIDATOR DATASET STRUCTURE
# ============================================================

# We create:
#
# validator_train/
#     hull/
#     non_hull/
#
# validator_val/
#     hull/
#     non_hull/
#
# Hull images are NOT copied.
# We create symlinks where possible, otherwise copy files.

import shutil


TEMP_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hull_validator",
)


TEMP_TRAIN_DIR = os.path.join(
    TEMP_DIR,
    "train",
)

TEMP_VAL_DIR = os.path.join(
    TEMP_DIR,
    "val",
)


def create_validator_dataset(source_dir, destination_dir):

    os.makedirs(destination_dir, exist_ok=True)

    hull_destination = os.path.join(
        destination_dir,
        "hull",
    )

    non_hull_destination = os.path.join(
        destination_dir,
        "non_hull",
    )

    os.makedirs(hull_destination, exist_ok=True)
    os.makedirs(non_hull_destination, exist_ok=True)

    # --------------------------------------------------------
    # Copy/link all four hull defect classes into hull folder
    # --------------------------------------------------------

    for class_name in HULL_CLASSES:

        source_class_dir = os.path.join(
            source_dir,
            class_name,
        )

        if not os.path.exists(source_class_dir):
            raise FileNotFoundError(
                f"Missing class directory: {source_class_dir}"
            )

        for filename in os.listdir(source_class_dir):

            source_file = os.path.join(
                source_class_dir,
                filename,
            )

            if not os.path.isfile(source_file):
                continue

            destination_file = os.path.join(
                hull_destination,
                f"{class_name}_{filename}",
            )

            if not os.path.exists(destination_file):

                try:
                    os.link(
                        source_file,
                        destination_file,
                    )

                except (OSError, NotImplementedError):

                    shutil.copy2(
                        source_file,
                        destination_file,
                    )

    # --------------------------------------------------------
    # Copy/link non-hull images
    # --------------------------------------------------------

    source_non_hull_dir = os.path.join(
        source_dir,
        "non_hull",
    )

    if not os.path.exists(source_non_hull_dir):
        raise FileNotFoundError(
            f"Missing non_hull directory: {source_non_hull_dir}"
        )

    for filename in os.listdir(source_non_hull_dir):

        source_file = os.path.join(
            source_non_hull_dir,
            filename,
        )

        if not os.path.isfile(source_file):
            continue

        destination_file = os.path.join(
            non_hull_destination,
            filename,
        )

        if not os.path.exists(destination_file):

            try:
                os.link(
                    source_file,
                    destination_file,
                )

            except (OSError, NotImplementedError):

                shutil.copy2(
                    source_file,
                    destination_file,
                )


# ============================================================
# BUILD VALIDATOR DATASET
# ============================================================

print("Preparing validator dataset...")

create_validator_dataset(
    TRAIN_DIR,
    TEMP_TRAIN_DIR,
)

create_validator_dataset(
    VAL_DIR,
    TEMP_VAL_DIR,
)

print("Validator dataset ready.")


# ============================================================
# LOAD DATASETS
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TEMP_TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=VALIDATOR_CLASSES,
    label_mode="binary",
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEMP_VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=VALIDATOR_CLASSES,
    label_mode="binary",
    shuffle=False,
)


print()
print("==============================================")
print("VALIDATOR CLASSES")
print("==============================================")
print(train_ds.class_names)
print()
print("0 = hull")
print("1 = non_hull")
print()


# ============================================================
# DATA COUNTS
# ============================================================

hull_train_count = len(
    tf.io.gfile.glob(
        os.path.join(
            TEMP_TRAIN_DIR,
            "hull",
            "*",
        )
    )
)

non_hull_train_count = len(
    tf.io.gfile.glob(
        os.path.join(
            TEMP_TRAIN_DIR,
            "non_hull",
            "*",
        )
    )
)

hull_val_count = len(
    tf.io.gfile.glob(
        os.path.join(
            TEMP_VAL_DIR,
            "hull",
            "*",
        )
    )
)

non_hull_val_count = len(
    tf.io.gfile.glob(
        os.path.join(
            TEMP_VAL_DIR,
            "non_hull",
            "*",
        )
    )
)


print("==============================================")
print("DATASET DISTRIBUTION")
print("==============================================")

print(
    f"TRAIN hull:     {hull_train_count}"
)

print(
    f"TRAIN non_hull: {non_hull_train_count}"
)

print(
    f"VAL hull:       {hull_val_count}"
)

print(
    f"VAL non_hull:   {non_hull_val_count}"
)

print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = np.concatenate(
    [
        labels.numpy().reshape(-1)
        for images, labels in train_ds
    ],
    axis=0,
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_labels.astype(int),
)

class_weights = {
    0: float(class_weights_array[0]),
    1: float(class_weights_array[1]),
}


print("==============================================")
print("CLASS WEIGHTS")
print("==============================================")

print(
    f"hull:     {class_weights[0]:.4f}"
)

print(
    f"non_hull: {class_weights[1]:.4f}"
)

print()


# ============================================================
# DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation",
)


# ============================================================
# MOBILENETV2
# ============================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(
        224,
        224,
        3,
    ),
)

base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        224,
        224,
        3,
    ),
    name="input_image",
)

x = data_augmentation(
    inputs
)

x = preprocess_input(
    x
)

x = base_model(
    x,
    training=False,
)

x = layers.GlobalAveragePooling2D()(
    x
)

x = layers.BatchNormalization()(
    x
)

x = layers.Dense(
    128,
    activation="relu",
)(
    x
)

x = layers.Dropout(
    0.4
)(
    x
)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="hull_validator_output",
)(
    x
)


model = models.Model(
    inputs,
    outputs,
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
    ],
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print()
print("==============================================")
print("STARTING HULL VALIDATOR TRAINING")
print("==============================================")
print()

history = model.fit(
    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(
        MODEL_PATH
    ),
    exist_ok=True,
)

model.save(
    MODEL_PATH
)


# ============================================================
# FINAL VALIDATOR CHECK
# ============================================================

print()
print("==============================================")
print("HULL VALIDATOR TRAINING COMPLETE")
print("==============================================")

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Input:",
    model.input_shape,
)

print(
    "Output:",
    model.output_shape,
)

print(
    "Activation:",
    model.layers[-1].activation.__name__,
)

print("==============================================")