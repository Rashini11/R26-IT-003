import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import seaborn as sns
from pathlib import Path
import json
import os

# =====================================================
# DATASET PATHS
# =====================================================

train_dir = "data/processed/images/train"
val_dir = "data/processed/images/val"
test_dir = "data/processed/images/test"

# =====================================================
# LOAD DATASETS
# =====================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=16,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# =====================================================
# CHECK CLASS ORDER
# =====================================================

print("\n========== HULL DATASET ==========")

print("TRAIN CLASSES:", train_ds.class_names)
print("VAL CLASSES:", val_ds.class_names)
print("TEST CLASSES:", test_ds.class_names)

class_names = train_ds.class_names

expected_classes = [
    "biofouling",
    "corrosion",
    "cracks",
    "non_hull",
    "paint_damage",
]

if class_names != expected_classes:
    raise ValueError(
        f"Unexpected class order: {class_names}\n"
        f"Expected: {expected_classes}"
    )

if val_ds.class_names != expected_classes:
    raise ValueError(
        f"Unexpected validation class order: {val_ds.class_names}"
    )

if test_ds.class_names != expected_classes:
    raise ValueError(
        f"Unexpected test class order: {test_ds.class_names}"
    )

# =====================================================
# DATASET DISTRIBUTION
# =====================================================

print("\n========== DATASET DISTRIBUTION ==========")

for class_name in class_names:

    train_count = len(
        tf.io.gfile.glob(
            f"{train_dir}/{class_name}/*"
        )
    )

    val_count = len(
        tf.io.gfile.glob(
            f"{val_dir}/{class_name}/*"
        )
    )

    test_count = len(
        tf.io.gfile.glob(
            f"{test_dir}/{class_name}/*"
        )
    )

    print(
        f"{class_name}: "
        f"train={train_count}, "
        f"val={val_count}, "
        f"test={test_count}"
    )

print("==========================================\n")

# =====================================================
# MOBILE NET V2 PREPROCESSING
# MobileNetV2 expects pixel values in [-1, 1]
# =====================================================

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess_dataset(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels

train_ds = train_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# =====================================================
# DATA AUGMENTATION
# =====================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# =====================================================
# CLASS WEIGHTS
# =====================================================

train_labels = np.concatenate(
    [y.numpy() for x, y in train_ds],
    axis=0
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_labels
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\n========== CLASS WEIGHTS ==========")

for i, weight in class_weights.items():
    print(
        f"{class_names[i]}: {weight:.3f}"
    )

print("===================================\n")

# =====================================================
# MOBILENETV2
# =====================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze most layers.
# Fine-tune only the final 30 layers.

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# =====================================================
# MODEL
# =====================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image_input"
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.BatchNormalization(
    name="batch_normalization"
)(x)

x = layers.Dense(
    128,
    activation="relu",
    name="dense_128"
)(x)

x = layers.Dropout(
    0.4,
    name="dropout"
)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax",
    name="hull_prediction"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# =====================================================
# TRAIN
# =====================================================

print("\n========== STARTING TRAINING ==========\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# SAVE CLASS NAMES
# =====================================================

os.makedirs("model", exist_ok=True)

with open(
    "model/hull_class_names.json",
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=2
    )

# =====================================================
# SAVE MODEL
# =====================================================

model.save(
    "model/hull_model.keras",
    include_optimizer=False
)

print(
    "\nModel saved to:"
    "\nmodel/hull_model.keras"
)

# =====================================================
# TRAINING ACCURACY GRAPH
# =====================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Hull Defect Model Accuracy")
plt.legend()

plt.savefig(
    "model/training_accuracy.png"
)

plt.close()

# =====================================================
# TEST EVALUATION
# =====================================================

print("\n========== TEST EVALUATION ==========\n")

test_loss, test_acc = model.evaluate(
    test_ds
)

print(
    f"Test Accuracy: {test_acc * 100:.2f}%"
)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

y_pred = model.predict(
    test_ds
)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

y_true = []

for images, labels in test_ds:
    y_true.extend(
        labels.numpy()
    )

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=class_names,
        digits=4
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred_classes
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hull Defect Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png"
)

plt.close()

print(
    "\nTraining and evaluation completed successfully."
)