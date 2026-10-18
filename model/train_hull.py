import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "data/processed/images"

CLASS_NAMES = [
    "biofouling",
    "corrosion",
    "cracks",
    "paint_damage"
]


# ============================================================
# LOAD DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + "/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12)
], name="data_augmentation")


# ============================================================
# MOBILE NET V2 BACKBONE
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze most pretrained layers
for layer in base_model.layers[:-27]:
    layer.trainable = False

# Fine-tune the last 27 layers
for layer in base_model.layers[-27:]:
    layer.trainable = True


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="image_input"
)

x = data_augmentation(inputs)

x = base_model(x)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.BatchNormalization(
    name="batch_normalization"
)(x)

x = layers.Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)

x = layers.Dropout(
    0.35,
    name="dropout"
)(x)

outputs = layers.Dense(
    4,
    activation="softmax",
    name="hull_prediction"
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save("model/hull_model.keras")

print("Training completed and model saved.")