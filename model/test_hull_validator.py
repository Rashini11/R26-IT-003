import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "model/hull_validator.keras"

TRAIN_DIR = "data/processed/hull_validator/train"
VAL_DIR = "data/processed/hull_validator/val"

print("\n==============================================")
print("HULL VALIDATOR TRAIN / VALIDATION TEST")
print("==============================================\n")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
print("Model loaded successfully.\n")

# IMPORTANT:
# Your validator was tested using raw pixel values (0-255),
# so we intentionally do NOT use rescale=1./255 here.
datagen = ImageDataGenerator()

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

print("\n==============================================")
print("CLASS INDICES")
print("==============================================")

print(train_gen.class_indices)

print("\n==============================================")
print("TRAINING PERFORMANCE")
print("==============================================")

train_loss, train_acc = model.evaluate(train_gen, verbose=1)

print(f"\nTraining Loss:     {train_loss:.4f}")
print(f"Training Accuracy: {train_acc * 100:.2f}%")

print("\n==============================================")
print("VALIDATION PERFORMANCE")
print("==============================================")

val_loss, val_acc = model.evaluate(val_gen, verbose=1)

print(f"\nValidation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

print("\n==============================================")
print("SUMMARY")
print("==============================================")

print(f"Training Accuracy:   {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Accuracy Gap:        {(train_acc - val_acc) * 100:.2f}%")

print("\n==============================================")