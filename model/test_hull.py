import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = "model/hull_model.keras"

TEST_IMAGE = input("Enter path to a hull test image: ").strip()

classes = [
    "biofouling",
    "corrosion",
    "cracks",
    "non_hull",
    "paint_damage"
]

print("\nLoading hull model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

image = tf.keras.utils.load_img(
    TEST_IMAGE,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = image_array / 255.0
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(
    image_array,
    verbose=0
)[0]

print("\n========== HULL PREDICTION ==========")

for i, probability in enumerate(predictions):
    print(
        f"{classes[i]:15s}: "
        f"{probability * 100:.2f}%"
    )

predicted_index = np.argmax(predictions)

print("-------------------------------------")
print("Prediction :", classes[predicted_index])
print("Confidence :", f"{predictions[predicted_index] * 100:.2f}%")
print("=====================================")