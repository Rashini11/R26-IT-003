import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = r"model\hull_validator_hardneg.keras"
IMAGE_DIR = r"data\raw\images\non_hull_hard"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

filenames = [
    "Iron-oxide-bolt.webp",
    "Norderney,_Buhne_--_2025_--_9182.jpg",
]

for filename in filenames:
    path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"\nCould not read image: {filename}")
        continue

    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image.astype(np.float32), axis=0)

    # The model already contains MobileNetV2 preprocessing.
    prediction = float(model.predict(image, verbose=0)[0][0])
    hull_probability = 1.0 - prediction
    non_hull_probability = prediction

    print(f"\nImage: {filename}")
    print(f"Hull confidence:     {hull_probability:.2%}")
    print(f"Non-hull confidence: {non_hull_probability:.2%}")

    if hull_probability >= 0.70:
        print("Decision: HULL")
    elif non_hull_probability >= 0.70:
        print("Decision: NON-HULL")
    else:
        print("Decision: UNCERTAIN")
