
import os
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = r"D:\Research\R26-IT-003"

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "hull_validator_hardneg.keras"
)

IMAGE_DIR = os.path.join(
    BASE_DIR, "data", "raw", "images", "non_hull_hard"
)

IMG_SIZE = (224, 224)
THRESHOLD = 0.70

print("Loading hard-negative validator...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
]

print(f"\nImages found: {len(image_files)}")
print("Note: these images were included in training, so this is")
print("a diagnostic check, NOT an independent accuracy test.\n")

for filename in sorted(image_files):
    path = os.path.join(IMAGE_DIR, filename)

    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(IMG_SIZE)
            array = np.asarray(image, dtype=np.float32)

        probability = float(
            np.squeeze(model.predict(np.expand_dims(array, axis=0), verbose=0))
        )

        # The training script defines sigmoid output as:
        # 0 = hull, 1 = non_hull
        non_hull_probability = probability
        hull_probability = 1.0 - probability

        if non_hull_probability >= THRESHOLD:
            result = "REJECT: NON-HULL"
        elif hull_probability >= THRESHOLD:
            result = "ACCEPT: HULL"
        else:
            result = "UNCERTAIN: REJECT"

        print(
            f"{filename}\n"
            f"  Hull: {hull_probability:.1%} | "
            f"Non-hull: {non_hull_probability:.1%}\n"
            f"  Result: {result}\n"
        )

    except Exception as exc:
        print(f"Could not process {filename}: {exc}\n")