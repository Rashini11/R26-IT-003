import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = r"model\hull_validator_hardneg.keras"
IMAGE_DIR = r"data\raw\images\non_hull_hard"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
results = []

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith(extensions):
        continue

    path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image.astype(np.float32), axis=0)

    prediction = float(model.predict(image, verbose=0)[0][0])
    hull = 1.0 - prediction
    non_hull = prediction

    if hull >= 0.70:
        decision = "FALSE POSITIVE - classified as HULL"
    elif non_hull >= 0.70:
        decision = "CORRECT - classified as NON-HULL"
    else:
        decision = "UNCERTAIN"

    results.append((filename, hull, non_hull, decision))

for filename, hull, non_hull, decision in results:
    print(f"\n{filename}")
    print(f"  Hull: {hull:.2%} | Non-hull: {non_hull:.2%}")
    print(f"  {decision}")

print("\n========== SUMMARY ==========")
print(f"Images tested: {len(results)}")
print(f"False positives: {sum('FALSE POSITIVE' in r[3] for r in results)}")
print(f"Correct non-hull: {sum('CORRECT' in r[3] for r in results)}")
print(f"Uncertain: {sum('UNCERTAIN' in r[3] for r in results)}")

