import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# =========================
# 📦 Load model
# =========================
model = tf.keras.models.load_model("model/hull_model.keras")

# ⚠️ MUST MATCH TRAINING ORDER
class_names = ['biofouling', 'corrosion', 'cracks', 'paint_damage']

# =========================
# 🧠 Prediction function
# =========================
def predict_image(img_path):
    if not os.path.exists(img_path):
        print("❌ Image not found!")
        return

    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 Prediction
    predictions = model.predict(img_array)[0]

    print("\n🔍 Prediction Breakdown:")
    for i, prob in enumerate(predictions):
        print(f"{class_names[i]}: {prob:.4f}")

    # Final result
    class_index = np.argmax(predictions)
    confidence = float(predictions[class_index])
    predicted_class = class_names[class_index]

    print("\n✅ FINAL RESULT")
    print("Prediction:", predicted_class)
    print("Confidence:", round(confidence, 4))

    # ⚠️ Confidence check
    if confidence < 0.6:
        print("⚠️ Model is uncertain. Consider manual inspection.")

# =========================
# ▶️ Run
# =========================
# =========================
# ▶️ Test multiple images
# =========================
folder_path = "test_images"

if not os.path.exists(folder_path):
    print("❌ test_images folder not found!")
else:
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            print("\n==============================")
            print("📷 Testing:", file)
            predict_image(os.path.join(folder_path, file))