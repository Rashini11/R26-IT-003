from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf
import numpy as np
import cv2
import base64

app = FastAPI()

# ✅ CORS (already correct)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "hull_model.keras")

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

classes = ["biofouling", "corrosion", "cracks"]

recommendations = {
    "biofouling": "Clean hull using high-pressure water or antifouling treatment.",
    "corrosion": "Apply anti-corrosion coating or replace damaged metal.",
    "cracks": "Critical damage. Perform welding repair immediately."
}

# 🔥 Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    conv_outputs = None

    with tf.GradientTape() as tape:
        x = img_tensor

        for layer in model.layers:
            x = layer(x)

            if layer.name == last_conv_layer_name:
                conv_outputs = x
                tape.watch(conv_outputs)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    original_img = img.copy()

    # preprocess
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    preds = model.predict(img_array)

    class_idx = np.argmax(preds)
    prediction = classes[class_idx]
    confidence = float(np.max(preds))

    LAST_CONV_LAYER = "mobilenetv2_1.00_224"


    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

    # convert heatmap to image
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # encode image
    _, buffer = cv2.imencode(".jpg", superimposed_img)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    warning = "Low confidence. Manual inspection recommended." if confidence < 0.7 else "Prediction reliable."

    return {
        "prediction": prediction,
        "confidence": confidence,
        "recommendation": recommendations[prediction],
        "warning": warning,
        "gradcam": gradcam_base64
    }