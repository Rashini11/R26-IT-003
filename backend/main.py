from fastapi import FastAPI, UploadFile, File
import os
import tensorflow as tf
import numpy as np
import cv2


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    class_idx = np.argmax(preds)
    prediction = classes[class_idx]
    confidence = float(np.max(preds))

    if confidence < 0.7:
        warning = "Low confidence. Manual inspection recommended."
    else:
        warning = "Prediction reliable."

    return {
        "prediction": prediction,
        "confidence": confidence,
        "recommendation": recommendations[prediction],
        "warning": warning
    }