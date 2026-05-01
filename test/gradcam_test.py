import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gradcam import (
    get_img_array,
    make_gradcam_heatmap,
    overlay_heatmap
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("model/hull_model.h5", compile=False)

# Force model to build graph
dummy_input = tf.random.normal([1, 224, 224, 3])
_ = model(dummy_input)

# -----------------------------
# IMAGE PATH (MUST BE FIRST)
# -----------------------------
img_path = "test_images/C1.jpg"

# -----------------------------
# PREPROCESS IMAGE
# -----------------------------
img_array = get_img_array(img_path)

_ = model(img_array)

# -----------------------------
# GET LAST CONV LAYER
# -----------------------------
last_conv_layer_name = None

for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

print("Using layer:", last_conv_layer_name)

# -----------------------------
# GRAD-CAM
# -----------------------------
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# -----------------------------
# OVERLAY HEATMAP
# -----------------------------
result = overlay_heatmap(img_path, heatmap)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_path = "results/gradcam_output.jpg"
cv2.imwrite(output_path, result)

print("Saved to:", output_path)

# -----------------------------
# SHOW RESULT
# -----------------------------
plt.imshow(cv2.cvtColor(result.astype("uint8"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()