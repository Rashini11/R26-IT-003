import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load trained model
# =========================
model = tf.keras.models.load_model("model/hull_model.keras")

# =========================
# Dataset
# =========================
test_dir = "data/processed/images/test"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:", class_names)

# =========================
# Normalization
# =========================
normalization_layer = tf.keras.layers.Rescaling(1./255)

test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

# =========================
# Test evaluation
# =========================
print("\n==============================")
print("TEST EVALUATION")
print("==============================")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Accuracy:", round(test_accuracy, 4))
print("Test Accuracy (%):", round(test_accuracy * 100, 2))

# =========================
# Predictions
# =========================
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

# =========================
# True labels
# =========================
y_true = []

for images, labels in test_ds:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

# =========================
# Classification Report
# =========================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=class_names,
        digits=4
    )
)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred_classes)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

# =========================
# Display confusion matrix
# =========================
plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hull Defect Classification - Confusion Matrix")

plt.tight_layout()

# Save instead of plt.show()
plt.savefig("model/confusion_matrix.png")

print("\nConfusion matrix saved as:")
print("model/confusion_matrix.png")