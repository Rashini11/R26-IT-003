import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = "data/processed/images/test"

CLASS_NAMES = [
    "biofouling",
    "corrosion",
    "cracks",
    "paint_damage"
]

# ============================================================
# LOAD TEST DATA
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=CLASS_NAMES
)

print("\nClasses:", test_ds.class_names)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    "model/hull_model.keras"
)

print("\nModel loaded successfully.")

# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n==============================")
print("FOUR-CLASS HULL DEFECT TEST")
print("==============================")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))
print("Test Accuracy (%):", round(test_accuracy * 100, 2))

# ============================================================
# PREDICTIONS
# ============================================================

y_pred_prob = model.predict(test_ds)
y_pred = np.argmax(y_pred_prob, axis=1)

# ============================================================
# TRUE LABELS
# ============================================================

y_true = []

for images, labels in test_ds:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

# ============================================================
# SAVE REPORT
# ============================================================

with open("model/hull_defect_evaluation_report.txt", "w") as f:

    f.write("FOUR-CLASS HULL DEFECT CLASSIFICATION REPORT\n")
    f.write("============================================\n\n")

    f.write(f"Test Dataset Size: {len(y_true)}\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hull Defect Classification - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "model/hull_defect_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nReport saved to:")
print("model/hull_defect_evaluation_report.txt")

print("\nConfusion matrix saved to:")
print("model/hull_defect_confusion_matrix.png")