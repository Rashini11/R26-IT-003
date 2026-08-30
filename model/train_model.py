import tensorflow as tf
import numpy as np
import cv2
import glob
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/hull_validator.keras"

TEST_ROOT = "data/processed/images/test"

THRESHOLDS = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("HULL / NON-HULL VALIDATOR TEST")
print("==============================================\n")

print("Loading validator model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Validator loaded successfully.\n")


# ============================================================
# TEST DATA
# ============================================================

folders = [
    ("hull", "biofouling"),
    ("hull", "corrosion"),
    ("hull", "cracks"),
    ("hull", "paint_damage"),
    ("non_hull", "non_hull"),
]


results = []


# ============================================================
# RUN PREDICTIONS
# ============================================================

for expected_label, folder in folders:

    folder_path = os.path.join(
        TEST_ROOT,
        folder,
    )

    files = glob.glob(
        os.path.join(
            folder_path,
            "*",
        )
    )

    print(
        f"Testing {folder}: {len(files)} images"
    )

    for file_path in files:

        image = cv2.imread(file_path)

        if image is None:
            print(
                "WARNING: Could not read:",
                file_path,
            )
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (224, 224),
        )

        image = image.astype(
            np.float32
        )

        image_array = np.expand_dims(
            image,
            axis=0,
        )

        prediction = model.predict(
            image_array,
            verbose=0,
        )[0][0]

        non_hull_probability = float(
            prediction
        )

        results.append(
            {
                "expected": expected_label,
                "file": file_path,
                "probability": non_hull_probability,
            }
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

total = len(results)

hull_count = sum(
    1
    for r in results
    if r["expected"] == "hull"
)

non_hull_count = sum(
    1
    for r in results
    if r["expected"] == "non_hull"
)


print("\n==============================================")
print("DATASET SUMMARY")
print("==============================================")

print(
    f"Total images:    {total}"
)

print(
    f"Hull images:     {hull_count}"
)

print(
    f"Non-hull images: {non_hull_count}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n==============================================")
print("THRESHOLD ANALYSIS")
print("==============================================")

for threshold in THRESHOLDS:

    correctly_rejected_non_hull = sum(
        1
        for r in results
        if (
            r["expected"] == "non_hull"
            and r["probability"] >= threshold
        )
    )

    correctly_accepted_hull = sum(
        1
        for r in results
        if (
            r["expected"] == "hull"
            and r["probability"] < threshold
        )
    )

    non_hull_accuracy = (
        correctly_rejected_non_hull
        / non_hull_count
        * 100
        if non_hull_count
        else 0
    )

    hull_accuracy = (
        correctly_accepted_hull
        / hull_count
        * 100
        if hull_count
        else 0
    )

    total_correct = (
        correctly_rejected_non_hull
        + correctly_accepted_hull
    )

    total_accuracy = (
        total_correct
        / total
        * 100
        if total
        else 0
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"  Non-hull correctly rejected: "
        f"{correctly_rejected_non_hull}/{non_hull_count} "
        f"({non_hull_accuracy:.2f}%)"
    )

    print(
        f"  Hull correctly accepted: "
        f"{correctly_accepted_hull}/{hull_count} "
        f"({hull_accuracy:.2f}%)"
    )

    print(
        f"  Overall accuracy: "
        f"{total_accuracy:.2f}%"
    )


# ============================================================
# MISCLASSIFIED AT CURRENT THRESHOLD
# ============================================================

CURRENT_THRESHOLD = 0.70


print("\n==============================================")
print(
    f"MISCLASSIFIED AT {CURRENT_THRESHOLD * 100:.0f}%"
)
print("==============================================")


misclassified = []


for r in results:

    probability = r["probability"]

    expected = r["expected"]

    predicted = (
        "non_hull"
        if probability >= CURRENT_THRESHOLD
        else "hull"
    )

    if predicted != expected:

        misclassified.append(r)

        print(
            f"\nExpected : {expected}"
        )

        print(
            f"Predicted: {predicted}"
        )

        print(
            f"Non-hull probability: "
            f"{probability * 100:.2f}%"
        )

        print(
            f"File: {os.path.basename(r['file'])}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("FINAL SUMMARY")
print("==============================================")

print(
    f"Misclassified: "
    f"{len(misclassified)}/{total}"
)

print(
    f"Accuracy: "
    f"{(total - len(misclassified)) / total * 100:.2f}%"
)

print("==============================================\n")