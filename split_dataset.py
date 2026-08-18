import os
import random
import shutil

# Paths
SOURCE_DIR = "data/raw/images"
DEST_DIR = "data/processed/images"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = os.listdir(SOURCE_DIR)

for cls in classes:

    src_folder = os.path.join(SOURCE_DIR, cls)

    images = [
        img for img in os.listdir(src_folder)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split, split_images in {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }.items():

        dest_folder = os.path.join(DEST_DIR, split, cls)
        os.makedirs(dest_folder, exist_ok=True)

        for img in split_images:
            shutil.copy(
                os.path.join(src_folder, img),
                os.path.join(dest_folder, img)
            )

print("Dataset split completed successfully!")
