import os
import shutil
import random

# Paths
source_dir = "data/raw/images"
base_dir = "data/processed/images"

splits = ["train", "val", "test"]
split_ratio = [0.7, 0.15, 0.15]

# Create folders
for split in splits:
    for category in os.listdir(source_dir):
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

# Split data
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    train_end = int(0.7 * len(images))
    val_end = int(0.85 * len(images))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(base_dir, "train", category, img))

    for img in val_imgs:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(base_dir, "val", category, img))

    for img in test_imgs:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(base_dir, "test", category, img))

print("Dataset split completed!")