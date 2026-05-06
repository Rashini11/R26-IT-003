from pathlib import Path
import random
import shutil

random.seed(42)

source_base = Path("ml/dataset_v2/train")
output_base = Path("ml/dataset_v2_balanced")

classes = ["bird", "ship", "unknown"]

split_counts = {
    "train": 3000,
    "val": 800,
    "test": 800
}

valid_ext = (".jpg", ".jpeg", ".png")

for split in split_counts:
    for cls in classes:
        (output_base / split / cls).mkdir(parents=True, exist_ok=True)

for cls in classes:
    files = [
        f for f in (source_base / cls).iterdir()
        if f.suffix.lower() in valid_ext
    ]

    random.shuffle(files)

    needed = sum(split_counts.values())

    if len(files) < needed:
        raise ValueError(f"Not enough images for {cls}. Found {len(files)}, need {needed}")

    start = 0

    for split, count in split_counts.items():
        selected = files[start:start + count]
        start += count

        for file in selected:
            dest = output_base / split / cls / file.name
            shutil.copy(file, dest)

        print(f"{cls} -> {split}: {len(selected)}")

print("Balanced dataset_v2 split created successfully.")