from pathlib import Path

base = Path("ml/dataset_v2")

classes = ["bird", "ship", "unknown"]
splits = ["train", "val", "test"]

for split in splits:
    for cls in classes:
        folder = base / split / cls
        folder.mkdir(parents=True, exist_ok=True)

print("dataset_v2 folder structure created successfully.")