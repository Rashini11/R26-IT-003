import os

dataset_path = "data/raw/images"

classes = [
    "biofouling",
    "corrosion",
    "cracks",
    "paint_damage"
]

for cls in classes:

    folder = os.path.join(dataset_path, cls)

    images = sorted(os.listdir(folder))

    count = 1

    for img in images:

        ext = os.path.splitext(img)[1]

        new_name = f"{cls}_{count:03d}{ext}"

        old_path = os.path.join(folder, img)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

        count += 1

print("Renaming Complete!")