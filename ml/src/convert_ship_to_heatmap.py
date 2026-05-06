from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

source_dir = Path("ml/dataset/train/ship")
output_dir = Path("ml/dataset_v2/train/ship")

output_dir.mkdir(parents=True, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png")
count = 0

for img_path in source_dir.iterdir():
    if img_path.suffix.lower() not in valid_ext:
        continue

    img = Image.open(img_path).convert("L")
    img = img.resize((128, 128))

    save_path = output_dir / f"ship_heatmap_{count}.png"

    plt.imshow(img, cmap="viridis")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    count += 1

print(f"Generated ship heatmap images: {count}")