from pathlib import Path
import csv
import json
import random
import shutil

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "ml"
    / "dataset_v4_raw"
    / "test"
)

DEST = (
    ROOT
    / "ml"
    / "dataset_v4_live10"
)

SEED = 42
COUNT_PER_CLASS = 5

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


if DEST.exists():
    shutil.rmtree(DEST)

for class_name in [
    "bird",
    "ship",
]:
    (
        DEST
        / class_name
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


manifest = []

for class_index, class_name in enumerate(
    ["bird", "ship"]
):

    source_dir = (
        SOURCE
        / class_name
    )

    candidates = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in EXTENSIONS
    )

    if len(candidates) < COUNT_PER_CLASS:
        raise RuntimeError(
            f"Not enough {class_name} images."
        )

    # Separate deterministic seed for each class.
    rng = random.Random(
        SEED + class_index
    )

    selected = rng.sample(
        candidates,
        COUNT_PER_CLASS,
    )

    selected = sorted(
        selected,
        key=lambda p: p.name,
    )

    for number, source in enumerate(
        selected,
        1,
    ):

        destination = (
            DEST
            / class_name
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        manifest.append(
            {
                "class":
                    class_name,
                "number":
                    number,
                "filename":
                    source.name,
                "source":
                    str(source),
                "destination":
                    str(destination),
            }
        )


csv_path = (
    DEST
    / "manifest.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "class",
            "number",
            "filename",
            "source",
            "destination",
        ],
    )

    writer.writeheader()
    writer.writerows(
        manifest
    )


summary = {
    "dataset":
        "radar_v4_live10",
    "source":
        "dataset_v4_raw/test",
    "seed":
        SEED,
    "total_images":
        10,
    "bird_images":
        5,
    "ship_images":
        5,
    "purpose":
        (
            "Fixed held-out radar-image "
            "sequence for Live Maritime Simulation"
        ),
    "images":
        manifest,
}


(
    DEST
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print("=" * 65)
print("RADAR LIVE SIMULATION 10-IMAGE DATASET")
print("=" * 65)

for item in manifest:
    print(
        f"{item['class']:4} | "
        f"{item['filename']}"
    )

print()
print("Bird :", 5)
print("Ship :", 5)
print("Total:", 10)

print()
print("Created:", DEST)
print("Manifest:", csv_path)

print("=" * 65)
