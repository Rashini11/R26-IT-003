from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import random
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]

V4_MANIFEST = ROOT / "ml" / "dataset_v4_raw" / "dataset_manifest.csv"
V4_SUMMARY = ROOT / "ml" / "dataset_v4_raw" / "summary.json"
SHIP_RAW = ROOT / "ml" / "dataset" / "train" / "ship"

OUT = ROOT / "ml" / "dataset_v5_final"

SEED = 2026

BIRD_VAL = {"05", "07", "18", "23"}
BIRD_TEST = {"03", "04", "16", "17", "22"}

TARGET = {
    "train": 514,
    "val": 112,
    "test": 114,
}

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)

for split in TARGET:
    for cls in ["bird", "ship"]:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)


# ==========================================================
# BIRDS
# ==========================================================

bird_records = []

with V4_MANIFEST.open() as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["class"] != "bird":
            continue

        group = row["source_group"]

        if group in BIRD_VAL:
            split = "val"
        elif group in BIRD_TEST:
            split = "test"
        else:
            split = "train"

        bird_records.append({
            "split": split,
            "group": group,
            "source": Path(row["source_path"]),
        })


for split in TARGET:
    rows = [
        r for r in bird_records
        if r["split"] == split
    ]

    print(
        f"Bird {split}:",
        len(rows)
    )

    if len(rows) != TARGET[split]:
        raise RuntimeError(
            f"Unexpected bird count for {split}: {len(rows)}"
        )

    for row in rows:
        shutil.copy2(
            row["source"],
            OUT / split / "bird" / row["source"].name,
        )


# ==========================================================
# SHIPS — ONLY GROUPS UNUSED IN V4
# ==========================================================

v4_summary = json.loads(
    V4_SUMMARY.read_text()
)

used_v4_groups = set()

for groups in (
    v4_summary["ship"]["assigned_groups"].values()
):
    used_v4_groups.update(groups)


ship_groups = defaultdict(list)

for path in SHIP_RAW.iterdir():

    if (
        not path.is_file()
        or path.suffix.lower() not in EXTS
    ):
        continue

    m = re.match(r"^(P\d+)_", path.name)

    if not m:
        continue

    group = m.group(1)

    if group in used_v4_groups:
        continue

    ship_groups[group].append(path)


available_groups = sorted(ship_groups)

print(
    "Ship groups unused in V4:",
    len(available_groups)
)

rng = random.Random(SEED)
rng.shuffle(available_groups)

remaining = list(available_groups)

assigned = {
    "train": [],
    "val": [],
    "test": [],
}


# Allocate whole groups to each split.
for split in ["test", "val", "train"]:

    capacity = 0

    while (
        capacity < TARGET[split]
        and remaining
    ):
        group = remaining.pop(0)

        assigned[split].append(group)

        capacity += len(
            ship_groups[group]
        )

    if capacity < TARGET[split]:
        raise RuntimeError(
            f"Insufficient ship capacity for {split}"
        )


ship_records = []

for split in ["train", "val", "test"]:

    candidates = []

    for group in assigned[split]:

        for path in ship_groups[group]:

            candidates.append({
                "group": group,
                "path": path,
            })

    split_rng = random.Random(
        SEED + {
            "train": 1,
            "val": 2,
            "test": 3,
        }[split]
    )

    split_rng.shuffle(candidates)

    selected = candidates[
        :TARGET[split]
    ]

    print(
        f"Ship {split}:",
        len(selected),
        "from",
        len(assigned[split]),
        "groups"
    )

    for item in selected:

        shutil.copy2(
            item["path"],
            OUT / split / "ship" / item["path"].name,
        )

        ship_records.append({
            "split": split,
            "group": item["group"],
            "source": str(item["path"]),
        })


# ==========================================================
# CROSS-SPLIT DUPLICATE CHECK
# ==========================================================

hashes = {}

for split in ["train", "val", "test"]:

    hashes[split] = set()

    for cls in ["bird", "ship"]:

        for path in (
            OUT / split / cls
        ).iterdir():

            if path.is_file():
                hashes[split].add(
                    file_hash(path)
                )


for a, b in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]:

    overlap = hashes[a] & hashes[b]

    print(
        f"Hash overlap {a}/{b}:",
        len(overlap)
    )

    if overlap:
        raise RuntimeError(
            "Cross-split duplicate leakage detected"
        )


summary = {
    "dataset": "radar_v5_final",
    "seed": SEED,
    "bird_val_groups": sorted(BIRD_VAL),
    "bird_test_groups": sorted(BIRD_TEST),
    "ship_v4_groups_excluded": sorted(used_v4_groups),
    "ship_groups": assigned,
    "counts": TARGET,
}

(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print()
print("=" * 60)
print("RADAR V5 FINAL DATASET READY")
print("=" * 60)

for split in ["train", "val", "test"]:

    print(
        split,
        "bird =",
        len(list(
            (OUT / split / "bird").iterdir()
        )),
        "ship =",
        len(list(
            (OUT / split / "ship").iterdir()
        )),
    )

print("=" * 60)
