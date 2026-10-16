from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageOps
import csv
import hashlib
import random
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml/chatbot"
OUT = DATA / "_router_v1"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

sources = {
    "hull": DATA / "_curated/hull",
    "unsupported": DATA / "_curated/unsupported",
}

# Match source folders while tolerating trailing spaces.
folder_labels = {
    "Seastate Dataset": "sea",
    "drone based vessel images": "vessel",
    "radar object classification": "radar",
}
for folder in DATA.iterdir():
    label = folder_labels.get(folder.name.strip())
    if folder.is_dir() and label:
        sources[label] = folder

def source_group(label, filename):
    name = filename.lower()

    # Remove the hash prefix added to curated filenames.
    name = re.sub(r"^[0-9a-f]{12}_", "", name)

    # Remove Roboflow version IDs and simple copy suffixes.
    name = name.split(".rf.")[0]
    name = re.sub(r" - copy(?: \(\d+\))?", "", name)
    name = re.sub(
        r"(?:_jpg|_jpeg|_png)?\.(?:jpg|jpeg|png|webp|bmp)$",
        "", name
    )
    name = re.sub(r"_(?:jpg|jpeg|png)$", "", name)

    if label == "sea":
        return name.split("_fa_")[0]

    if label == "vessel":
        # Keep different views/versions of a named vessel together.
        return re.sub(r"[-_]\d+$", "", name)

    if label == "radar":
        match = re.match(r"(\d{2})-\d+", name)
        if match:
            return "sequence_" + match.group(1)
        if re.match(r"^\d{6}", name):
            return "numeric_frames_source_unverified"

    if label == "hull":
        # Conservatively keep unidentified video frames together.
        if name.startswith("frame_"):
            return "video_frames_source_unverified"
        for family in ("cracks", "corrosion", "biofouling", "paint_damage"):
            if name.startswith(family):
                return family + "_source_unverified"

    if label == "unsupported":
        # Conservative grouping of visibly related example families.
        for token, family in (
            ("leaf", "trees"),
            ("standing", "people"),
            ("000000", "aircraft"),
        ):
            if token in name:
                return family

    return name

rows = []
seen = {}
duplicates = 0

for label in ("radar", "vessel", "sea", "hull", "unsupported"):
    folder = sources.get(label)
    if folder is None or not folder.exists():
        raise SystemExit(f"Missing source folder for {label}")

    for path in sorted(folder.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in EXTENSIONS
            or path.name.startswith("._")
            or "__MACOSX" in path.parts
        ):
            continue

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            digest = hashlib.sha256(
                str(image.size).encode() + image.tobytes()
            ).hexdigest()

        if digest in seen:
            previous = seen[digest]
            if previous != label:
                raise SystemExit(
                    f"Conflicting labels for identical image: "
                    f"{previous} and {label}: {path}"
                )
            duplicates += 1
            continue

        seen[digest] = label
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "label": label,
            "group": label + ":" + source_group(label, path.name),
            "pixel_hash": digest,
            "split": "",
        })

# Split whole groups, never individual images within a group.
# Approximately 60/20/20 by group count, not necessarily image count.
rng = random.Random(42)

for label in sources:
    groups = sorted({r["group"] for r in rows if r["label"] == label})
    if len(groups) < 3:
        raise SystemExit(
            f"{label} has only {len(groups)} candidate groups. "
            "Need more independent sources before a three-way split."
        )

    rng.shuffle(groups)
    holdout = max(1, round(len(groups) * 0.20))
    val_groups = set(groups[:holdout])
    test_groups = set(groups[holdout:2 * holdout])

    for row in rows:
        if row["label"] == label:
            row["split"] = (
                "val" if row["group"] in val_groups
                else "test" if row["group"] in test_groups
                else "train"
            )

group_splits = defaultdict(set)
for row in rows:
    group_splits[row["group"]].add(row["split"])
assert all(len(splits) == 1 for splits in group_splits.values())

OUT.mkdir(exist_ok=True)
manifest = OUT / "manifest.csv"

# Avoid silently changing an existing experimental split.
if manifest.exists():
    raise SystemExit(
        f"Manifest already exists: {manifest}\n"
        "Keep it unchanged; share its current summary instead."
    )

with manifest.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["path", "label", "group", "pixel_hash", "split"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nPROVISIONAL ROUTER SPLIT")
print(f"{'Category':<14} {'Train':>7} {'Val':>7} {'Test':>7} {'Groups':>8}")

for label in ("radar", "vessel", "sea", "hull", "unsupported"):
    selected = [r for r in rows if r["label"] == label]
    counts = Counter(r["split"] for r in selected)
    groups = {r["group"] for r in selected}
    print(
        f"{label:<14} {counts['train']:>7} "
        f"{counts['val']:>7} {counts['test']:>7} {len(groups):>8}"
    )

print("\nDuplicate images skipped:", duplicates)
print("Manifest:", manifest)
print("\nSOURCE GROUPS")
for group in sorted(group_splits):
    members = [r for r in rows if r["group"] == group]
    print(f"{members[0]['split']:<6} {len(members):>3}  {group}")

print("\nFilename groups are provisional, not verified source identities.")
print("No training or evaluation was performed.")
