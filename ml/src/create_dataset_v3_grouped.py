from pathlib import Path
import csv
import json
import random
import re
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BIRD_RAW_DIR = (
    PROJECT_ROOT
    / "ml"
    / "dataset"
    / "bird_radar_data_20211008_20211014"
)

SHIP_RAW_DIR = (
    PROJECT_ROOT
    / "ml"
    / "dataset"
    / "train"
    / "ship"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "dataset_v3_grouped"
)

SEED = 42
IMAGE_SIZE = 128
BIRD_WINDOW_SIZE = 300

TARGET_COUNTS = {
    "train": 2500,
    "val": 600,
    "test": 600,
}

BIRD_COLUMNS = [
    "距离",
    "航速",
    "信噪比",
    "高度",
]

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CLEAN OUTPUT
# ============================================================

if OUTPUT_DIR.exists():
    print("Removing existing dataset_v3_grouped...")
    shutil.rmtree(OUTPUT_DIR)

for split in TARGET_COUNTS:
    for class_name in ["bird", "ship", "unknown"]:
        (
            OUTPUT_DIR
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


manifest_rows = []


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalise_to_uint8(array):
    array = np.asarray(
        array,
        dtype=np.float32,
    )

    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))

    if maximum <= minimum:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    normalised = (
        (array - minimum)
        / (maximum - minimum)
    )

    return np.clip(
        normalised * 255.0,
        0,
        255,
    ).astype(np.uint8)


def grayscale_to_heatmap(gray):
    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_VIRIDIS,
    )

    return heatmap


def save_ship_heatmap(source_path, destination):
    image = cv2.imread(
        str(source_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read ship image: {source_path}"
        )

    heatmap = grayscale_to_heatmap(image)

    cv2.imwrite(
        str(destination),
        heatmap,
    )


def get_ship_group(path):
    match = re.match(
        r"(P\d+)_",
        path.name,
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# BIRD SOURCE SPLIT
#
# TRAIN : 8-12 October
# VAL   : 13 October
# TEST  : 14 October
# ============================================================

bird_files = sorted(
    BIRD_RAW_DIR.glob("*.txt")
)

bird_files_by_split = {
    "train": [],
    "val": [],
    "test": [],
}

for path in bird_files:
    name = path.name

    if any(
        name.startswith(f"2021-10-{day:02d}")
        for day in range(8, 13)
    ):
        bird_files_by_split["train"].append(path)

    elif name.startswith("2021-10-13"):
        bird_files_by_split["val"].append(path)

    elif name.startswith("2021-10-14"):
        bird_files_by_split["test"].append(path)


print("\nBird source recordings")

for split, files in bird_files_by_split.items():
    print(
        f"{split:5}: "
        f"{len(files)} recording files"
    )

    for path in files:
        print("  ", path.name)


# ============================================================
# BIRD HELPERS
# ============================================================

def load_bird_features(path):
    dataframe = pd.read_csv(
        path,
        sep="\t",
        encoding="gbk",
        usecols=BIRD_COLUMNS,
    )

    dataframe = dataframe.dropna(
        subset=BIRD_COLUMNS
    )

    return dataframe[BIRD_COLUMNS]


def get_bird_capacities(paths):
    capacities = {}

    for path in paths:
        print(
            "Scanning bird recording:",
            path.name,
        )

        features = load_bird_features(path)

        capacities[path] = (
            len(features)
            // BIRD_WINDOW_SIZE
        )

        print(
            "  Valid chunks:",
            capacities[path],
        )

    return capacities


def allocate_bird_quota(
    capacities,
    target_count,
):
    total_capacity = sum(
        capacities.values()
    )

    if total_capacity < target_count:
        raise RuntimeError(
            "Not enough bird chunks. "
            f"Available={total_capacity}, "
            f"required={target_count}"
        )

    quota = {
        path: 0
        for path in capacities
    }

    # Distribute samples approximately
    # proportional to each recording size.
    for _ in range(target_count):
        available = [
            path
            for path, capacity
            in capacities.items()
            if quota[path] < capacity
        ]

        if not available:
            raise RuntimeError(
                "Bird quota allocation failed."
            )

        selected = max(
            available,
            key=lambda path: (
                capacities[path]
                / (quota[path] + 1)
            ),
        )

        quota[selected] += 1

    return quota


def bird_chunk_to_heatmap(chunk):
    values = chunk.to_numpy(
        dtype=np.float32
    )

    minimum = np.nanmin(
        values,
        axis=0,
        keepdims=True,
    )

    maximum = np.nanmax(
        values,
        axis=0,
        keepdims=True,
    )

    denominator = maximum - minimum

    denominator[
        denominator < 1e-8
    ] = 1.0

    normalised = (
        values - minimum
    ) / denominator

    # Same conceptual orientation as:
    # plt.imshow(chunk.T, aspect="auto")
    image_matrix = (
        normalised.T * 255.0
    ).clip(
        0,
        255,
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        image_matrix,
        cv2.COLORMAP_VIRIDIS,
    )

    heatmap = cv2.resize(
        heatmap,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    return heatmap


# ============================================================
# GENERATE BIRDS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING GROUPED BIRD DATA")
print("=" * 70)

for split in ["train", "val", "test"]:

    source_files = (
        bird_files_by_split[split]
    )

    target = TARGET_COUNTS[split]

    capacities = get_bird_capacities(
        source_files
    )

    quotas = allocate_bird_quota(
        capacities,
        target,
    )

    rng = random.Random(
        SEED
        + {
            "train": 1,
            "val": 2,
            "test": 3,
        }[split]
    )

    generated = 0

    for source_path in source_files:

        quota = quotas[source_path]

        if quota == 0:
            continue

        features = load_bird_features(
            source_path
        )

        total_chunks = (
            len(features)
            // BIRD_WINDOW_SIZE
        )

        selected_indices = sorted(
            rng.sample(
                range(total_chunks),
                quota,
            )
        )

        for chunk_index in selected_indices:

            start = (
                chunk_index
                * BIRD_WINDOW_SIZE
            )

            end = (
                start
                + BIRD_WINDOW_SIZE
            )

            chunk = features.iloc[
                start:end
            ]

            heatmap = bird_chunk_to_heatmap(
                chunk
            )

            output_name = (
                f"bird_"
                f"{source_path.stem}_"
                f"chunk_{chunk_index:06d}.png"
            )

            destination = (
                OUTPUT_DIR
                / split
                / "bird"
                / output_name
            )

            cv2.imwrite(
                str(destination),
                heatmap,
            )

            manifest_rows.append({
                "split": split,
                "class": "bird",
                "output_file": str(
                    destination.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "source_group": (
                    source_path.stem
                ),
                "source_file": (
                    source_path.name
                ),
                "source_detail": (
                    f"chunk_{chunk_index}"
                ),
            })

            generated += 1

    if generated != target:
        raise RuntimeError(
            f"Bird {split}: generated "
            f"{generated}, expected {target}"
        )

    print(
        f"Bird {split}: {generated}"
    )


# ============================================================
# CLEAN SHIP SOURCES
#
# Keep only files with reliable Pxxxx groups.
# Exclude the 794 unmatched files.
# ============================================================

all_ship_files = sorted([
    path
    for path in SHIP_RAW_DIR.glob("*.jpg")
    if get_ship_group(path) is not None
])

ship_groups = np.array([
    get_ship_group(path)
    for path in all_ship_files
])

ship_files_array = np.array(
    all_ship_files,
    dtype=object,
)

print("\n" + "=" * 70)
print("SHIP SOURCE DATA")
print("=" * 70)

print(
    "Grouped ship images:",
    len(all_ship_files),
)

print(
    "Unique P-groups:",
    len(set(ship_groups)),
)


# ============================================================
# FIND GROUPED SHIP SPLIT
# ============================================================

required_total = sum(
    TARGET_COUNTS.values()
)

if len(all_ship_files) < required_total:
    raise RuntimeError(
        "Not enough grouped ship images. "
        f"Available={len(all_ship_files)}, "
        f"required={required_total}"
    )


def find_ship_split():

    temporary_fraction = (
        (
            TARGET_COUNTS["val"]
            + TARGET_COUNTS["test"]
        )
        / required_total
    )

    for attempt_seed in range(
        SEED,
        SEED + 5000,
    ):

        first_splitter = (
            GroupShuffleSplit(
                n_splits=1,
                test_size=temporary_fraction,
                random_state=attempt_seed,
            )
        )

        train_indexes, temp_indexes = next(
            first_splitter.split(
                ship_files_array,
                groups=ship_groups,
            )
        )

        temp_files = ship_files_array[
            temp_indexes
        ]

        temp_groups = ship_groups[
            temp_indexes
        ]

        second_splitter = (
            GroupShuffleSplit(
                n_splits=1,
                test_size=0.5,
                random_state=(
                    attempt_seed + 10000
                ),
            )
        )

        val_relative, test_relative = next(
            second_splitter.split(
                temp_files,
                groups=temp_groups,
            )
        )

        pools = {
            "train": list(
                ship_files_array[
                    train_indexes
                ]
            ),
            "val": list(
                temp_files[
                    val_relative
                ]
            ),
            "test": list(
                temp_files[
                    test_relative
                ]
            ),
        }

        enough = all(
            len(pools[split])
            >= TARGET_COUNTS[split]
            for split in pools
        )

        if enough:
            return (
                pools,
                attempt_seed,
            )

    raise RuntimeError(
        "Unable to create grouped ship "
        "split with required sample counts."
    )


ship_pools, ship_split_seed = (
    find_ship_split()
)

print(
    "Ship group split seed:",
    ship_split_seed,
)

for split, pool in ship_pools.items():
    groups = {
        get_ship_group(path)
        for path in pool
    }

    print(
        f"{split:5}: "
        f"{len(pool)} source images, "
        f"{len(groups)} P-groups"
    )


# ============================================================
# VERIFY SHIP GROUP ISOLATION
# ============================================================

group_sets = {
    split: {
        get_ship_group(path)
        for path in pool
    }
    for split, pool in ship_pools.items()
}

if (
    group_sets["train"]
    & group_sets["val"]
):
    raise RuntimeError(
        "Ship train/val group leakage."
    )

if (
    group_sets["train"]
    & group_sets["test"]
):
    raise RuntimeError(
        "Ship train/test group leakage."
    )

if (
    group_sets["val"]
    & group_sets["test"]
):
    raise RuntimeError(
        "Ship val/test group leakage."
    )

print(
    "Ship group leakage check: PASSED"
)


# ============================================================
# GENERATE SHIP CLASS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING GROUPED SHIP DATA")
print("=" * 70)

selected_ship_files = {}

for split, pool in ship_pools.items():

    rng = random.Random(
        SEED
        + {
            "train": 100,
            "val": 200,
            "test": 300,
        }[split]
    )

    target = TARGET_COUNTS[split]

    selected = rng.sample(
        pool,
        target,
    )

    selected_ship_files[split] = (
        selected
    )

    for index, source_path in enumerate(
        selected,
        start=1,
    ):

        group = get_ship_group(
            source_path
        )

        output_name = (
            f"ship_{group}_"
            f"{index:05d}.png"
        )

        destination = (
            OUTPUT_DIR
            / split
            / "ship"
            / output_name
        )

        save_ship_heatmap(
            source_path,
            destination,
        )

        manifest_rows.append({
            "split": split,
            "class": "ship",
            "output_file": str(
                destination.relative_to(
                    PROJECT_ROOT
                )
            ),
            "source_group": group,
            "source_file": (
                source_path.name
            ),
            "source_detail": "",
        })

    print(
        f"Ship {split}: {len(selected)}"
    )


# ============================================================
# GENERATE UNKNOWN CLASS
#
# Unknown samples are regenerated AFTER group splitting.
# Each crop remains in the SAME split as its parent P-group.
# ============================================================

print("\n" + "=" * 70)
print("GENERATING GROUPED UNKNOWN DATA")
print("=" * 70)


def create_unknown_crop(
    source_path,
    rng,
):
    image = cv2.imread(
        str(source_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return None, None

    height, width = image.shape[:2]

    crop_size = 128

    if (
        width < crop_size
        or height < crop_size
    ):
        return None, None

    x = rng.randint(
        0,
        width - crop_size,
    )

    y = rng.randint(
        0,
        height - crop_size,
    )

    crop = image[
        y:y + crop_size,
        x:x + crop_size,
    ]

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    heatmap = grayscale_to_heatmap(
        gray
    )

    return heatmap, (x, y)


for split, pool in ship_pools.items():

    target = TARGET_COUNTS[split]

    rng = random.Random(
        SEED
        + {
            "train": 1000,
            "val": 2000,
            "test": 3000,
        }[split]
    )

    candidate_parents = list(pool)
    rng.shuffle(candidate_parents)

    generated = 0

    # One unknown crop per parent first.
    # If needed, loop through again with
    # different random coordinates.
    parent_round = 0

    while generated < target:

        parent_round += 1

        for source_path in candidate_parents:

            if generated >= target:
                break

            heatmap, coordinates = (
                create_unknown_crop(
                    source_path,
                    rng,
                )
            )

            if heatmap is None:
                continue

            group = get_ship_group(
                source_path
            )

            generated += 1

            output_name = (
                f"unknown_{group}_"
                f"{generated:05d}.png"
            )

            destination = (
                OUTPUT_DIR
                / split
                / "unknown"
                / output_name
            )

            cv2.imwrite(
                str(destination),
                heatmap,
            )

            x, y = coordinates

            manifest_rows.append({
                "split": split,
                "class": "unknown",
                "output_file": str(
                    destination.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "source_group": group,
                "source_file": (
                    source_path.name
                ),
                "source_detail": (
                    f"crop_x{x}_y{y}_"
                    f"round{parent_round}"
                ),
            })

        if parent_round > 20:
            raise RuntimeError(
                f"Unable to generate "
                f"{target} unknown samples "
                f"for {split}."
            )

    print(
        f"Unknown {split}: {generated}"
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_DIR
    / "dataset_manifest.csv"
)

with manifest_path.open(
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split",
            "class",
            "output_file",
            "source_group",
            "source_file",
            "source_detail",
        ],
    )

    writer.writeheader()
    writer.writerows(
        manifest_rows
    )


# ============================================================
# FINAL COUNTS
# ============================================================

final_counts = {}

for split in TARGET_COUNTS:

    final_counts[split] = {}

    for class_name in [
        "bird",
        "ship",
        "unknown",
    ]:

        count = len(
            list(
                (
                    OUTPUT_DIR
                    / split
                    / class_name
                ).glob("*.png")
            )
        )

        final_counts[
            split
        ][class_name] = count


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "seed": SEED,

    "dataset_policy": (
        "Source/group-aware Radar dataset. "
        "Bird recordings are separated "
        "temporally. Ship P-groups are "
        "isolated between splits. Unknown "
        "crops remain in the same split "
        "as their parent ship P-group."
    ),

    "targets_per_class": (
        TARGET_COUNTS
    ),

    "final_counts": (
        final_counts
    ),

    "bird_recordings": {
        split: [
            path.name
            for path
            in paths
        ]
        for split, paths
        in bird_files_by_split.items()
    },

    "ship_group_split_seed": (
        ship_split_seed
    ),

    "ship_groups": {
        split: sorted(
            group_sets[split]
        )
        for split
        in group_sets
    },

    "ship_group_counts": {
        split: len(
            group_sets[split]
        )
        for split
        in group_sets
    },

    "excluded_unmatched_ship_files": (
        len([
            path
            for path
            in SHIP_RAW_DIR.glob("*.jpg")
            if get_ship_group(path)
            is None
        ])
    ),

    "group_overlap": {
        "train_val": len(
            group_sets["train"]
            & group_sets["val"]
        ),

        "train_test": len(
            group_sets["train"]
            & group_sets["test"]
        ),

        "val_test": len(
            group_sets["val"]
            & group_sets["test"]
        ),
    },
}


with (
    OUTPUT_DIR
    / "summary.json"
).open(
    "w"
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )


print("\n" + "=" * 70)
print("DATASET V3 GROUPED CREATED")
print("=" * 70)

for split in [
    "train",
    "val",
    "test",
]:
    print(
        f"\n{split.upper()}"
    )

    for class_name in [
        "bird",
        "ship",
        "unknown",
    ]:
        print(
            f"  {class_name:8}: "
            f"{final_counts[split][class_name]}"
        )


print(
    "\nShip group overlap:"
)

print(
    "  train/val :",
    summary[
        "group_overlap"
    ]["train_val"],
)

print(
    "  train/test:",
    summary[
        "group_overlap"
    ]["train_test"],
)

print(
    "  val/test  :",
    summary[
        "group_overlap"
    ]["val_test"],
)

print(
    "\nExcluded unmatched ships:",
    summary[
        "excluded_unmatched_ship_files"
    ],
)

print(
    "\nManifest:",
    manifest_path,
)

print(
    "Summary:",
    OUTPUT_DIR / "summary.json",
)

print("=" * 70)
