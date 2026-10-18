from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import random
import re
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BIRD_RAW_DIR = Path(
    "/Users/dewna/Documents/bird dataset original best"
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
    / "dataset_v4_raw"
)

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ------------------------------------------------------------
# Bird source-group split
#
# Validation = 110 unique bird images
# Test       = 111 unique bird images
# Remaining numeric groups + shapes-* = training
# ------------------------------------------------------------

BIRD_VAL_GROUPS = {
    "02",
    "10",
    "11",
    "14",
    "19",
    "21",
    "33",
    "41",
}

BIRD_TEST_GROUPS = {
    "01",
    "08",
    "09",
    "12",
    "13",
    "20",
    "32",
    "40",
}

TARGET_COUNTS = {
    "train": 519,
    "val": 110,
    "test": 111,
}


random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_file(paths: list[Path]) -> Path:
    """
    When exact duplicates exist, prefer the original filename
    rather than a filename containing 'Copy'.
    """

    return sorted(
        paths,
        key=lambda path: (
            "copy" in path.stem.lower(),
            path.name.lower(),
        ),
    )[0]


def prepare_output():
    if OUTPUT_DIR.exists():
        print(
            "Removing existing dataset_v4_raw..."
        )
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for class_name in ["bird", "ship"]:
            (
                OUTPUT_DIR
                / split
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# BIRD DATASET
# ============================================================

def prepare_birds():
    print()
    print("=" * 70)
    print("PREPARING NEW RAW BIRD RADAR DATASET")
    print("=" * 70)

    files = [
        path
        for path in BIRD_RAW_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    print(
        "Raw bird files:",
        len(files),
    )

    # --------------------------------------------------------
    # Exact deduplication across the complete bird dataset
    # --------------------------------------------------------

    by_hash = defaultdict(list)

    for index, path in enumerate(files, 1):
        digest = sha256_file(path)

        by_hash[digest].append(path)

        if index % 100 == 0:
            print(
                f"Hashed bird images: "
                f"{index}/{len(files)}"
            )

    unique_records = []

    for digest, duplicates in by_hash.items():

        selected = canonical_file(
            duplicates
        )

        match = re.match(
            r"^(\d+)-",
            selected.name,
        )

        if match:
            source_group = match.group(1)
        else:
            source_group = "shapes"

        unique_records.append(
            {
                "path": selected,
                "hash": digest,
                "source_group": source_group,
                "duplicate_count": len(
                    duplicates
                ),
            }
        )

    print(
        "Unique bird images:",
        len(unique_records),
    )

    print(
        "Duplicates removed:",
        len(files) - len(unique_records),
    )

    # --------------------------------------------------------
    # Split by SOURCE GROUP
    # --------------------------------------------------------

    split_records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for record in unique_records:

        group = record[
            "source_group"
        ]

        if group == "shapes":
            # No reliable source identifier.
            # Genuine radar screenshots, but training-only.
            split = "train"

        elif group in BIRD_VAL_GROUPS:
            split = "val"

        elif group in BIRD_TEST_GROUPS:
            split = "test"

        else:
            split = "train"

        split_records[
            split
        ].append(record)

    print()
    print("Bird split")

    for split in [
        "train",
        "val",
        "test",
    ]:
        print(
            f"{split:5}: "
            f"{len(split_records[split])}"
        )

    expected = TARGET_COUNTS

    for split, count in expected.items():

        actual = len(
            split_records[split]
        )

        if actual != count:
            raise RuntimeError(
                f"Bird {split} count mismatch. "
                f"Expected={count}, "
                f"actual={actual}"
            )

    return split_records


# ============================================================
# SHIP DATASET
# ============================================================

def prepare_ships():
    print()
    print("=" * 70)
    print("PREPARING RAW SHIP RADAR DATASET")
    print("=" * 70)

    files = [
        path
        for path in SHIP_RAW_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    groups = defaultdict(list)

    unmatched = []

    for path in files:

        match = re.match(
            r"^(P\d+)_",
            path.name,
        )

        if match:
            groups[
                match.group(1)
            ].append(path)
        else:
            unmatched.append(path)

    print(
        "Total ship files:",
        len(files),
    )

    print(
        "Grouped ship files:",
        sum(
            len(paths)
            for paths
            in groups.values()
        ),
    )

    print(
        "P source groups:",
        len(groups),
    )

    print(
        "Excluded unmatched files:",
        len(unmatched),
    )

    # --------------------------------------------------------
    # We have far more ship images than required.
    #
    # Shuffle WHOLE source groups, then assign distinct groups
    # to test, val and train until each split has enough
    # capacity.
    #
    # Images from one Pxxxx source can never appear in two
    # different splits.
    # --------------------------------------------------------

    group_names = sorted(
        groups.keys()
    )

    random.Random(
        SEED
    ).shuffle(
        group_names
    )

    remaining = list(
        group_names
    )

    assigned_groups = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Assign test first, then validation,
    # then training.
    for split in [
        "test",
        "val",
        "train",
    ]:

        target = TARGET_COUNTS[
            split
        ]

        capacity = 0

        while (
            capacity < target
            and remaining
        ):

            group = remaining.pop(0)

            assigned_groups[
                split
            ].append(group)

            capacity += len(
                groups[group]
            )

        if capacity < target:
            raise RuntimeError(
                "Not enough grouped ship "
                f"images for {split}. "
                f"Required={target}, "
                f"capacity={capacity}"
            )

    # --------------------------------------------------------
    # Select the exact number required from each isolated
    # source-group pool.
    # --------------------------------------------------------

    split_records = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        selected_groups = (
            assigned_groups[
                split
            ]
        )

        candidates = []

        for group in selected_groups:

            for path in groups[
                group
            ]:

                candidates.append(
                    {
                        "path": path,
                        "source_group": group,
                    }
                )

        rng = random.Random(
            SEED
            + {
                "train": 1,
                "val": 2,
                "test": 3,
            }[split]
        )

        rng.shuffle(
            candidates
        )

        selected = candidates[
            : TARGET_COUNTS[split]
        ]

        split_records[
            split
        ] = selected

    print()
    print("Ship split")

    for split in [
        "train",
        "val",
        "test",
    ]:

        groups_used = sorted(
            set(
                record[
                    "source_group"
                ]
                for record
                in split_records[
                    split
                ]
            )
        )

        print(
            f"{split:5}: "
            f"{len(split_records[split])} "
            f"images from "
            f"{len(groups_used)} groups"
        )

        print(
            "       ",
            ", ".join(
                groups_used
            ),
        )

    # Verify zero group overlap.

    train_groups = set(
        assigned_groups[
            "train"
        ]
    )

    val_groups = set(
        assigned_groups[
            "val"
        ]
    )

    test_groups = set(
        assigned_groups[
            "test"
        ]
    )

    if (
        train_groups & val_groups
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise RuntimeError(
            "Ship source-group leakage detected."
        )

    return (
        split_records,
        assigned_groups,
        unmatched,
    )


# ============================================================
# WRITE DATASET
# ============================================================

def copy_dataset(
    bird_records,
    ship_records,
):
    manifest = []

    print()
    print("=" * 70)
    print("COPYING RAW V4 DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Bird
    # --------------------------------------------------------

    for split, records in bird_records.items():

        for index, record in enumerate(
            sorted(
                records,
                key=lambda r:
                r["path"].name.lower(),
            ),
            1,
        ):

            source = record[
                "path"
            ]

            destination = (
                OUTPUT_DIR
                / split
                / "bird"
                / source.name
            )

            shutil.copy2(
                source,
                destination,
            )

            manifest.append(
                {
                    "split": split,
                    "class": "bird",
                    "source_group":
                        record[
                            "source_group"
                        ],
                    "source_path":
                        str(source),
                    "destination":
                        str(destination),
                    "sha256":
                        record["hash"],
                    "note":
                        (
                            "deduplicated"
                            if record[
                                "duplicate_count"
                            ] > 1
                            else "unique-original"
                        ),
                }
            )

    # --------------------------------------------------------
    # Ship
    # --------------------------------------------------------

    for split, records in ship_records.items():

        for record in records:

            source = record[
                "path"
            ]

            destination = (
                OUTPUT_DIR
                / split
                / "ship"
                / source.name
            )

            shutil.copy2(
                source,
                destination,
            )

            manifest.append(
                {
                    "split": split,
                    "class": "ship",
                    "source_group":
                        record[
                            "source_group"
                        ],
                    "source_path":
                        str(source),
                    "destination":
                        str(destination),
                    "sha256":
                        sha256_file(source),
                    "note":
                        "raw-grouped-ship",
                }
            )

    return manifest


# ============================================================
# VERIFY OUTPUT
# ============================================================

def verify_dataset():
    print()
    print("=" * 70)
    print("VERIFYING DATASET")
    print("=" * 70)

    hashes = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    final_counts = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        final_counts[
            split
        ] = {}

        for class_name in [
            "bird",
            "ship",
        ]:

            directory = (
                OUTPUT_DIR
                / split
                / class_name
            )

            files = [
                path
                for path
                in directory.iterdir()
                if path.is_file()
            ]

            final_counts[
                split
            ][
                class_name
            ] = len(files)

            for path in files:
                hashes[
                    split
                ].add(
                    sha256_file(
                        path
                    )
                )

            print(
                f"{split:5} "
                f"{class_name:5}: "
                f"{len(files)}"
            )

    train_val = (
        hashes["train"]
        & hashes["val"]
    )

    train_test = (
        hashes["train"]
        & hashes["test"]
    )

    val_test = (
        hashes["val"]
        & hashes["test"]
    )

    print()
    print(
        "Exact hash overlap "
        "train/val:",
        len(train_val),
    )

    print(
        "Exact hash overlap "
        "train/test:",
        len(train_test),
    )

    print(
        "Exact hash overlap "
        "val/test:",
        len(val_test),
    )

    if (
        train_val
        or train_test
        or val_test
    ):
        raise RuntimeError(
            "Cross-split exact duplicate "
            "leakage detected."
        )

    return final_counts


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("OCEANIQ RADAR V4 RAW DATASET BUILDER")
    print("=" * 70)

    if not BIRD_RAW_DIR.exists():
        raise FileNotFoundError(
            BIRD_RAW_DIR
        )

    if not SHIP_RAW_DIR.exists():
        raise FileNotFoundError(
            SHIP_RAW_DIR
        )

    prepare_output()

    bird_records = prepare_birds()

    (
        ship_records,
        ship_group_assignments,
        unmatched_ship,
    ) = prepare_ships()

    manifest = copy_dataset(
        bird_records,
        ship_records,
    )

    final_counts = verify_dataset()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_path = (
        OUTPUT_DIR
        / "dataset_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "class",
                "source_group",
                "source_path",
                "destination",
                "sha256",
                "note",
            ],
        )

        writer.writeheader()
        writer.writerows(
            manifest
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "dataset_version":
            "radar_v4_raw",
        "seed": SEED,
        "heatmap_preprocessing":
            False,
        "classes": [
            "bird",
            "ship",
        ],
        "unknown_strategy":
            (
                "confidence-threshold "
                "during inference"
            ),
        "bird": {
            "raw_files": 1002,
            "unique_images": 740,
            "duplicates_removed": 262,
            "shapes_images":
                "training-only",
            "validation_groups":
                sorted(
                    BIRD_VAL_GROUPS
                ),
            "test_groups":
                sorted(
                    BIRD_TEST_GROUPS
                ),
        },
        "ship": {
            "raw_files": 4717,
            "grouped_files": 3923,
            "source_groups": 137,
            "unmatched_excluded":
                len(
                    unmatched_ship
                ),
            "assigned_groups":
                ship_group_assignments,
        },
        "counts": final_counts,
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("RADAR V4 RAW DATASET READY")
    print("=" * 70)

    print(
        "Dataset:",
        OUTPUT_DIR,
    )

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print()
    print(
        "NO HEATMAPS WERE CREATED."
    )

    print(
        "OLD BIRD DATA WAS NOT USED."
    )

    print(
        "UNMATCHED SHIP FILES "
        "WERE NOT USED."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
