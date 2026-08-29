from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)

from backend.main import (
    classify_radar_image_path,
)


DATA_DIR = (
    ROOT
    / "ml"
    / "dataset_v4_live10"
)

OUT_DIR = (
    ROOT
    / "ml"
    / "evaluation"
    / "radar_live10"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


classes = [
    "bird",
    "ship",
]

records = []


print()
print("=" * 75)
print("OCEANIQ RADAR — LIVE SIMULATION 10-IMAGE EVALUATION")
print("=" * 75)


for ground_truth in classes:

    directory = (
        DATA_DIR
        / ground_truth
    )

    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        }
    )

    for image_path in images:

        result = (
            classify_radar_image_path(
                image_path,
                display_name=image_path.name,
            )
        )

        model_version = (
            result.get(
                "model_version"
            )
        )

        if (
            model_version
            != "radar_v4_raw_mobilenetv3"
        ):
            raise RuntimeError(
                "Live10 evaluation did not "
                "use the Radar V4 model. "
                f"Received model_version="
                f"{model_version}"
            )

        binary_prediction = (
            result[
                "binary_prediction"
            ]
        )

        final_prediction = (
            result[
                "final_prediction"
            ]
        )

        confidence = float(
            result[
                "confidence"
            ]
        )

        binary_correct = (
            binary_prediction
            == ground_truth
        )

        final_correct = (
            final_prediction
            == ground_truth
        )

        accepted = (
            final_prediction
            != "unknown"
        )

        record = {
            "filename":
                image_path.name,

            "ground_truth":
                ground_truth,

            "binary_prediction":
                binary_prediction,

            "final_prediction":
                final_prediction,

            "confidence_percent":
                confidence,

            "bird_probability_percent":
                result[
                    "bird_probability"
                ],

            "ship_probability_percent":
                result[
                    "ship_probability"
                ],

            "binary_correct":
                binary_correct,

            "final_correct":
                final_correct,

            "accepted":
                accepted,
        }

        records.append(
            record
        )

        status = (
            "CORRECT"
            if final_correct
            else "WRONG/UNKNOWN"
        )

        print(
            f"{ground_truth.upper():4} | "
            f"{final_prediction.upper():7} | "
            f"{confidence:6.2f}% | "
            f"{status} | "
            f"{image_path.name}"
        )


total = len(records)

if total != 10:
    raise RuntimeError(
        f"Expected 10 images, found {total}."
    )


binary_correct_count = sum(
    int(
        item[
            "binary_correct"
        ]
    )
    for item in records
)

final_correct_count = sum(
    int(
        item[
            "final_correct"
        ]
    )
    for item in records
)

accepted_count = sum(
    int(
        item[
            "accepted"
        ]
    )
    for item in records
)

unknown_count = (
    total
    - accepted_count
)


binary_accuracy = (
    binary_correct_count
    / total
)

final_accuracy = (
    final_correct_count
    / total
)

coverage = (
    accepted_count
    / total
)


if accepted_count:

    accepted_correct = sum(
        int(
            item[
                "final_correct"
            ]
        )
        for item in records
        if item[
            "accepted"
        ]
    )

    accepted_accuracy = (
        accepted_correct
        / accepted_count
    )

else:
    accepted_accuracy = 0.0


mean_confidence = sum(
    item[
        "confidence_percent"
    ]
    for item in records
) / total


bird_records = [
    item
    for item in records
    if item[
        "ground_truth"
    ] == "bird"
]

ship_records = [
    item
    for item in records
    if item[
        "ground_truth"
    ] == "ship"
]


bird_accuracy = (
    sum(
        int(
            item[
                "final_correct"
            ]
        )
        for item in bird_records
    )
    / len(
        bird_records
    )
)

ship_accuracy = (
    sum(
        int(
            item[
                "final_correct"
            ]
        )
        for item in ship_records
    )
    / len(
        ship_records
    )
)


print()
print("=" * 75)
print("LIVE SIMULATION 10-IMAGE RESULTS")
print("=" * 75)

print(
    f"Images evaluated       : {total}"
)

print(
    f"Bird images            : "
    f"{len(bird_records)}"
)

print(
    f"Ship images            : "
    f"{len(ship_records)}"
)

print()
print(
    f"Binary accuracy        : "
    f"{binary_accuracy * 100:.2f}%"
)

print(
    f"Final decision accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Bird final accuracy    : "
    f"{bird_accuracy * 100:.2f}%"
)

print(
    f"Ship final accuracy    : "
    f"{ship_accuracy * 100:.2f}%"
)

print()
print(
    f"Accepted classifications: "
    f"{accepted_count}/{total}"
)

print(
    f"Marked unknown           : "
    f"{unknown_count}/{total}"
)

print(
    f"Coverage                 : "
    f"{coverage * 100:.2f}%"
)

print(
    f"Accepted accuracy        : "
    f"{accepted_accuracy * 100:.2f}%"
)

print(
    f"Mean confidence          : "
    f"{mean_confidence:.2f}%"
)


metrics = {
    "evaluation":
        "Radar Live Simulation 10-image integration test",

    "source":
        "dataset_v4_raw held-out test split",

    "samples":
        total,

    "bird_samples":
        len(
            bird_records
        ),

    "ship_samples":
        len(
            ship_records
        ),

    "binary_accuracy":
        binary_accuracy,

    "final_decision_accuracy":
        final_accuracy,

    "bird_final_accuracy":
        bird_accuracy,

    "ship_final_accuracy":
        ship_accuracy,

    "unknown_threshold":
        0.85,

    "accepted":
        accepted_count,

    "unknown":
        unknown_count,

    "coverage":
        coverage,

    "accepted_accuracy":
        accepted_accuracy,

    "mean_confidence_percent":
        mean_confidence,

    "results":
        records,
}


json_path = (
    OUT_DIR
    / "live10_metrics.json"
)

json_path.write_text(
    json.dumps(
        metrics,
        indent=2,
    )
)


csv_path = (
    OUT_DIR
    / "live10_results.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            records[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        records
    )


print()
print(
    "Metrics:",
    json_path,
)

print(
    "Per-image results:",
    csv_path,
)

print("=" * 75)
