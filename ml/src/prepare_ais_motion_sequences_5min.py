from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path(
    "ml/external_datasets/ais_motion/processed_AIS_dataset.csv"
)

OUTPUT_DIR = Path("ml/ais_motion/sequences_5min")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_INTERVAL = "1min"

HISTORY_STEPS = 10
FORECAST_STEPS = 5
TOTAL_WINDOW_STEPS = HISTORY_STEPS + FORECAST_STEPS

SEQUENCE_STRIDE = 1

MAX_SOURCE_TIME_GAP_SECONDS = 300
MAX_INTERPOLATED_MINUTES = 3
MAX_REALISTIC_SPEED_KNOTS = 80

MIN_OBSERVED_HISTORY_POINTS = 6

RANDOM_SEED = 42

CLASS_NAMES = [
    "Stopped",
    "Slow",
    "Moderate",
    "Fast",
]

FEATURE_NAMES = [
    "east_position_metres",
    "north_position_metres",
    "speed_over_ground_knots",
    "sin_course",
    "cos_course",
    "observed_record_indicator",
    "course_missing_indicator",
]

REQUIRED_COLUMNS = [
    "MMSI",
    "BaseDateTime",
    "LAT",
    "LON",
    "SOG",
    "COG",
]

EARTH_RADIUS_METRES = 6_371_000.0
METRES_PER_SECOND_TO_KNOTS = 1.943844


# ============================================================
# UTILITIES
# ============================================================
def classify_speed(speed_knots):
    if speed_knots <= 2.0:
        return 0

    if speed_knots <= 6.0:
        return 1

    if speed_knots <= 12.0:
        return 2

    return 3


def local_xy_metres(
    latitudes,
    longitudes,
    reference_latitude,
    reference_longitude,
):
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)

    reference_latitude_radians = np.radians(
        reference_latitude
    )

    east = (
        EARTH_RADIUS_METRES
        * np.radians(
            longitudes - reference_longitude
        )
        * np.cos(reference_latitude_radians)
    )

    north = (
        EARTH_RADIUS_METRES
        * np.radians(
            latitudes - reference_latitude
        )
    )

    return east, north


def haversine_distance_metres(
    lat1,
    lon1,
    lat2,
    lon2,
):
    lat1 = np.radians(
        np.asarray(lat1, dtype=np.float64)
    )

    lon1 = np.radians(
        np.asarray(lon1, dtype=np.float64)
    )

    lat2 = np.radians(
        np.asarray(lat2, dtype=np.float64)
    )

    lon2 = np.radians(
        np.asarray(lon2, dtype=np.float64)
    )

    delta_latitude = lat2 - lat1
    delta_longitude = lon2 - lon1

    value = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_longitude / 2.0) ** 2
    )

    value = np.clip(value, 0.0, 1.0)

    return (
        2.0
        * EARTH_RADIUS_METRES
        * np.arcsin(np.sqrt(value))
    )


# ============================================================
# LOAD AND CLEAN
# ============================================================
print("Loading AIS dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=REQUIRED_COLUMNS,
)

print(f"Raw records: {len(df):,}")

df["BaseDateTime"] = pd.to_datetime(
    df["BaseDateTime"],
    errors="coerce",
)

for column in [
    "MMSI",
    "LAT",
    "LON",
    "SOG",
    "COG",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

valid_rows = (
    df["MMSI"].notna()
    & df["BaseDateTime"].notna()
    & df["LAT"].between(-90, 90)
    & df["LON"].between(-180, 180)
    & df["SOG"].between(
        0,
        MAX_REALISTIC_SPEED_KNOTS,
    )
)

df = df.loc[valid_rows].copy()

df["MMSI"] = df["MMSI"].astype("int64")

invalid_course = (
    df["COG"].isna()
    | (df["COG"] < 0)
    | (df["COG"] >= 360)
)

df.loc[invalid_course, "COG"] = np.nan

df = (
    df.sort_values(
        ["MMSI", "BaseDateTime"]
    )
    .drop_duplicates(
        subset=["MMSI", "BaseDateTime"],
        keep="last",
    )
    .reset_index(drop=True)
)

print(f"Clean records: {len(df):,}")
print(f"Unique vessels: {df['MMSI'].nunique():,}")


# ============================================================
# CREATE CONTINUOUS TRACK SEGMENTS
# ============================================================
grouped = df.groupby("MMSI", sort=False)

df["time_gap_seconds"] = (
    grouped["BaseDateTime"]
    .diff()
    .dt.total_seconds()
)

previous_latitude = grouped["LAT"].shift()
previous_longitude = grouped["LON"].shift()

step_distance = haversine_distance_metres(
    previous_latitude,
    previous_longitude,
    df["LAT"],
    df["LON"],
)

with np.errstate(
    divide="ignore",
    invalid="ignore",
):
    implied_speed_knots = (
        step_distance
        / df["time_gap_seconds"].to_numpy()
        * METRES_PER_SECOND_TO_KNOTS
    )

first_record = grouped.cumcount() == 0

new_segment = (
    first_record
    | df["time_gap_seconds"].isna()
    | (df["time_gap_seconds"] <= 0)
    | (
        df["time_gap_seconds"]
        > MAX_SOURCE_TIME_GAP_SECONDS
    )
    | (
        implied_speed_knots
        > MAX_REALISTIC_SPEED_KNOTS
    )
)

df["segment_id"] = (
    new_segment
    .groupby(df["MMSI"])
    .cumsum()
    .astype("int32")
)

total_segments = df.groupby(
    ["MMSI", "segment_id"]
).ngroups

print(f"Continuous segments: {total_segments:,}")


# ============================================================
# STRATIFIED VESSEL SPLIT
# The stratum is based on the maximum speed reached by a vessel.
# ============================================================
vessel_profiles = (
    df.groupby("MMSI")["SOG"]
    .max()
    .rename("maximum_speed")
    .reset_index()
)

vessel_profiles["activity_class"] = (
    vessel_profiles["maximum_speed"]
    .apply(classify_speed)
)

train_profiles, temporary_profiles = train_test_split(
    vessel_profiles,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=vessel_profiles["activity_class"],
)

validation_profiles, test_profiles = train_test_split(
    temporary_profiles,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temporary_profiles["activity_class"],
)

train_vessels = set(
    train_profiles["MMSI"].tolist()
)

validation_vessels = set(
    validation_profiles["MMSI"].tolist()
)

test_vessels = set(
    test_profiles["MMSI"].tolist()
)


def split_for_vessel(mmsi):
    if mmsi in train_vessels:
        return "train"

    if mmsi in validation_vessels:
        return "val"

    return "test"


print("\nVessel split")
print("-" * 35)
print(f"Train vessels: {len(train_vessels):,}")
print(f"Val vessels  : {len(validation_vessels):,}")
print(f"Test vessels : {len(test_vessels):,}")


# ============================================================
# STORAGE
# ============================================================
storage = {
    split_name: {
        "X": [],
        "y_position": [],
        "y_speed": [],
        "y_class": [],
        "mmsi": [],
        "target_timestamp": [],
    }
    for split_name in [
        "train",
        "val",
        "test",
    ]
}


# ============================================================
# RESAMPLE TRACKS AND CREATE SEQUENCES
# ============================================================
segments = df.groupby(
    ["MMSI", "segment_id"],
    sort=False,
)

print("\nResampling tracks and creating sequences...")

for segment_number, (
    (mmsi, segment_id),
    segment,
) in enumerate(segments, start=1):

    if len(segment) < 5:
        continue

    segment = segment.copy()

    duration_minutes = (
        segment["BaseDateTime"].max()
        - segment["BaseDateTime"].min()
    ).total_seconds() / 60.0

    if duration_minutes < (
        TOTAL_WINDOW_STEPS - 1
    ):
        continue

    segment["course_missing"] = (
        segment["COG"].isna()
    ).astype(np.float32)

    course_radians = np.radians(
        segment["COG"]
    )

    segment["sin_course"] = np.sin(
        course_radians
    )

    segment["cos_course"] = np.cos(
        course_radians
    )

    segment["observed"] = 1.0

    segment = segment.set_index(
        "BaseDateTime"
    )

    resampled = segment[
        [
            "LAT",
            "LON",
            "SOG",
            "sin_course",
            "cos_course",
            "course_missing",
            "observed",
        ]
    ].resample(
        RESAMPLE_INTERVAL
    ).agg(
        {
            "LAT": "mean",
            "LON": "mean",
            "SOG": "mean",
            "sin_course": "mean",
            "cos_course": "mean",
            "course_missing": "max",
            "observed": "sum",
        }
    )

    resampled["observed"] = (
        resampled["observed"]
        .fillna(0)
        .gt(0)
        .astype(np.float32)
    )

    resampled["course_missing"] = (
        resampled["course_missing"]
        .fillna(1.0)
        .astype(np.float32)
    )

    for column in [
        "LAT",
        "LON",
        "SOG",
        "sin_course",
        "cos_course",
    ]:
        resampled[column] = (
            resampled[column]
            .interpolate(
                method="time",
                limit=MAX_INTERPOLATED_MINUTES,
                limit_area="inside",
            )
        )

    # Derive direction from movement when AIS COG is unavailable.
    valid_position = (
        resampled["LAT"].notna()
        & resampled["LON"].notna()
    )

    if valid_position.sum() < TOTAL_WINDOW_STEPS:
        continue

    first_valid_index = np.flatnonzero(
        valid_position.to_numpy()
    )[0]

    reference_latitude = float(
        resampled["LAT"].iloc[first_valid_index]
    )

    reference_longitude = float(
        resampled["LON"].iloc[first_valid_index]
    )

    all_east, all_north = local_xy_metres(
        resampled["LAT"].to_numpy(),
        resampled["LON"].to_numpy(),
        reference_latitude,
        reference_longitude,
    )

    east_change = np.diff(
        all_east,
        prepend=all_east[0],
    )

    north_change = np.diff(
        all_north,
        prepend=all_north[0],
    )

    derived_course = np.arctan2(
        east_change,
        north_change,
    )

    derived_sin = np.sin(derived_course)
    derived_cos = np.cos(derived_course)

    missing_direction = (
        resampled["sin_course"].isna()
        | resampled["cos_course"].isna()
    )

    resampled.loc[
        missing_direction,
        "sin_course",
    ] = derived_sin[missing_direction.to_numpy()]

    resampled.loc[
        missing_direction,
        "cos_course",
    ] = derived_cos[missing_direction.to_numpy()]

    direction_norm = np.sqrt(
        resampled["sin_course"] ** 2
        + resampled["cos_course"] ** 2
    )

    valid_direction_norm = direction_norm > 1e-6

    resampled.loc[
        valid_direction_norm,
        "sin_course",
    ] /= direction_norm[valid_direction_norm]

    resampled.loc[
        valid_direction_norm,
        "cos_course",
    ] /= direction_norm[valid_direction_norm]

    resampled["valid"] = (
        resampled[
            [
                "LAT",
                "LON",
                "SOG",
                "sin_course",
                "cos_course",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    split_name = split_for_vessel(int(mmsi))

    number_of_rows = len(resampled)

    for start_index in range(
        0,
        number_of_rows - TOTAL_WINDOW_STEPS + 1,
        SEQUENCE_STRIDE,
    ):
        end_index = (
            start_index
            + TOTAL_WINDOW_STEPS
        )

        window = resampled.iloc[
            start_index:end_index
        ]

        if not window["valid"].all():
            continue

        history = window.iloc[
            :HISTORY_STEPS
        ]

        target = window.iloc[-1]

        if (
            history["observed"].sum()
            < MIN_OBSERVED_HISTORY_POINTS
        ):
            continue

        # The future target should come from a real AIS report,
        # not only from interpolation.
        if target["observed"] < 1:
            continue

        history_east, history_north = local_xy_metres(
            history["LAT"].to_numpy(),
            history["LON"].to_numpy(),
            history["LAT"].iloc[0],
            history["LON"].iloc[0],
        )

        features = np.column_stack(
            [
                history_east,
                history_north,
                history["SOG"].to_numpy(),
                history[
                    "sin_course"
                ].to_numpy(),
                history[
                    "cos_course"
                ].to_numpy(),
                history[
                    "observed"
                ].to_numpy(),
                history[
                    "course_missing"
                ].to_numpy(),
            ]
        ).astype(np.float32)

        last_history = history.iloc[-1]

        target_east, target_north = local_xy_metres(
            [target["LAT"]],
            [target["LON"]],
            last_history["LAT"],
            last_history["LON"],
        )

        target_displacement = np.array(
            [
                target_east[0],
                target_north[0],
            ],
            dtype=np.float32,
        )

        target_speed = float(target["SOG"])

        if not (
            0 <= target_speed
            <= MAX_REALISTIC_SPEED_KNOTS
        ):
            continue

        target_class = classify_speed(
            target_speed
        )

        storage[split_name]["X"].append(
            features
        )

        storage[split_name][
            "y_position"
        ].append(target_displacement)

        storage[split_name][
            "y_speed"
        ].append(target_speed)

        storage[split_name][
            "y_class"
        ].append(target_class)

        storage[split_name]["mmsi"].append(
            int(mmsi)
        )

        storage[split_name][
            "target_timestamp"
        ].append(
            int(target.name.timestamp())
        )

    if segment_number % 2_000 == 0:
        print(
            f"Processed "
            f"{segment_number:,}/"
            f"{total_segments:,} segments"
        )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================
arrays = {}

for split_name, values in storage.items():
    if not values["X"]:
        raise RuntimeError(
            f"No samples generated for {split_name}"
        )

    arrays[split_name] = {
        "X": np.stack(
            values["X"]
        ).astype(np.float32),

        "y_position": np.stack(
            values["y_position"]
        ).astype(np.float32),

        "y_speed": np.asarray(
            values["y_speed"],
            dtype=np.float32,
        ),

        "y_class": np.asarray(
            values["y_class"],
            dtype=np.int64,
        ),

        "mmsi": np.asarray(
            values["mmsi"],
            dtype=np.int64,
        ),

        "target_timestamp": np.asarray(
            values["target_timestamp"],
            dtype=np.int64,
        ),
    }


# ============================================================
# NORMALISE INPUTS USING TRAINING DATA ONLY
# ============================================================
feature_mean = arrays["train"]["X"].mean(
    axis=(0, 1)
)

feature_std = arrays["train"]["X"].std(
    axis=(0, 1)
)

feature_std = np.where(
    feature_std < 1e-6,
    1.0,
    feature_std,
)

for split_name in arrays:
    arrays[split_name]["X"] = (
        (
            arrays[split_name]["X"]
            - feature_mean
        )
        / feature_std
    ).astype(np.float32)


# ============================================================
# SAVE
# ============================================================
metadata = {
    "resample_interval": RESAMPLE_INTERVAL,
    "history_steps": HISTORY_STEPS,
    "history_minutes": HISTORY_STEPS,
    "forecast_steps": FORECAST_STEPS,
    "forecast_minutes": FORECAST_STEPS,
    "feature_names": FEATURE_NAMES,
    "feature_mean": feature_mean.tolist(),
    "feature_std": feature_std.tolist(),
    "class_names": CLASS_NAMES,
    "class_boundaries_knots": {
        "Stopped": "0.0 to 2.0",
        "Slow": "2.1 to 6.0",
        "Moderate": "6.1 to 12.0",
        "Fast": "above 12.0",
    },
    "split_method": (
        "70/15/15 unique MMSI split stratified "
        "by maximum vessel activity class"
    ),
    "splits": {},
}

for split_name, split_arrays in arrays.items():
    output_path = (
        OUTPUT_DIR
        / f"{split_name}_motion_sequences.npz"
    )

    np.savez_compressed(
        output_path,
        **split_arrays,
    )

    class_counts = np.bincount(
        split_arrays["y_class"],
        minlength=len(CLASS_NAMES),
    )

    metadata["splits"][split_name] = {
        "samples": int(
            len(split_arrays["X"])
        ),
        "unique_vessels": int(
            np.unique(
                split_arrays["mmsi"]
            ).size
        ),
        "class_distribution": {
            CLASS_NAMES[index]: int(
                class_counts[index]
            )
            for index in range(
                len(CLASS_NAMES)
            )
        },
    }

    print(
        f"Saved {split_name}: "
        f"{len(split_arrays['X']):,} samples"
    )

with (
    OUTPUT_DIR / "metadata.json"
).open("w") as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FIVE-MINUTE MOTION DATASET READY")
print("=" * 70)

for split_name in [
    "train",
    "val",
    "test",
]:
    split_metadata = metadata[
        "splits"
    ][split_name]

    print(f"\n{split_name.upper()}")
    print(
        f"Samples        : "
        f"{split_metadata['samples']:,}"
    )
    print(
        f"Unique vessels : "
        f"{split_metadata['unique_vessels']:,}"
    )

    print("Class distribution:")

    for class_name, count in split_metadata[
        "class_distribution"
    ].items():
        print(
            f"  {class_name:8}: {count:,}"
        )

print("\nSaved to:")
print(OUTPUT_DIR)
