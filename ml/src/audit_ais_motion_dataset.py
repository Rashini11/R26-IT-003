from pathlib import Path
import json

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "ml/external_datasets/ais_motion/processed_AIS_dataset.csv"
)

OUTPUT_DIR = Path("ml/ais_motion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 10
SEQUENCE_STRIDE = 5
MAX_TIME_GAP_SECONDS = 600  # 10 minutes


REQUIRED_COLUMNS = [
    "MMSI",
    "BaseDateTime",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Heading",
    "Speed_Category",
]


print("Loading required AIS columns...")

df = pd.read_csv(
    DATA_PATH,
    usecols=REQUIRED_COLUMNS,
)

raw_row_count = len(df)

print(f"Raw rows: {raw_row_count:,}")
print(f"Raw vessels: {df['MMSI'].nunique():,}")


# ============================================================
# CONVERT DATA TYPES
# ============================================================
df["BaseDateTime"] = pd.to_datetime(
    df["BaseDateTime"],
    errors="coerce",
)

numeric_columns = [
    "MMSI",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Heading",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# COUNT INVALID VALUES
# ============================================================
invalid_timestamp_count = int(
    df["BaseDateTime"].isna().sum()
)

invalid_position_mask = (
    df["LAT"].isna()
    | df["LON"].isna()
    | ~df["LAT"].between(-90, 90)
    | ~df["LON"].between(-180, 180)
)

invalid_speed_mask = (
    df["SOG"].isna()
    | (df["SOG"] < 0)
    | (df["SOG"] > 70)
)

invalid_cog_mask = (
    df["COG"].isna()
    | (df["COG"] < 0)
    | (df["COG"] >= 360)
)

invalid_heading_mask = (
    df["Heading"].isna()
    | (df["Heading"] < 0)
    | (df["Heading"] >= 360)
)


print("\nInvalid values")
print("------------------------------")
print(f"Invalid timestamps : {invalid_timestamp_count:,}")
print(f"Invalid positions  : {invalid_position_mask.sum():,}")
print(f"Invalid speeds     : {invalid_speed_mask.sum():,}")
print(f"Invalid COG values : {invalid_cog_mask.sum():,}")
print(f"Invalid headings   : {invalid_heading_mask.sum():,}")


# Invalid directions can later be imputed.
df.loc[invalid_cog_mask, "COG"] = np.nan
df.loc[invalid_heading_mask, "Heading"] = np.nan


# Remove rows unusable for trajectory learning.
valid_mask = (
    df["MMSI"].notna()
    & df["BaseDateTime"].notna()
    & ~invalid_position_mask
    & ~invalid_speed_mask
)

df = df.loc[valid_mask].copy()

df["MMSI"] = df["MMSI"].astype("int64")

df = df.sort_values(
    ["MMSI", "BaseDateTime"],
).reset_index(drop=True)


# ============================================================
# TIME DIFFERENCES AND TRACK SEGMENTS
# ============================================================
df["time_gap_seconds"] = (
    df.groupby("MMSI")["BaseDateTime"]
    .diff()
    .dt.total_seconds()
)

new_segment_mask = (
    df["time_gap_seconds"].isna()
    | (df["time_gap_seconds"] <= 0)
    | (
        df["time_gap_seconds"]
        > MAX_TIME_GAP_SECONDS
    )
)

df["segment_id"] = (
    new_segment_mask
    .groupby(df["MMSI"])
    .cumsum()
    .astype("int32")
)

segment_sizes = (
    df.groupby(
        ["MMSI", "segment_id"],
        sort=False,
    )
    .size()
    .rename("records")
    .reset_index()
)

eligible_segments = segment_sizes[
    segment_sizes["records"] > SEQUENCE_LENGTH
].copy()

eligible_segments["sequence_count"] = (
    (
        eligible_segments["records"]
        - SEQUENCE_LENGTH
        - 1
    )
    // SEQUENCE_STRIDE
    + 1
)

eligible_segments = eligible_segments[
    eligible_segments["sequence_count"] > 0
]

estimated_sequences = int(
    eligible_segments["sequence_count"].sum()
)


# ============================================================
# REPORT DISTRIBUTIONS
# ============================================================
category_distribution = (
    df["Speed_Category"]
    .fillna("Missing")
    .value_counts()
    .to_dict()
)

records_per_vessel = (
    df.groupby("MMSI")
    .size()
)

positive_time_gaps = df.loc[
    df["time_gap_seconds"] > 0,
    "time_gap_seconds",
]

gap_quantiles = (
    positive_time_gaps.quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    .round(2)
    .to_dict()
)


summary = {
    "raw_rows": raw_row_count,
    "clean_rows": int(len(df)),
    "removed_rows": int(raw_row_count - len(df)),
    "unique_vessels": int(df["MMSI"].nunique()),
    "start_time": str(df["BaseDateTime"].min()),
    "end_time": str(df["BaseDateTime"].max()),
    "sequence_configuration": {
        "history_records": SEQUENCE_LENGTH,
        "stride": SEQUENCE_STRIDE,
        "maximum_time_gap_seconds": (
            MAX_TIME_GAP_SECONDS
        ),
    },
    "invalid_values": {
        "timestamps": invalid_timestamp_count,
        "positions": int(
            invalid_position_mask.sum()
        ),
        "speeds": int(invalid_speed_mask.sum()),
        "course_over_ground": int(
            invalid_cog_mask.sum()
        ),
        "heading": int(
            invalid_heading_mask.sum()
        ),
    },
    "records_per_vessel": {
        "minimum": int(records_per_vessel.min()),
        "median": float(records_per_vessel.median()),
        "mean": round(
            float(records_per_vessel.mean()),
            2,
        ),
        "maximum": int(records_per_vessel.max()),
    },
    "time_gap_quantiles_seconds": {
        str(key): float(value)
        for key, value in gap_quantiles.items()
    },
    "speed_category_distribution": {
        str(key): int(value)
        for key, value in category_distribution.items()
    },
    "segments": {
        "total_segments": int(
            len(segment_sizes)
        ),
        "eligible_segments": int(
            len(eligible_segments)
        ),
        "eligible_vessels": int(
            eligible_segments["MMSI"].nunique()
        ),
        "estimated_training_sequences": (
            estimated_sequences
        ),
    },
}


summary_path = OUTPUT_DIR / "dataset_audit.json"

with summary_path.open("w") as file:
    json.dump(summary, file, indent=2)


print("\nDataset summary")
print("=" * 60)
print(f"Clean rows               : {len(df):,}")
print(f"Rows removed             : {raw_row_count - len(df):,}")
print(f"Unique vessels           : {df['MMSI'].nunique():,}")
print(f"Total track segments     : {len(segment_sizes):,}")
print(f"Eligible segments        : {len(eligible_segments):,}")
print(
    "Eligible vessels        : "
    f"{eligible_segments['MMSI'].nunique():,}"
)
print(
    "Estimated sequences     : "
    f"{estimated_sequences:,}"
)

print("\nRecords per vessel")
print("------------------------------")
print(
    records_per_vessel.describe().round(2)
)

print("\nTime-gap quantiles in seconds")
print("------------------------------")
print(
    positive_time_gaps.quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).round(2)
)

print("\nSpeed-category distribution")
print("------------------------------")
print(
    df["Speed_Category"]
    .fillna("Missing")
    .value_counts()
)

print("\nAudit saved to:")
print(summary_path)
