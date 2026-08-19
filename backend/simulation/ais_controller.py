from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import AIS_CSV_PATH
from .geo import local_xy_metres, offset_lat_lon

REQUIRED_COLUMNS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"]


@dataclass
class AISScenario:
    scenario_id: str
    mode: str
    description: str
    own_states: list[dict[str, Any]]
    target_states: list[dict[str, Any]]

    @property
    def length(self) -> int:
        return min(len(self.own_states), len(self.target_states))


class AISSimulationController:
    def __init__(self):
        self._dataframe: pd.DataFrame | None = None
        self._segment_stats: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._dataframe is not None:
            return self._dataframe
        if not AIS_CSV_PATH.exists():
            raise FileNotFoundError(f"AIS CSV not found: {AIS_CSV_PATH}")
        dataframe = pd.read_csv(AIS_CSV_PATH, usecols=REQUIRED_COLUMNS)
        dataframe["BaseDateTime"] = pd.to_datetime(
            dataframe["BaseDateTime"], errors="coerce"
        )
        for column in ["MMSI", "LAT", "LON", "SOG", "COG"]:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        valid = (
            dataframe["MMSI"].notna()
            & dataframe["BaseDateTime"].notna()
            & dataframe["LAT"].between(-90, 90)
            & dataframe["LON"].between(-180, 180)
            & dataframe["SOG"].between(0, 80)
        )
        dataframe = dataframe.loc[valid].copy()
        dataframe["MMSI"] = dataframe["MMSI"].astype("int64")
        dataframe.loc[~dataframe["COG"].between(0, 359.999), "COG"] = np.nan
        dataframe = (
            dataframe.sort_values(["MMSI", "BaseDateTime"])
            .drop_duplicates(["MMSI", "BaseDateTime"], keep="last")
            .reset_index(drop=True)
        )
        gaps = dataframe.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds()
        new_segment = gaps.isna() | (gaps <= 0) | (gaps > 300)
        dataframe["segment_id"] = new_segment.groupby(dataframe["MMSI"]).cumsum().astype("int32")
        stats = dataframe.groupby(["MMSI", "segment_id"]).agg(
            start=("BaseDateTime", "min"),
            end=("BaseDateTime", "max"),
            rows=("BaseDateTime", "size"),
            mean_speed=("SOG", "mean"),
            first_lat=("LAT", "first"),
            first_lon=("LON", "first"),
        ).reset_index()
        stats["duration_minutes"] = (stats["end"] - stats["start"]).dt.total_seconds() / 60.0
        self._dataframe = dataframe
        self._segment_stats = stats
        return dataframe

    @staticmethod
    def _resample(segment: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
        segment = segment.sort_values("BaseDateTime").copy().set_index("BaseDateTime")
        segment["course_missing"] = segment["COG"].isna().astype(np.float32)
        radians = np.radians(segment["COG"])
        segment["sin_course"] = np.sin(radians)
        segment["cos_course"] = np.cos(radians)
        segment["observed"] = 1.0
        rule = f"{interval_seconds}s"
        resampled = segment[[
            "LAT",
            "LON",
            "SOG",
            "sin_course",
            "cos_course",
            "course_missing",
            "observed",
        ]].resample(rule).agg({
            "LAT": "mean",
            "LON": "mean",
            "SOG": "mean",
            "sin_course": "mean",
            "cos_course": "mean",
            "course_missing": "max",
            "observed": "sum",
        })
        resampled["observed"] = resampled["observed"].fillna(0).gt(0)
        resampled["course_missing"] = (
            resampled["course_missing"].fillna(1.0).astype(bool)
        )
        for column in ["LAT", "LON", "SOG", "sin_course", "cos_course"]:
            resampled[column] = resampled[column].interpolate(method="time", limit_area="inside")
        resampled = resampled.dropna(subset=["LAT", "LON", "SOG"])
        east, north = local_xy_metres(
            resampled["LAT"].to_numpy(),
            resampled["LON"].to_numpy(),
            float(resampled["LAT"].iloc[0]),
            float(resampled["LON"].iloc[0]),
        )
        derived = np.degrees(np.arctan2(np.diff(east, prepend=east[0]), np.diff(north, prepend=north[0]))) % 360
        course = np.degrees(np.arctan2(resampled["sin_course"], resampled["cos_course"])) % 360
        resampled["COG"] = np.where(np.isfinite(course), course, derived)
        return resampled

    def _get_segment(self, mmsi: int, segment_id: int) -> pd.DataFrame:
        dataframe = self._load()
        return dataframe.loc[
            (dataframe["MMSI"] == mmsi) & (dataframe["segment_id"] == segment_id)
        ].copy()

    @staticmethod
    def _slice_track(track: pd.DataFrame, point_count: int, rng: random.Random) -> pd.DataFrame | None:
        if len(track) < point_count:
            return None
        max_start = len(track) - point_count
        start = rng.randint(0, max_start) if max_start else 0
        return track.iloc[start:start + point_count].copy()

    @staticmethod
    def _states(track: pd.DataFrame, mmsi: int) -> list[dict[str, Any]]:
        return [
            {
                "mmsi": int(mmsi),
                "timestamp": index.to_pydatetime(),
                "latitude": float(row.LAT),
                "longitude": float(row.LON),
                "speed_knots": float(row.SOG),
                "course_degrees": float(row.COG) % 360.0,
                "observed": bool(row.observed),
                "course_missing": bool(row.course_missing),
            }
            for index, row in track.iterrows()
        ]

    def create_scenario(
        self,
        mode: str,
        interval_seconds: int = 10,
        duration_minutes: int = 20,
        seed: int = 42,
        allowed_mmsi: set[int] | None = None,
    ) -> AISScenario:
        self._load()
        if mode == "actual":
            return self._create_actual(
                interval_seconds, duration_minutes, seed, allowed_mmsi
            )
        return self._create_constructed(
            interval_seconds, duration_minutes, seed, allowed_mmsi
        )

    def _eligible_segments(
        self,
        duration_minutes: int,
        allowed_mmsi: set[int] | None = None,
    ) -> pd.DataFrame:
        assert self._segment_stats is not None
        eligible = self._segment_stats.loc[
            (self._segment_stats["duration_minutes"] >= duration_minutes)
            & (self._segment_stats["rows"] >= 10)
            & (self._segment_stats["mean_speed"] >= 2.1)
        ].copy()
        if allowed_mmsi is not None:
            eligible = eligible.loc[eligible["MMSI"].isin(allowed_mmsi)].copy()
        return eligible

    def _create_constructed(
        self,
        interval_seconds: int,
        duration_minutes: int,
        seed: int,
        allowed_mmsi: set[int] | None = None,
    ) -> AISScenario:
        rng = random.Random(seed)
        point_count = duration_minutes * 60 // interval_seconds + 1
        eligible = self._eligible_segments(duration_minutes + 2, allowed_mmsi)
        if len(eligible) < 2:
            raise RuntimeError("Not enough continuous moving AIS segments for a constructed encounter.")
        indices = list(eligible.index)
        rng.shuffle(indices)
        selected: list[tuple[pd.Series, pd.DataFrame]] = []
        used_mmsi: set[int] = set()
        for index in indices:
            row = eligible.loc[index]
            mmsi = int(row.MMSI)
            if mmsi in used_mmsi:
                continue
            track = self._resample(self._get_segment(mmsi, int(row.segment_id)), interval_seconds)
            sliced = self._slice_track(track, point_count, rng)
            if sliced is not None:
                selected.append((row, sliced))
                used_mmsi.add(mmsi)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            raise RuntimeError("Could not create two sufficiently long AIS tracks.")
        own_row, own_track = selected[0]
        target_row, target_track = selected[1]

        own_lat0 = float(own_track["LAT"].iloc[0])
        own_lon0 = float(own_track["LON"].iloc[0])
        own_east, own_north = local_xy_metres(
            own_track["LAT"], own_track["LON"], own_lat0, own_lon0
        )
        target_east, target_north = local_xy_metres(
            target_track["LAT"], target_track["LON"],
            float(target_track["LAT"].iloc[0]), float(target_track["LON"].iloc[0])
        )
        own_delta = np.array([own_east[-1] - own_east[0], own_north[-1] - own_north[0]])
        target_delta = np.array([target_east[-1] - target_east[0], target_north[-1] - target_north[0]])
        own_angle = math.atan2(own_delta[1], own_delta[0]) if np.linalg.norm(own_delta) > 1 else 0.0
        target_angle = math.atan2(target_delta[1], target_delta[0]) if np.linalg.norm(target_delta) > 1 else math.pi
        rotation = own_angle + math.pi - target_angle
        cosine, sine = math.cos(rotation), math.sin(rotation)
        rotated_east = target_east * cosine - target_north * sine
        rotated_north = target_east * sine + target_north * cosine
        initial_distance = 3.0 * 1852.0
        placement_east = initial_distance * math.cos(own_angle)
        placement_north = initial_distance * math.sin(own_angle)
        transformed_east = rotated_east + placement_east
        transformed_north = rotated_north + placement_north

        transformed_lat = []
        transformed_lon = []
        for east_value, north_value in zip(transformed_east, transformed_north):
            latitude, longitude = offset_lat_lon(
                own_lat0, own_lon0, float(east_value), float(north_value)
            )
            transformed_lat.append(latitude)
            transformed_lon.append(longitude)
        target_track = target_track.copy()
        target_track["LAT"] = transformed_lat
        target_track["LON"] = transformed_lon
        east_step = np.diff(transformed_east, prepend=transformed_east[0])
        north_step = np.diff(transformed_north, prepend=transformed_north[0])
        target_track["COG"] = np.degrees(np.arctan2(east_step, north_step)) % 360.0
        if len(target_track) > 1:
            target_track.iloc[0, target_track.columns.get_loc("COG")] = float(
                target_track["COG"].iloc[1]
            )
        # Align target timestamps to the own-vessel simulation clock.
        target_track.index = own_track.index

        return AISScenario(
            scenario_id=f"constructed-{seed}-{int(own_row.MMSI)}-{int(target_row.MMSI)}",
            mode="constructed",
            description=(
                "Two historical AIS motion patterns placed into a controlled head-on encounter. "
                "Motion behaviour is retained, while the target track is rotated and geographically shifted."
            ),
            own_states=self._states(own_track, int(own_row.MMSI)),
            target_states=self._states(target_track, int(target_row.MMSI)),
        )

    def _create_actual(
        self,
        interval_seconds: int,
        duration_minutes: int,
        seed: int,
        allowed_mmsi: set[int] | None = None,
    ) -> AISScenario:
        dataframe = self._load()
        if allowed_mmsi is not None:
            dataframe = dataframe.loc[dataframe["MMSI"].isin(allowed_mmsi)].copy()
            if dataframe.empty:
                raise RuntimeError("No AIS rows match the allowed MMSI evaluation set.")
        rng = random.Random(seed)
        point_count = duration_minutes * 60 // interval_seconds + 1
        sample = dataframe[["MMSI", "BaseDateTime", "LAT", "LON"]].copy()
        sample["minute"] = sample["BaseDateTime"].dt.floor("min")
        sample["lat_bin"] = (sample["LAT"] / 0.05).round().astype(int)
        sample["lon_bin"] = (sample["LON"] / 0.05).round().astype(int)
        grouped = sample.groupby(["minute", "lat_bin", "lon_bin"], sort=False)
        candidate_counts = grouped["MMSI"].nunique()
        candidate_keys = list(candidate_counts[candidate_counts >= 2].index)
        rng.shuffle(candidate_keys)
        for minute, lat_bin, lon_bin in candidate_keys[:500]:
            candidate_rows = grouped.get_group((minute, lat_bin, lon_bin))
            vessel_ids = [int(value) for value in pd.unique(candidate_rows["MMSI"])]
            rng.shuffle(vessel_ids)
            own_mmsi, target_mmsi = vessel_ids[:2]
            start = minute - pd.Timedelta(minutes=1)
            end = start + pd.Timedelta(minutes=duration_minutes + 2)

            own_near = dataframe.loc[
                (dataframe["MMSI"] == own_mmsi)
                & dataframe["BaseDateTime"].between(
                    minute - pd.Timedelta(minutes=5),
                    minute + pd.Timedelta(minutes=5),
                )
            ].copy()
            target_near = dataframe.loc[
                (dataframe["MMSI"] == target_mmsi)
                & dataframe["BaseDateTime"].between(
                    minute - pd.Timedelta(minutes=5),
                    minute + pd.Timedelta(minutes=5),
                )
            ].copy()
            if own_near.empty or target_near.empty:
                continue
            own_near["distance_to_candidate"] = (
                own_near["BaseDateTime"] - minute
            ).abs()
            target_near["distance_to_candidate"] = (
                target_near["BaseDateTime"] - minute
            ).abs()
            own_segment_id = int(
                own_near.sort_values("distance_to_candidate")["segment_id"].iloc[0]
            )
            target_segment_id = int(
                target_near.sort_values("distance_to_candidate")["segment_id"].iloc[0]
            )
            own_raw = dataframe.loc[
                (dataframe["MMSI"] == own_mmsi)
                & (dataframe["segment_id"] == own_segment_id)
                & dataframe["BaseDateTime"].between(start, end)
            ]
            target_raw = dataframe.loc[
                (dataframe["MMSI"] == target_mmsi)
                & (dataframe["segment_id"] == target_segment_id)
                & dataframe["BaseDateTime"].between(start, end)
            ]
            if own_raw.empty or target_raw.empty:
                continue
            own_track = self._resample(own_raw, interval_seconds)
            target_track = self._resample(target_raw, interval_seconds)
            common = own_track.index.intersection(target_track.index)
            if len(common) < point_count:
                continue
            common = common[:point_count]
            own_track = own_track.loc[common]
            target_track = target_track.loc[common]
            return AISScenario(
                scenario_id=f"actual-{seed}-{own_mmsi}-{target_mmsi}",
                mode="actual",
                description="Two historical AIS vessels observed at overlapping times and nearby spatial bins.",
                own_states=self._states(own_track, own_mmsi),
                target_states=self._states(target_track, target_mmsi),
            )
        raise RuntimeError(
            "No suitable actual encounter was found quickly. Use constructed mode or another seed."
        )
