from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import EARTH_RADIUS_METRES, KNOT_TO_METRES_PER_SECOND


def local_xy_metres(
    latitudes: Iterable[float],
    longitudes: Iterable[float],
    reference_latitude: float,
    reference_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(list(latitudes), dtype=np.float64)
    lon = np.asarray(list(longitudes), dtype=np.float64)
    reference_latitude_radians = math.radians(reference_latitude)
    east = (
        EARTH_RADIUS_METRES
        * np.radians(lon - reference_longitude)
        * math.cos(reference_latitude_radians)
    )
    north = EARTH_RADIUS_METRES * np.radians(lat - reference_latitude)
    return east, north


def offset_lat_lon(
    latitude: float,
    longitude: float,
    east_metres: float,
    north_metres: float,
) -> tuple[float, float]:
    latitude_offset = math.degrees(north_metres / EARTH_RADIUS_METRES)
    cosine = max(abs(math.cos(math.radians(latitude))), 1e-12)
    longitude_offset = math.degrees(east_metres / (EARTH_RADIUS_METRES * cosine))
    return latitude + latitude_offset, longitude + longitude_offset


def haversine_metres(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    lat1 = math.radians(latitude_1)
    lat2 = math.radians(latitude_2)
    dlat = lat2 - lat1
    dlon = math.radians(longitude_2 - longitude_1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_METRES * math.asin(min(1.0, math.sqrt(a)))


def bearing_degrees(east_metres: float, north_metres: float) -> float:
    return (math.degrees(math.atan2(east_metres, north_metres)) + 360.0) % 360.0


def velocity_components(speed_knots: float, course_degrees: float) -> tuple[float, float]:
    speed_mps = speed_knots * KNOT_TO_METRES_PER_SECOND
    radians = math.radians(course_degrees)
    return speed_mps * math.sin(radians), speed_mps * math.cos(radians)


def course_from_points(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    east, north = local_xy_metres(
        [latitude_2], [longitude_2], latitude_1, longitude_1
    )
    return bearing_degrees(float(east[0]), float(north[0]))
