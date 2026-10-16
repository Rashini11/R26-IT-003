from __future__ import annotations

import json
import math
import threading
import time

from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

MARINE_URL = (
    "https://marine-api.open-meteo.com/v1/marine"
)

CACHE_TTL_SECONDS = 600

# Don't use a marine grid point that is
# unreasonably far from the vessel encounter.
MAX_MARINE_GRID_DISTANCE_KM = 50.0


_CACHE = {}
_CACHE_LOCK = threading.Lock()


# ============================================================
# UTILITIES
# ============================================================

def _request_json(
    url: str,
    params: dict,
) -> dict:

    query = urlencode(params)

    request = Request(
        f"{url}?{query}",
        headers={
            "User-Agent":
                "OceanIQ-R26-IT-003/1.0"
        },
    )

    with urlopen(
        request,
        timeout=6,
    ) as response:

        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def _haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2
        * radius
        * math.asin(
            math.sqrt(value)
        )
    )


def _speed_to_knots(
    value,
    unit,
):

    if value is None:
        return None

    value = float(value)

    unit = str(
        unit or ""
    ).lower()

    if "km/h" in unit:
        return value * 0.539957

    if "m/s" in unit:
        return value * 1.943844

    if "mph" in unit:
        return value * 0.868976

    return value


def _round(
    value,
    digits=2,
):

    if value is None:
        return None

    return round(
        float(value),
        digits,
    )


# ============================================================
# ATMOSPHERIC WEATHER
# ============================================================

def _fetch_weather(
    latitude,
    longitude,
):

    variables = ",".join(
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "pressure_msl",
            "visibility",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ]
    )

    data = _request_json(
        WEATHER_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": variables,
            "timezone": "UTC",
            "forecast_days": 1,
        },
    )

    current = (
        data.get("current")
        or {}
    )

    units = (
        data.get("current_units")
        or {}
    )

    visibility = current.get(
        "visibility"
    )

    return {
        "available": bool(current),

        "observation_time":
            current.get("time"),

        "temperature_c":
            _round(
                current.get(
                    "temperature_2m"
                ),
                1,
            ),

        "relative_humidity_percent":
            _round(
                current.get(
                    "relative_humidity_2m"
                ),
                1,
            ),

        "precipitation_mm":
            _round(
                current.get(
                    "precipitation"
                ),
                2,
            ),

        "rain_mm":
            _round(
                current.get("rain"),
                2,
            ),

        "weather_code":
            current.get(
                "weather_code"
            ),

        "cloud_cover_percent":
            _round(
                current.get(
                    "cloud_cover"
                ),
                1,
            ),

        "pressure_msl_hpa":
            _round(
                current.get(
                    "pressure_msl"
                ),
                1,
            ),

        "visibility_km":
            (
                _round(
                    visibility / 1000,
                    2,
                )
                if visibility is not None
                else None
            ),

        "wind_speed_knots":
            _round(
                _speed_to_knots(
                    current.get(
                        "wind_speed_10m"
                    ),
                    units.get(
                        "wind_speed_10m"
                    ),
                ),
                2,
            ),

        "wind_direction_degrees":
            _round(
                current.get(
                    "wind_direction_10m"
                ),
                1,
            ),

        "wind_gust_knots":
            _round(
                _speed_to_knots(
                    current.get(
                        "wind_gusts_10m"
                    ),
                    units.get(
                        "wind_gusts_10m"
                    ),
                ),
                2,
            ),
    }


# ============================================================
# MARINE ENVIRONMENT
# ============================================================

def _fetch_marine(
    latitude,
    longitude,
):

    variables = ",".join(
        [
            "wave_height",
            "wave_direction",
            "wave_period",
            "swell_wave_height",
            "swell_wave_direction",
            "swell_wave_period",
            "sea_surface_temperature",
            "ocean_current_velocity",
            "ocean_current_direction",
        ]
    )

    data = _request_json(
        MARINE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": variables,
            "timezone": "UTC",
            "forecast_days": 1,
            "cell_selection": "sea",
        },
    )

    returned_latitude = data.get(
        "latitude"
    )

    returned_longitude = data.get(
        "longitude"
    )

    if (
        returned_latitude is None
        or returned_longitude is None
    ):
        return {
            "available": False,
            "reason":
                "Marine grid unavailable.",
        }

    grid_distance_km = (
        _haversine_km(
            latitude,
            longitude,
            float(returned_latitude),
            float(returned_longitude),
        )
    )

    if (
        grid_distance_km
        > MAX_MARINE_GRID_DISTANCE_KM
    ):
        return {
            "available": False,
            "reason": (
                "Nearest marine grid is "
                "too far from the encounter."
            ),
            "grid_distance_km":
                _round(
                    grid_distance_km,
                    2,
                ),
        }

    current = (
        data.get("current")
        or {}
    )

    units = (
        data.get("current_units")
        or {}
    )

    if not current:
        return {
            "available": False,
            "reason":
                "No marine conditions returned.",
        }

    usable_values = [
        current.get("wave_height"),
        current.get("wave_direction"),
        current.get("wave_period"),
        current.get("swell_wave_height"),
        current.get("swell_wave_direction"),
        current.get("swell_wave_period"),
        current.get("sea_surface_temperature"),
        current.get("ocean_current_velocity"),
        current.get("ocean_current_direction"),
    ]

    if all(
        value is None
        for value in usable_values
    ):
        return {
            "available": False,
            "reason": (
                "Marine grid found, but no usable "
                "ocean conditions are available "
                "at this location."
            ),
            "grid_distance_km":
                _round(
                    grid_distance_km,
                    2,
                ),
        }

    return {
        "available": True,

        "observation_time":
            current.get("time"),

        "grid_distance_km":
            _round(
                grid_distance_km,
                2,
            ),

        "wave_height_m":
            _round(
                current.get(
                    "wave_height"
                ),
                2,
            ),

        "wave_direction_degrees":
            _round(
                current.get(
                    "wave_direction"
                ),
                1,
            ),

        "wave_period_seconds":
            _round(
                current.get(
                    "wave_period"
                ),
                1,
            ),

        "swell_height_m":
            _round(
                current.get(
                    "swell_wave_height"
                ),
                2,
            ),

        "swell_direction_degrees":
            _round(
                current.get(
                    "swell_wave_direction"
                ),
                1,
            ),

        "swell_period_seconds":
            _round(
                current.get(
                    "swell_wave_period"
                ),
                1,
            ),

        "sea_surface_temperature_c":
            _round(
                current.get(
                    "sea_surface_temperature"
                ),
                1,
            ),

        "ocean_current_speed_knots":
            _round(
                _speed_to_knots(
                    current.get(
                        "ocean_current_velocity"
                    ),
                    units.get(
                        "ocean_current_velocity"
                    ),
                ),
                2,
            ),

        "ocean_current_direction_degrees":
            _round(
                current.get(
                    "ocean_current_direction"
                ),
                1,
            ),
    }


# ============================================================
# PUBLIC SERVICE
# ============================================================

def get_environment_snapshot(
    latitude: float,
    longitude: float,
) -> dict:

    latitude = float(latitude)
    longitude = float(longitude)

    # Cache by approximately 1 km.
    cache_key = (
        round(latitude, 2),
        round(longitude, 2),
    )

    now = time.time()

    with _CACHE_LOCK:
        cached = _CACHE.get(
            cache_key
        )

        if (
            cached
            and now
            - cached["cached_at"]
            < CACHE_TTL_SECONDS
        ):
            return dict(
                cached["snapshot"]
            )

    snapshot = {
        "provider":
            "Open-Meteo",

        "retrieved_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "latitude":
            latitude,

        "longitude":
            longitude,

        "context_mode":
            (
                "current_environment_"
                "at_simulated_position"
            ),

        "used_for_model_inference":
            False,

        "used_for_collision_risk":
            False,

        "attribution":
            "Weather data by Open-Meteo",
    }

    try:
        snapshot[
            "atmospheric"
        ] = _fetch_weather(
            latitude,
            longitude,
        )

    except Exception as error:
        snapshot[
            "atmospheric"
        ] = {
            "available": False,
            "error": str(error),
        }

    try:
        snapshot[
            "marine"
        ] = _fetch_marine(
            latitude,
            longitude,
        )

    except Exception as error:
        snapshot[
            "marine"
        ] = {
            "available": False,
            "error": str(error),
        }

    snapshot["available"] = (
        snapshot[
            "atmospheric"
        ].get("available", False)
        or snapshot[
            "marine"
        ].get("available", False)
    )

    with _CACHE_LOCK:
        _CACHE[
            cache_key
        ] = {
            "cached_at": now,
            "snapshot": snapshot,
        }

    return dict(snapshot)
