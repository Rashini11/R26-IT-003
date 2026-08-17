from __future__ import annotations

import math
from typing import Any


def aggregate_minute_observations(
    tick_states: list[dict[str, Any]],
    ticks_per_minute: int,
    max_minutes: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate complete fixed-rate tick groups into one-minute AIS observations.

    This mirrors the training preparation more closely than taking every Nth tick:
    latitude, longitude and SOG are averaged; course is averaged as a circular
    quantity; `observed` is true when any source report occurred during the
    minute; and `course_missing` is true when any contributing tick carried a
    missing-course indicator.
    """
    if ticks_per_minute <= 0:
        raise ValueError("ticks_per_minute must be positive.")

    complete_groups = len(tick_states) // ticks_per_minute
    observations: list[dict[str, Any]] = []

    for group_index in range(complete_groups):
        start = group_index * ticks_per_minute
        chunk = tick_states[start : start + ticks_per_minute]
        mmsi_values = {int(item["mmsi"]) for item in chunk}
        if len(mmsi_values) != 1:
            raise ValueError("A minute aggregation cannot mix different MMSIs.")

        sin_values = []
        cos_values = []
        for item in chunk:
            course = item.get("course_degrees")
            if course is None:
                continue
            radians = math.radians(float(course) % 360.0)
            sin_values.append(math.sin(radians))
            cos_values.append(math.cos(radians))

        if sin_values and cos_values:
            course_degrees = (
                math.degrees(
                    math.atan2(
                        sum(sin_values) / len(sin_values),
                        sum(cos_values) / len(cos_values),
                    )
                )
                + 360.0
            ) % 360.0
        else:
            course_degrees = None

        observations.append(
            {
                "mmsi": next(iter(mmsi_values)),
                "timestamp": chunk[0]["timestamp"],
                "latitude": sum(float(item["latitude"]) for item in chunk)
                / len(chunk),
                "longitude": sum(float(item["longitude"]) for item in chunk)
                / len(chunk),
                "speed_knots": sum(float(item["speed_knots"]) for item in chunk)
                / len(chunk),
                "course_degrees": course_degrees,
                "observed": any(bool(item.get("observed", True)) for item in chunk),
                "course_missing": any(
                    bool(
                        item.get(
                            "course_missing",
                            item.get("course_degrees") is None,
                        )
                    )
                    for item in chunk
                ),
            }
        )

    if max_minutes is not None:
        return observations[-max_minutes:]
    return observations
