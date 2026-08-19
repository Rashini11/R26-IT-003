from __future__ import annotations

import math
from typing import Any

from .config import METRES_PER_NAUTICAL_MILE, RISK_THRESHOLDS, RiskThresholds
from .geo import bearing_degrees, local_xy_metres, velocity_components


def _risk_label(
    dcpa_metres: float,
    tcpa_seconds: float | None,
    thresholds: RiskThresholds,
) -> tuple[str, list[str]]:
    dcpa_nm = dcpa_metres / METRES_PER_NAUTICAL_MILE
    reasons: list[str] = []
    if tcpa_seconds is None:
        return "Low", ["Relative velocity is too small for a meaningful TCPA."]
    tcpa_minutes = tcpa_seconds / 60.0
    if tcpa_seconds < 0:
        return "Low", ["Closest approach is in the past; vessels are not on a forward closing solution."]
    if tcpa_minutes > thresholds.assessment_horizon_minutes:
        return "Low", ["Closest approach is beyond the configured assessment horizon."]
    if (
        tcpa_minutes <= thresholds.critical_tcpa_minutes
        and dcpa_nm < thresholds.critical_dcpa_nm
    ):
        reasons.extend([
            f"DCPA is below {thresholds.critical_dcpa_nm:.1f} NM.",
            f"TCPA is within {thresholds.critical_tcpa_minutes:.0f} minutes.",
        ])
        return "Critical", reasons
    if tcpa_minutes <= thresholds.high_tcpa_minutes and dcpa_nm < thresholds.high_dcpa_nm:
        reasons.extend([
            f"DCPA is below {thresholds.high_dcpa_nm:.1f} NM.",
            f"TCPA is within {thresholds.high_tcpa_minutes:.0f} minutes.",
        ])
        return "High", reasons
    if (
        tcpa_minutes <= thresholds.medium_tcpa_minutes
        and dcpa_nm < thresholds.medium_dcpa_nm
    ):
        reasons.extend([
            f"DCPA is below {thresholds.medium_dcpa_nm:.1f} NM.",
            f"TCPA is within {thresholds.medium_tcpa_minutes:.0f} minutes.",
        ])
        return "Medium", reasons
    return "Low", ["DCPA/TCPA do not cross the configured research-warning thresholds."]


def calculate_collision_risk(
    own_vessel: dict[str, Any],
    target_vessel: dict[str, Any],
    thresholds: RiskThresholds = RISK_THRESHOLDS,
) -> dict[str, Any]:
    east, north = local_xy_metres(
        [target_vessel["latitude"]],
        [target_vessel["longitude"]],
        own_vessel["latitude"],
        own_vessel["longitude"],
    )
    relative_position = (float(east[0]), float(north[0]))
    separation = math.hypot(*relative_position)

    own_velocity = velocity_components(
        float(own_vessel["speed_knots"]), float(own_vessel["course_degrees"])
    )
    target_velocity = velocity_components(
        float(target_vessel["speed_knots"]), float(target_vessel["course_degrees"])
    )
    relative_velocity = (
        target_velocity[0] - own_velocity[0],
        target_velocity[1] - own_velocity[1],
    )
    velocity_squared = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    radial_dot = (
        relative_position[0] * relative_velocity[0]
        + relative_position[1] * relative_velocity[1]
    )

    if velocity_squared < 1e-8:
        tcpa_seconds = None
        dcpa_metres = separation
        movement_relationship = "stable"
    else:
        raw_tcpa = -radial_dot / velocity_squared
        tcpa_seconds = float(raw_tcpa)
        closest_east = relative_position[0] + relative_velocity[0] * raw_tcpa
        closest_north = relative_position[1] + relative_velocity[1] * raw_tcpa
        dcpa_metres = math.hypot(closest_east, closest_north)
        closing_speed = -radial_dot / max(separation, 1e-9)
        if closing_speed > 0.05:
            movement_relationship = "approaching"
        elif closing_speed < -0.05:
            movement_relationship = "receding"
        else:
            movement_relationship = "stable"

    risk_level, reasons = _risk_label(dcpa_metres, tcpa_seconds, thresholds)
    relative_speed_mps = math.hypot(*relative_velocity)

    return {
        "current_separation_metres": round(separation, 2),
        "current_separation_nautical_miles": round(
            separation / METRES_PER_NAUTICAL_MILE, 4
        ),
        "relative_speed_knots": round(
            relative_speed_mps * 3600.0 / METRES_PER_NAUTICAL_MILE, 3
        ),
        "relative_bearing_degrees": round(
            bearing_degrees(*relative_position), 2
        ),
        "movement_relationship": movement_relationship,
        "dcpa_metres": round(dcpa_metres, 2),
        "dcpa_nautical_miles": round(dcpa_metres / METRES_PER_NAUTICAL_MILE, 4),
        "tcpa_seconds": None if tcpa_seconds is None else round(tcpa_seconds, 2),
        "tcpa_minutes": None if tcpa_seconds is None else round(tcpa_seconds / 60.0, 2),
        "risk_level": risk_level,
        "risk_reasons": reasons,
        "thresholds_used": {
            "critical": {
                "dcpa_nautical_miles_below": thresholds.critical_dcpa_nm,
                "tcpa_minutes_within": thresholds.critical_tcpa_minutes,
            },
            "high": {
                "dcpa_nautical_miles_below": thresholds.high_dcpa_nm,
                "tcpa_minutes_within": thresholds.high_tcpa_minutes,
            },
            "medium": {
                "dcpa_nautical_miles_below": thresholds.medium_dcpa_nm,
                "tcpa_minutes_within": thresholds.medium_tcpa_minutes,
            },
            "assessment_horizon_minutes": thresholds.assessment_horizon_minutes,
        },
        "research_use_notice": (
            "Configurable prototype thresholds; not universal operational navigation rules."
        ),
    }
