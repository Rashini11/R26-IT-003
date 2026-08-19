from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from backend.simulation.geo import KNOT_TO_METRES_PER_SECOND, local_xy_metres
from backend.simulation.risk_engine import calculate_collision_risk
from backend.simulation.sar_streamer import resolve_sar_source
from backend.simulation.schemas import MotionPredictionRequest


def vessel(mmsi, latitude, longitude, speed, course):
    return {
        "mmsi": mmsi,
        "timestamp": datetime.now(timezone.utc),
        "latitude": latitude,
        "longitude": longitude,
        "speed_knots": speed,
        "course_degrees": course,
    }


def observations(count=10, second_mmsi=False):
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "mmsi": 222 if second_mmsi and index == count - 1 else 111,
            "timestamp": start + timedelta(minutes=index),
            "latitude": 6.0 + index * 0.001,
            "longitude": 79.0,
            "speed_knots": 8.0,
            "course_degrees": 0.0,
            "observed": True,
        }
        for index in range(count)
    ]


def test_knots_conversion_constant():
    assert KNOT_TO_METRES_PER_SECOND == pytest.approx(0.5144444444, rel=1e-6)


def test_local_coordinates_increase_eastward():
    east, north = local_xy_metres([0.0], [0.01], 0.0, 0.0)
    assert east[0] > 1000
    assert abs(north[0]) < 1e-6


def test_head_on_solution_has_forward_tcpa_and_small_dcpa():
    own = vessel(1, 0.0, 0.0, 10.0, 90.0)
    target = vessel(2, 0.0, 0.05, 10.0, 270.0)
    result = calculate_collision_risk(own, target)
    assert result["tcpa_seconds"] is not None
    assert result["tcpa_seconds"] > 0
    assert result["dcpa_metres"] < 10
    assert result["movement_relationship"] == "approaching"


def test_receding_solution_is_low_risk():
    own = vessel(1, 0.0, 0.0, 10.0, 270.0)
    target = vessel(2, 0.0, 0.05, 10.0, 90.0)
    result = calculate_collision_risk(own, target)
    assert result["tcpa_seconds"] is not None
    assert result["tcpa_seconds"] < 0
    assert result["risk_level"] == "Low"


def test_sar_source_does_not_accept_arbitrary_path():
    with pytest.raises(ValueError):
        resolve_sar_source("../../etc")


def test_motion_request_requires_ten_rows():
    with pytest.raises(ValidationError):
        MotionPredictionRequest(observations=observations(9))


def test_motion_request_rejects_mixed_mmsi():
    with pytest.raises(ValidationError):
        MotionPredictionRequest(observations=observations(10, second_mmsi=True))


def test_motion_request_accepts_valid_history():
    request = MotionPredictionRequest(observations=observations())
    assert len(request.observations) == 10


def test_risk_threshold_boundary_is_strict():
    from backend.simulation.config import METRES_PER_NAUTICAL_MILE, RISK_THRESHOLDS
    from backend.simulation.risk_engine import _risk_label

    just_below = RISK_THRESHOLDS.critical_dcpa_nm * METRES_PER_NAUTICAL_MILE - 0.01
    at_boundary = RISK_THRESHOLDS.critical_dcpa_nm * METRES_PER_NAUTICAL_MILE

    assert _risk_label(just_below, 5 * 60, RISK_THRESHOLDS)[0] == "Critical"
    assert _risk_label(at_boundary, 5 * 60, RISK_THRESHOLDS)[0] == "High"


def test_six_ten_second_ticks_are_aggregated_into_one_minute():
    from backend.simulation.encounter_controller import aggregate_minute_observations

    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    ticks = [
        {
            "mmsi": 111,
            "timestamp": start + timedelta(seconds=index * 10),
            "latitude": 6.0 + index * 0.001,
            "longitude": 79.0,
            "speed_knots": float(index),
            "course_degrees": 90.0,
            "observed": index == 3,
            "course_missing": index == 4,
        }
        for index in range(6)
    ]

    result = aggregate_minute_observations(ticks, ticks_per_minute=6)

    assert len(result) == 1
    assert result[0]["speed_knots"] == pytest.approx(2.5)
    assert result[0]["latitude"] == pytest.approx(6.0025)
    assert result[0]["observed"] is True
    assert result[0]["course_missing"] is True
    assert result[0]["course_degrees"] == pytest.approx(90.0)


def test_incomplete_minute_is_not_sent_to_gru():
    from backend.simulation.encounter_controller import aggregate_minute_observations

    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    ticks = [
        {
            "mmsi": 111,
            "timestamp": start + timedelta(seconds=index * 10),
            "latitude": 6.0,
            "longitude": 79.0,
            "speed_knots": 5.0,
            "course_degrees": 90.0,
            "observed": True,
        }
        for index in range(5)
    ]
    assert aggregate_minute_observations(ticks, ticks_per_minute=6) == []
