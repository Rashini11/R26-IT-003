from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

import backend.simulation.ais_controller as ais_module


def write_test_ais(path):
    start = datetime(2022, 3, 31, 0, 0, 0)
    rows = []
    for index in range(31):
        timestamp = start + timedelta(minutes=index)
        rows.append({
            "MMSI": 111111111,
            "BaseDateTime": timestamp.isoformat(sep=" "),
            "LAT": 6.0000,
            "LON": 79.0000 + index * 0.0015,
            "SOG": 8.0,
            "COG": 90.0,
        })
        rows.append({
            "MMSI": 222222222,
            "BaseDateTime": timestamp.isoformat(sep=" "),
            "LAT": 6.0050,
            "LON": 79.0400 - index * 0.0012,
            "SOG": 7.0,
            "COG": 270.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_constructed_scenario_uses_two_mmsi_and_ten_second_ticks(tmp_path, monkeypatch):
    csv_path = tmp_path / "ais.csv"
    write_test_ais(csv_path)
    monkeypatch.setattr(ais_module, "AIS_CSV_PATH", csv_path)
    controller = ais_module.AISSimulationController()

    scenario = controller.create_scenario(
        mode="constructed",
        interval_seconds=10,
        duration_minutes=15,
        seed=42,
    )

    assert scenario.mode == "constructed"
    assert scenario.length == 91
    assert scenario.own_states[0]["mmsi"] != scenario.target_states[0]["mmsi"]
    gap = scenario.own_states[1]["timestamp"] - scenario.own_states[0]["timestamp"]
    assert gap.total_seconds() == 10


def test_actual_scenario_uses_overlapping_historical_times(tmp_path, monkeypatch):
    csv_path = tmp_path / "ais.csv"
    write_test_ais(csv_path)
    monkeypatch.setattr(ais_module, "AIS_CSV_PATH", csv_path)
    controller = ais_module.AISSimulationController()

    scenario = controller.create_scenario(
        mode="actual",
        interval_seconds=10,
        duration_minutes=15,
        seed=42,
    )

    assert scenario.mode == "actual"
    assert scenario.length == 91
    assert scenario.own_states[0]["timestamp"] == scenario.target_states[0]["timestamp"]
