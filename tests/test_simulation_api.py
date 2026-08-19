from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import backend.simulation.app as simulation_app
from backend.simulation.ais_controller import AISScenario


class DummyThread:
    def __init__(self, *args, **kwargs):
        self.started = False

    def start(self):
        self.started = True


def make_state(mmsi: int, second: int):
    return {
        "mmsi": mmsi,
        "timestamp": datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=second),
        "latitude": 6.0,
        "longitude": 79.0,
        "speed_knots": 5.0,
        "course_degrees": 90.0,
        "observed": True,
    }


def test_start_conflict_and_stop(monkeypatch):
    scenario = AISScenario(
        scenario_id="test-scenario",
        mode="constructed",
        description="test",
        own_states=[make_state(1, 0)],
        target_states=[make_state(2, 0)],
    )
    monkeypatch.setattr(
        simulation_app.AIS_CONTROLLER,
        "create_scenario",
        lambda **kwargs: scenario,
    )
    monkeypatch.setattr(simulation_app, "SARImageStreamer", lambda *args, **kwargs: object())
    monkeypatch.setattr(simulation_app.threading, "Thread", DummyThread)

    with simulation_app.STATE.lock:
        simulation_app.STATE.running = False
        simulation_app.STATE.starting = False
        simulation_app.STATE.stop_event.clear()

    client = TestClient(simulation_app.app)
    request = {
        "sar_source": "ship",
        "mode": "constructed",
        "real_interval_seconds": 1,
        "simulated_interval_seconds": 10,
        "loop": True,
        "seed": 42,
        "scenario_minutes": 20,
    }

    first = client.post("/simulation/start", json=request)
    assert first.status_code == 200

    second = client.post("/simulation/start", json=request)
    assert second.status_code == 409

    stopped = client.post("/simulation/stop")
    assert stopped.status_code == 200
    assert simulation_app.STATE.stop_event.is_set()

    with simulation_app.STATE.lock:
        simulation_app.STATE.running = False
        simulation_app.STATE.starting = False
        simulation_app.STATE.stop_event.clear()


def test_worker_emits_combined_event(tmp_path):
    image_path = tmp_path / "ship.png"
    image_path.write_bytes(b"not-decoded-by-the-fake-streamer")

    class FakeStreamer:
        def next_image(self):
            return image_path

        def classify(self, _image_path):
            return {
                "final_prediction": "ship",
                "yolo_prediction": "ship",
                "yolo_confidence": 99.0,
                "cnn_prediction": "ship",
                "cnn_confidence": 95.0,
                "decision_status": "High confidence - both models agree",
            }

    scenario = AISScenario(
        scenario_id="combined-event-test",
        mode="constructed",
        description="test",
        own_states=[make_state(1, 0)],
        target_states=[{
            **make_state(2, 0),
            "longitude": 79.01,
            "course_degrees": 270.0,
        }],
    )
    request = simulation_app.SimulationStartRequest(
        sar_source="ship",
        mode="constructed",
        real_interval_seconds=0.2,
        simulated_interval_seconds=10,
        loop=False,
        seed=42,
        scenario_minutes=15,
    )

    simulation_app.STATE.reset()
    with simulation_app.STATE.lock:
        simulation_app.STATE.running = True
    simulation_app._run_simulation(request, scenario, FakeStreamer())

    event = simulation_app.STATE.latest
    assert event is not None
    assert event["sar"]["classification"] == "ship"
    assert event["own_vessel"]["current_state"]["mmsi"] == 1
    assert event["target_vessel"]["current_state"]["mmsi"] == 2
    assert "dcpa_metres" in event["encounter"]
    assert event["running"] is False
