"""Chat Assistant orchestration tests (mocked model outputs — not model evaluation)."""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

from backend.chatbot import intents, routing, templates
from backend.chatbot.image_utils import validate_image_bytes
from backend.chatbot.runners import register_component_runners


def _auth_ok(request: Request):
    return {
        "id": "test-user",
        "username": "tester",
        "role": "admin",
        "approval_status": "approved",
        "access_level": "read_write",
    }


def _png_bytes(color=(20, 80, 140), size=(64, 64)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def chat_client():
    from backend.auth import require_authenticated_request
    from backend.chatbot.router import router as chat_router

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[require_authenticated_request] = _auth_ok

    register_component_runners(
        radar=lambda **kwargs: {
            "final_prediction": "ship",
            "binary_prediction": "ship",
            "confidence": 88.5,
            "bird_probability": 5.0,
            "ship_probability": 88.5,
            "unknown_probability": 6.5,
            "decision_status": "accepted",
            "model_name": "RadarTargetCNN",
            "model_version": "radar_unknown3_v1",
        },
        vessel=lambda **kwargs: {
            "status": "Detected",
            "confidence": 91.0,
            "count": 1,
            "vessel_origin": "Local Boat",
            "estimated_size": "Medium Vessel",
            "results": [
                {
                    "label": "Local Boat",
                    "confidence": 91.0,
                    "box": [10, 10, 40, 40],
                    "detection_type": "boat",
                }
            ],
            "annotated_image": None,
        },
        sea=lambda **kwargs: {
            "predicted_sea_state": "moderate",
            "confidence": 77.2,
            "probabilities": {
                "calm": 10.0,
                "moderate": 77.2,
                "rough": 8.0,
                "very_rough": 4.8,
            },
            "recommendation": "Proceed with caution.",
            "warnings": [],
            "validation": {"is_valid": True},
        },
        hull=lambda **kwargs: {
            "prediction": "corrosion",
            "confidence": 82.0,
            "valid_hull_image": True,
            "detected_defects": [
                {"defect": "corrosion", "confidence": 82.0},
                {"defect": "paint_damage", "confidence": 12.0},
            ],
            "multiple_defects": False,
            "recommendation": "Apply anti-corrosion coating.",
            "warning": "Prediction reliable.",
            "gradcam": None,
        },
    )

    with TestClient(app) as client:
        yield client


def test_router_needs_selection():
    decision = routing.DEFAULT_ROUTER.resolve(None)
    assert decision.status == routing.RoutingStatus.NEEDS_SELECTION
    assert decision.automatic_routing_available is False


def test_router_selected_and_unsupported():
    selected = routing.DEFAULT_ROUTER.resolve("radar")
    assert selected.status == routing.RoutingStatus.SELECTED
    assert selected.selected_component == "radar"

    bad = routing.DEFAULT_ROUTER.resolve("ais_only")
    assert bad.status == routing.RoutingStatus.UNSUPPORTED


def test_invalid_image_bytes():
    result = validate_image_bytes(b"not-an-image")
    assert result.ok is False


def test_analyze_needs_selection(chat_client):
    response = chat_client.post(
        "/chat/analyze",
        files={"file": ("sea.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_selection"
    assert body["routing"]["automatic_routing_available"] is False


def test_analyze_invalid_image(chat_client):
    response = chat_client.post(
        "/chat/analyze",
        data={"analysis": "radar"},
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "validation_error"
    assert body["validation"]["is_valid"] is False


def test_analyze_unsupported_component(chat_client):
    response = chat_client.post(
        "/chat/analyze",
        data={"analysis": "not_a_real_component"},
        files={"file": ("sea.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unsupported"


def test_analyze_dispatch_radar_mocked(chat_client):
    response = chat_client.post(
        "/chat/analyze",
        data={"analysis": "radar"},
        files={"file": ("radar.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["analysis_id"] == "radar"
    assert body["model_outputs"]["final_prediction"] == "ship"
    assert body["visualisation"]["probability_chart"]["items"]


def test_template_maps_inference_failure():
    payload = templates.build_sea_response({"error": "boom"}, "rid-1")
    assert payload["status"] == "inference_error"
    assert payload["result_id"] == "rid-1"


def test_follow_up_uses_selected_result(chat_client):
    result_payload = templates.build_radar_response(
        {
            "final_prediction": "unknown",
            "confidence": 41.0,
            "bird_probability": 20.0,
            "ship_probability": 39.0,
            "unknown_probability": 41.0,
            "model_version": "radar_unknown3_v1",
        },
        "abc",
    )
    response = chat_client.post(
        "/chat/follow-up",
        json={"action": "explain_unknown", "result_payload": result_payload},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "unknown" in body["summary"].lower()
    assert body["details"]["analysis_id"] == "radar"


def test_intent_matching_and_unsupported_text():
    assert intents.match_intent("please explain confidence") == "explain_confidence"
    assert intents.match_intent("tell me a joke about ships") is None


def test_welcome(chat_client):
    response = chat_client.get("/chat/welcome")
    assert response.status_code == 200
    body = response.json()
    assert "Welcome" in body["summary"]
    assert len(body["analysis_choices"]) == 4
