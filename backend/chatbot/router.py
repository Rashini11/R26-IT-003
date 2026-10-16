from __future__ import annotations

import asyncio
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..auth import require_authenticated_request
from .image_utils import validate_image_bytes
from .intents import DEFAULT_ACTIONS, match_intent, unsupported_text_message
from .routing import DEFAULT_ROUTER, SUPPORTED_COMPONENTS
from .runners import available_runners, get_runner
from .schemas import (
    ChatAnalyzeResponse,
    ChatFollowUpRequest,
    ChatFollowUpResponse,
    ChatIntentRequest,
    ChatIntentResponse,
)
from .templates import (
    build_follow_up,
    build_hull_response,
    build_radar_response,
    build_sea_response,
    build_vessel_response,
    welcome_payload,
)


router = APIRouter(prefix="/chat", tags=["Chat Assistant"])

# Limit concurrent heavy inference on shared CPU/GPU resources.
_INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

TEMP_DIR = Path(__file__).resolve().parents[1] / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _response_from_raw(component: str, raw: dict, result_id: str) -> dict:
    if component == "radar":
        return build_radar_response(raw, result_id)
    if component == "vessel":
        return build_vessel_response(raw, result_id)
    if component == "sea":
        return build_sea_response(raw, result_id)
    if component == "hull":
        return build_hull_response(raw, result_id)
    return {
        "analysis_id": component,
        "component": component,
        "status": "unsupported",
        "summary": "Unsupported analysis component.",
        "model_outputs": {},
        "visualisation": {},
        "follow_up_actions": DEFAULT_ACTIONS,
        "result_id": result_id,
        "error": "Unsupported component",
        "limitations": [],
    }


@router.get("/welcome")
def chat_welcome(_user=Depends(require_authenticated_request)):
    payload = welcome_payload()
    payload["runners"] = available_runners()
    payload["routing"] = DEFAULT_ROUTER.resolve(None).to_dict()
    return payload


@router.post("/analyze", response_model=ChatAnalyzeResponse)
async def chat_analyze(
    file: UploadFile = File(...),
    analysis: Optional[str] = Form(None),
    apply_enhancement: bool = Form(False),
    _user=Depends(require_authenticated_request),
):
    result_id = str(uuid.uuid4())
    raw_path = None

    try:
        contents = await file.read()
        validation = validate_image_bytes(contents, file.filename)
        if not validation.ok:
            routing = DEFAULT_ROUTER.resolve(analysis).to_dict()
            return ChatAnalyzeResponse(
                analysis_id=routing.get("selected_component"),
                component=(
                    SUPPORTED_COMPONENTS.get(routing["selected_component"], {}).get("label")
                    if routing.get("selected_component")
                    else None
                ),
                status="validation_error",
                summary=validation.reason,
                validation={
                    "is_valid": False,
                    "reason": validation.reason,
                    "width": validation.width,
                    "height": validation.height,
                    "format": validation.format,
                },
                routing=routing,
                result_id=result_id,
                follow_up_actions=[
                    {"id": "analyse_another", "label": "Analyse another image"},
                    {"id": "change_analysis", "label": "Change analysis type"},
                ],
                limitations=["Upload a readable JPEG, PNG, BMP, or WEBP image."],
                error=validation.reason,
            )

        routing_decision = DEFAULT_ROUTER.resolve(
            selected_component=analysis,
            image_bytes=validation.rgb_bytes,
        )
        routing = routing_decision.to_dict()

        if routing_decision.status.value == "needs_selection":
            return ChatAnalyzeResponse(
                status="needs_selection",
                summary=routing_decision.message,
                routing=routing,
                result_id=result_id,
                follow_up_actions=[
                    {"id": key, "label": meta["label"]}
                    for key, meta in SUPPORTED_COMPONENTS.items()
                ],
                limitations=[
                    "Automatic image-type routing is unavailable in this version.",
                ],
            )

        if routing_decision.status.value == "unsupported":
            return ChatAnalyzeResponse(
                status="unsupported",
                summary=routing_decision.message,
                routing=routing,
                result_id=result_id,
                follow_up_actions=[
                    {"id": key, "label": meta["label"]}
                    for key, meta in SUPPORTED_COMPONENTS.items()
                ],
                error=routing_decision.message,
            )

        component = routing_decision.selected_component
        runner = get_runner(component)
        if runner is None:
            return ChatAnalyzeResponse(
                analysis_id=component,
                component=SUPPORTED_COMPONENTS[component]["label"],
                status="model_unavailable",
                summary=(
                    f"The {SUPPORTED_COMPONENTS[component]['label']} model is not loaded "
                    "on this server."
                ),
                routing=routing,
                result_id=result_id,
                follow_up_actions=[
                    {"id": "change_analysis", "label": "Change analysis type"},
                    {"id": "analyse_another", "label": "Analyse another image"},
                ],
                error="Model unavailable",
            )

        safe_name = Path(file.filename or "upload.jpg").name
        raw_path = TEMP_DIR / f"chat_{uuid.uuid4()}_{safe_name}"
        raw_path.write_bytes(validation.rgb_bytes or contents)

        async with _INFERENCE_SEMAPHORE:
            if component == "sea":
                raw = await asyncio.to_thread(
                    runner,
                    image_path=raw_path,
                    filename=safe_name,
                    apply_enhancement=apply_enhancement,
                    persist=True,
                    source="chatbot",
                )
            elif component == "radar":
                raw = await asyncio.to_thread(
                    runner,
                    image_path=raw_path,
                    display_name=safe_name,
                    persist=True,
                    source="chatbot",
                )
            elif component == "vessel":
                raw = await asyncio.to_thread(
                    runner,
                    image_path=raw_path,
                    filename=safe_name,
                    persist=True,
                    source="chatbot",
                    include_annotation=True,
                )
            else:  # hull
                raw = await asyncio.to_thread(
                    runner,
                    image_bytes=validation.rgb_bytes or contents,
                    filename=safe_name,
                    persist=True,
                    source="chatbot",
                )

        payload = _response_from_raw(component, raw, result_id)
        payload["routing"] = routing
        return ChatAnalyzeResponse(**payload)

    except Exception:
        traceback.print_exc()
        return ChatAnalyzeResponse(
            status="inference_error",
            summary="Analysis failed while processing the image.",
            error="Analysis failed. Please try again with another image.",
            result_id=result_id,
            follow_up_actions=[
                {"id": "analyse_another", "label": "Analyse another image"},
                {"id": "change_analysis", "label": "Change analysis type"},
            ],
            limitations=["Internal details are not exposed for security."],
        )
    finally:
        if raw_path is not None:
            Path(raw_path).unlink(missing_ok=True)


@router.post("/follow-up", response_model=ChatFollowUpResponse)
async def chat_follow_up(
    payload: ChatFollowUpRequest,
    _user=Depends(require_authenticated_request),
):
    action = payload.action.strip().lower()
    # Allow selecting a component via follow-up button ids.
    if action in SUPPORTED_COMPONENTS:
        meta = SUPPORTED_COMPONENTS[action]
        return ChatFollowUpResponse(
            status="success",
            summary=(
                f"Analysis set to {meta['label']}. "
                "Upload or keep your image, then submit to run this component."
            ),
            details={"selected_component": action, "analysis_choices": [meta]},
            follow_up_actions=[
                {"id": "analyse_another", "label": "Upload / analyse image"},
                {"id": "help", "label": "What can you do?"},
            ],
        )

    if payload.text and action in {"text", "message", "intent"}:
        intent = match_intent(payload.text)
        if not intent:
            return ChatFollowUpResponse(
                status="unsupported",
                summary=unsupported_text_message(),
                details={},
                follow_up_actions=DEFAULT_ACTIONS,
            )
        action = intent

    result = build_follow_up(action, payload.result_payload or {})
    return ChatFollowUpResponse(**result)


@router.post("/intent", response_model=ChatIntentResponse)
async def chat_intent(
    payload: ChatIntentRequest,
    _user=Depends(require_authenticated_request),
):
    intent = match_intent(payload.text)
    if not intent:
        return ChatIntentResponse(
            matched=False,
            intent=None,
            message=unsupported_text_message(),
            follow_up_actions=DEFAULT_ACTIONS,
        )
    follow = build_follow_up(intent, {})
    return ChatIntentResponse(
        matched=True,
        intent=intent,
        message=follow["summary"],
        follow_up_actions=follow.get("follow_up_actions") or DEFAULT_ACTIONS,
    )


# OCEANIQ_IMAGE_ROUTER_V1
@router.post("/route")
async def chat_route_image(
    file: UploadFile = File(...),
    _user=Depends(require_authenticated_request),
):
    from .image_router_model import suggest_image_component

    try:
        contents = await file.read()
        validation = validate_image_bytes(contents, file.filename)

        if not validation.ok:
            return {
                "status": "validation_error",
                "automatic_routing_available": False,
                "message": validation.reason,
            }

        async with _INFERENCE_SEMAPHORE:
            return await asyncio.to_thread(
                suggest_image_component,
                validation.rgb_bytes or contents,
            )
    except Exception:
        traceback.print_exc()
        return {
            "status": "unavailable",
            "automatic_routing_available": False,
            "message": "Could not suggest an analysis. Please select one manually.",
        }
    finally:
        await file.close()

