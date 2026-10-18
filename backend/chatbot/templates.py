from __future__ import annotations

from typing import Any

from .intents import DEFAULT_ACTIONS
from .routing import SUPPORTED_COMPONENTS


def _confidence_clause(confidence: Any) -> str:
    if confidence is None:
        return ""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return ""
    return (
        f" The model score is {value:.2f}% "
        "(a model confidence score, not a guarantee of correctness)."
    )


def welcome_payload() -> dict[str, Any]:
    choices = [
        {
            "id": key,
            "label": meta["label"],
            "description": meta["description"],
        }
        for key, meta in SUPPORTED_COMPONENTS.items()
    ]
    return {
        "summary": (
            "Welcome to the OceanIQ Chat Assistant. "
            "Upload a maritime image and choose an analysis. "
            "Automatic image-type recognition is not available in this version — "
            "you select the component yourself. "
            "AIS motion forecasting and collision-risk simulation need movement "
            "data and runtime assets; they are not derived from a still image."
        ),
        "analysis_choices": choices,
        "follow_up_actions": [
            {"id": "help", "label": "What can you do?"},
            {"id": "change_analysis", "label": "Choose analysis type"},
        ],
        "limitations": [
            "No automatic image-type routing in this version.",
            "No LLM or free-form natural-language understanding.",
            "Chat messages stay in this browser session only.",
        ],
    }


def build_radar_response(raw: dict[str, Any], result_id: str) -> dict[str, Any]:
    if raw.get("error"):
        return {
            "analysis_id": "radar",
            "component": "Radar Objects",
            "status": "inference_error",
            "summary": "Radar analysis could not be completed.",
            "model_outputs": {},
            "error": "Radar analysis failed. Please try another image.",
            "follow_up_actions": [
                {"id": "analyse_another", "label": "Analyse another image"},
                {"id": "change_analysis", "label": "Change analysis type"},
            ],
            "result_id": result_id,
            "limitations": [],
            "visualisation": {},
        }

    label = raw.get("final_prediction")
    confidence = raw.get("confidence")
    summary = (
        f"Radar classification result: {label}."
        f"{_confidence_clause(confidence)}"
    )
    chart = {
        "type": "probability_bars",
        "title": "Class model scores",
        "unit": "percent",
        "items": [
            {"label": "bird", "value": raw.get("bird_probability")},
            {"label": "ship", "value": raw.get("ship_probability")},
            {"label": "unknown", "value": raw.get("unknown_probability")},
        ],
    }
    # Drop items with missing values rather than fabricating them.
    chart["items"] = [
        item for item in chart["items"] if isinstance(item["value"], (int, float))
    ]

    limitations = []
    if label == "unknown":
        limitations.append(
            "The model predicted the class 'unknown'. This is a radar class output, "
            "not a rejection of the analysis type."
        )

    return {
        "analysis_id": "radar",
        "component": "Radar Objects",
        "status": "success",
        "summary": summary,
        "model_outputs": {
            "final_prediction": label,
            "binary_prediction": raw.get("binary_prediction"),
            "confidence": confidence,
            "bird_probability": raw.get("bird_probability"),
            "ship_probability": raw.get("ship_probability"),
            "unknown_probability": raw.get("unknown_probability"),
            "decision_status": raw.get("decision_status"),
        },
        "model_version": raw.get("model_version") or raw.get("model_name"),
        "visualisation": {"probability_chart": chart} if chart["items"] else {},
        "follow_up_actions": DEFAULT_ACTIONS,
        "validation": None,
        "limitations": limitations,
        "result_id": result_id,
        "error": None,
    }


def build_sea_response(raw: dict[str, Any], result_id: str) -> dict[str, Any]:
    if raw.get("error"):
        validation = raw.get("validation")
        status = "validation_error" if validation and not validation.get("is_valid", True) else "inference_error"
        return {
            "analysis_id": "sea",
            "component": "Sea State",
            "status": status,
            "summary": raw.get("error") or "Sea-state analysis could not be completed.",
            "model_outputs": {},
            "validation": validation,
            "error": raw.get("error"),
            "follow_up_actions": [
                {"id": "analyse_another", "label": "Analyse another image"},
                {"id": "change_analysis", "label": "Change analysis type"},
            ],
            "result_id": result_id,
            "limitations": [],
            "visualisation": {},
        }

    label = raw.get("predicted_sea_state")
    confidence = raw.get("confidence")
    probs = raw.get("probabilities") or {}
    chart_items = [
        {"label": key, "value": value}
        for key, value in probs.items()
        if isinstance(value, (int, float))
    ]
    visualisation = {}
    if chart_items:
        visualisation["probability_chart"] = {
            "type": "probability_bars",
            "title": "Sea-state model scores",
            "unit": "percent",
            "items": chart_items,
        }

    summary = (
        f"Sea-state classification result: {label}."
        f"{_confidence_clause(confidence)}"
    )
    if raw.get("recommendation"):
        summary += f" Guidance from the sea-state module: {raw['recommendation']}"

    return {
        "analysis_id": "sea",
        "component": "Sea State",
        "status": "success",
        "summary": summary,
        "model_outputs": {
            "predicted_sea_state": label,
            "confidence": confidence,
            "probabilities": probs,
            "recommendation": raw.get("recommendation"),
            "weather_suitability": raw.get("weather_suitability"),
            "risk_indicator": raw.get("risk_indicator"),
            "warnings": raw.get("warnings"),
            "image_quality": raw.get("image_quality"),
            "enhancement_applied": raw.get("enhancement_applied"),
        },
        "model_version": "image_only_mobilenet_v2",
        "visualisation": visualisation,
        "follow_up_actions": DEFAULT_ACTIONS,
        "validation": raw.get("validation"),
        "limitations": list(raw.get("warnings") or []),
        "result_id": result_id,
        "error": None,
    }


def build_vessel_response(raw: dict[str, Any], result_id: str) -> dict[str, Any]:
    if raw.get("error"):
        return {
            "analysis_id": "vessel",
            "component": "Drone Vessel Detection",
            "status": "inference_error",
            "summary": "Vessel detection could not be completed.",
            "model_outputs": {},
            "error": "Vessel detection failed. Please try another image.",
            "follow_up_actions": [
                {"id": "analyse_another", "label": "Analyse another image"},
                {"id": "change_analysis", "label": "Change analysis type"},
            ],
            "result_id": result_id,
            "limitations": [],
            "visualisation": {},
        }

    results = raw.get("results") or []
    count = raw.get("count", 0)
    status_text = raw.get("status")
    if count:
        labels = ", ".join(
            f"{item.get('label')} ({item.get('confidence')}%)"
            for item in results
            if item.get("detection_type") != "flag" and item.get("label")
        )
        summary = (
            f"Vessel detection status: {status_text}. "
            f"Detected vessel count: {count}. "
            f"Labels: {labels or 'n/a'}."
            f"{_confidence_clause(raw.get('confidence'))}"
        )
    else:
        summary = (
            f"Vessel detection status: {status_text or 'No Vessel Detected'}. "
            "No vessel boxes met the detector threshold."
        )

    visualisation: dict[str, Any] = {
        "kind_note": (
            "Bounding-box annotation shows detector localisations. "
            "This is not Grad-CAM and not a class-probability chart."
        )
    }
    if raw.get("annotated_image"):
        visualisation["annotated_image"] = {
            "mime_type": "image/jpeg",
            "base64": raw["annotated_image"],
            "description": "Annotated detection boxes from the YOLO detector.",
        }
    # Detection scores are per-box, not a class distribution — do not chart them as probabilities.

    return {
        "analysis_id": "vessel",
        "component": "Drone Vessel Detection",
        "status": "success",
        "summary": summary,
        "model_outputs": {
            "status": status_text,
            "confidence": raw.get("confidence"),
            "count": count,
            "vessel_origin": raw.get("vessel_origin"),
            "estimated_size": raw.get("estimated_size"),
            "results": results,
            "image_size": raw.get("image_size"),
        },
        "model_version": "yolo_vessel_sl_flag",
        "visualisation": visualisation,
        "follow_up_actions": DEFAULT_ACTIONS,
        "validation": None,
        "limitations": [
            "Detector confidence values are per detection, not a shared class-probability distribution.",
        ],
        "result_id": result_id,
        "error": None,
    }


def build_hull_response(raw: dict[str, Any], result_id: str) -> dict[str, Any]:
    if raw.get("error") and raw.get("prediction") is None and not raw.get("valid_hull_image"):
        # Some hull paths return structured rejection without "error"
        pass

    if raw.get("error") and "prediction" not in raw:
        return {
            "analysis_id": "hull",
            "component": "Hull Condition",
            "status": "inference_error",
            "summary": "Hull analysis could not be completed.",
            "model_outputs": {},
            "error": "Hull analysis failed. Please try another image.",
            "follow_up_actions": [
                {"id": "analyse_another", "label": "Analyse another image"},
                {"id": "change_analysis", "label": "Change analysis type"},
            ],
            "result_id": result_id,
            "limitations": [],
            "visualisation": {},
        }

    prediction = raw.get("prediction")
    confidence = raw.get("confidence")
    valid = raw.get("valid_hull_image")
    warning = raw.get("warning")

    if valid is False or prediction == "non_hull":
        status = "validation_error"
        summary = (
            f"Hull module outcome: {prediction or 'rejected'}."
            f"{_confidence_clause(confidence)} "
            f"{raw.get('recommendation') or warning or ''}"
        ).strip()
    elif prediction == "uncertain":
        status = "success"
        summary = (
            f"Hull module outcome: uncertain."
            f"{_confidence_clause(confidence)} "
            f"{raw.get('recommendation') or warning or ''}"
        ).strip()
    else:
        status = "success"
        summary = (
            f"Hull condition result: {prediction}."
            f"{_confidence_clause(confidence)}"
        )
        if raw.get("recommendation"):
            summary += f" Module guidance: {raw['recommendation']}"

    visualisation: dict[str, Any] = {}
    if raw.get("gradcam"):
        visualisation["gradcam"] = {
            "mime_type": "image/jpeg",
            "base64": raw["gradcam"],
            "description": (
                "Grad-CAM heatmap highlighting regions that influenced the hull model. "
                "This is not a bounding-box detection overlay."
            ),
        }

    defects = raw.get("detected_defects") or []
    chart_items = [
        {"label": item.get("defect"), "value": item.get("confidence")}
        for item in defects
        if item.get("defect") is not None and isinstance(item.get("confidence"), (int, float))
    ]
    if chart_items:
        visualisation["probability_chart"] = {
            "type": "probability_bars",
            "title": "Hull class model scores",
            "unit": "percent",
            "items": chart_items,
        }

    return {
        "analysis_id": "hull",
        "component": "Hull Condition",
        "status": status,
        "summary": summary,
        "model_outputs": {
            "prediction": prediction,
            "confidence": confidence,
            "valid_hull_image": valid,
            "detected_defects": defects,
            "multiple_defects": raw.get("multiple_defects"),
            "recommendation": raw.get("recommendation"),
            "warning": warning,
        },
        "model_version": "hull_mobilenetv2_keras",
        "visualisation": visualisation,
        "follow_up_actions": DEFAULT_ACTIONS,
        "validation": raw.get("validation"),
        "limitations": [warning] if warning else [],
        "result_id": result_id,
        "error": raw.get("error"),
    }


def build_follow_up(action: str, result_payload: dict[str, Any]) -> dict[str, Any]:
    analysis_id = result_payload.get("analysis_id")
    outputs = result_payload.get("model_outputs") or {}
    confidence = outputs.get("confidence")
    actions = DEFAULT_ACTIONS

    if action == "explain_result":
        summary = result_payload.get("summary") or "No stored result is available to explain."
        details = {
            "analysis_id": analysis_id,
            "component": result_payload.get("component"),
            "status": result_payload.get("status"),
            "model_version": result_payload.get("model_version"),
            "key_outputs": outputs,
        }
        return {"status": "success", "summary": summary, "details": details, "follow_up_actions": actions}

    if action == "show_probabilities":
        chart = (result_payload.get("visualisation") or {}).get("probability_chart")
        if chart and chart.get("items"):
            lines = [
                f"{item['label']}: {item['value']}%"
                for item in chart["items"]
                if item.get("label") is not None and item.get("value") is not None
            ]
            summary = "Available model scores: " + "; ".join(lines) + "."
            return {
                "status": "success",
                "summary": summary,
                "details": {"probability_chart": chart},
                "follow_up_actions": actions,
            }
        if analysis_id == "vessel":
            dets = outputs.get("results") or []
            if dets:
                lines = [
                    f"{d.get('label')}: {d.get('confidence')}% (per-detection score)"
                    for d in dets
                    if d.get("label") is not None
                ]
                return {
                    "status": "success",
                    "summary": (
                        "Vessel detector scores are per detection, not a class-probability "
                        "distribution. " + "; ".join(lines) + "."
                    ),
                    "details": {"detections": dets},
                    "follow_up_actions": actions,
                }
        return {
            "status": "success",
            "summary": "No probability distribution is available for this stored result.",
            "details": {},
            "follow_up_actions": actions,
        }

    if action == "explain_confidence":
        if confidence is None:
            summary = "This stored result does not include a confidence score."
        else:
            summary = (
                f"The reported value {confidence}% is the selected model's confidence score "
                "for its decision. It is not a guarantee that the prediction is correct. "
                "Low scores or uncertain/unknown outcomes should be reviewed manually."
            )
        return {
            "status": "success",
            "summary": summary,
            "details": {"confidence": confidence},
            "follow_up_actions": actions,
        }

    if action == "explain_unknown":
        if analysis_id == "radar":
            summary = (
                "For radar, 'unknown' is one of the three trained classes (bird, ship, unknown). "
                f"Current prediction: {outputs.get('final_prediction')}. "
                "This does not mean the wrong analysis type was chosen."
            )
        elif analysis_id == "hull":
            summary = (
                f"Hull outcome '{outputs.get('prediction')}' may mean the image was treated as "
                "non-hull, uncertain, or below the module's confidence threshold. "
                f"Warning from the module: {outputs.get('warning') or 'none'}."
            )
        elif analysis_id == "sea":
            warnings = outputs.get("warnings") or result_payload.get("limitations") or []
            summary = (
                "Sea-state uncertainty notes from the module: "
                + ("; ".join(map(str, warnings)) if warnings else "none recorded")
                + "."
            )
        else:
            summary = (
                "Unknown/uncertain explanations depend on the selected component. "
                "No unknown-class semantics apply to this stored vessel-detection result "
                "beyond absent detections."
            )
        return {
            "status": "success",
            "summary": summary,
            "details": {"analysis_id": analysis_id, "outputs": outputs},
            "follow_up_actions": actions,
        }

    if action == "analyse_another":
        return {
            "status": "success",
            "summary": "Upload another image when ready. Your previous results remain in this chat session until you clear the conversation.",
            "details": {"hint": "select_image"},
            "follow_up_actions": [
                {"id": "change_analysis", "label": "Change analysis type"},
                {"id": "help", "label": "What can you do?"},
            ],
        }

    if action == "change_analysis":
        choices = [
            {"id": key, "label": meta["label"]}
            for key, meta in SUPPORTED_COMPONENTS.items()
        ]
        return {
            "status": "success",
            "summary": "Choose an analysis type for the next image. Automatic routing is unavailable.",
            "details": {"analysis_choices": choices},
            "follow_up_actions": [
                {"id": item["id"], "label": item["label"]} for item in choices
            ],
        }

    if action == "help":
        return {
            "status": "success",
            "summary": (
                "Supported workflow: upload an image, select Radar Objects, Drone Vessel Detection, "
                "Sea State, or Hull Condition, then review the model result and use follow-up buttons. "
                "AIS/GRU/DCPA-TCPA simulation is a separate module and needs movement inputs."
            ),
            "details": {"actions": DEFAULT_ACTIONS},
            "follow_up_actions": DEFAULT_ACTIONS,
        }

    return {
        "status": "unsupported",
        "summary": (
            "That follow-up is not supported. Use the provided buttons or predefined commands "
            "such as explain result, show probabilities, or analyse another image."
        ),
        "details": {},
        "follow_up_actions": DEFAULT_ACTIONS,
    }
