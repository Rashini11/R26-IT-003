from __future__ import annotations

import re
from typing import Optional


INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("explain_result", re.compile(r"\b(explain|what does|interpret).*(result|this|prediction)\b", re.I)),
    ("show_probabilities", re.compile(r"\b(probabilities|probability|scores|distribution|chart)\b", re.I)),
    ("explain_confidence", re.compile(r"\b(confidence|how sure|certainty|score)\b", re.I)),
    ("explain_unknown", re.compile(r"\b(unknown|uncertain|non[- ]?hull|invalid)\b", re.I)),
    ("analyse_another", re.compile(r"\b(another|new|different)\s+(image|photo|picture)\b", re.I)),
    ("change_analysis", re.compile(r"\b(change|switch|different)\s+(analysis|component|module|type)\b", re.I)),
    ("clear_conversation", re.compile(r"\b(clear|reset)\s+(chat|conversation|history)\b", re.I)),
    ("help", re.compile(r"^\s*(help|commands|what can you do)\s*$", re.I)),
]


DEFAULT_ACTIONS = [
    {"id": "explain_result", "label": "Explain this result"},
    {"id": "show_probabilities", "label": "Show available probabilities"},
    {"id": "explain_confidence", "label": "Explain confidence"},
    {"id": "explain_unknown", "label": "Explain unknown / uncertain"},
    {"id": "analyse_another", "label": "Analyse another image"},
    {"id": "change_analysis", "label": "Change analysis type"},
]


def match_intent(text: str) -> Optional[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    for intent, pattern in INTENT_PATTERNS:
        if pattern.search(cleaned):
            return intent
    return None


def unsupported_text_message() -> str:
    return (
        "I use predefined commands and analysis buttons, not free-form natural language. "
        "Upload an image, choose an analysis (Radar Objects, Drone Vessel Detection, "
        "Sea State, or Hull Condition), then use the follow-up buttons for explanations."
    )
