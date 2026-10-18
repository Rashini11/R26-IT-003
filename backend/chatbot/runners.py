"""
Registry for shared component inference callables.

backend.main registers the active runners after models are loaded so the
chatbot and HTTP endpoints share the same model instances and pipelines.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


Runner = Callable[..., dict[str, Any]]

_RUNNERS: dict[str, Runner] = {}


def register_component_runners(
    *,
    radar: Optional[Runner] = None,
    vessel: Optional[Runner] = None,
    sea: Optional[Runner] = None,
    hull: Optional[Runner] = None,
) -> None:
    if radar is not None:
        _RUNNERS["radar"] = radar
    if vessel is not None:
        _RUNNERS["vessel"] = vessel
    if sea is not None:
        _RUNNERS["sea"] = sea
    if hull is not None:
        _RUNNERS["hull"] = hull


def get_runner(component: str) -> Optional[Runner]:
    return _RUNNERS.get(component)


def available_runners() -> dict[str, bool]:
    return {
        "radar": "radar" in _RUNNERS,
        "vessel": "vessel" in _RUNNERS,
        "sea": "sea" in _RUNNERS,
        "hull": "hull" in _RUNNERS,
    }
