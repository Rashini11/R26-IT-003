"""OceanIQ deterministic Chat Assistant (no LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter

__all__ = ["router"]


def __getattr__(name: str):
    if name == "router":
        from .router import router as chat_router

        return chat_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
