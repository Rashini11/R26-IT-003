from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


AnalysisId = Literal["radar", "vessel", "sea", "hull"]
ProcessStatus = Literal[
    "needs_selection",
    "processing",
    "success",
    "validation_error",
    "inference_error",
    "unsupported",
    "model_unavailable",
]


class ChatAnalyzeResponse(BaseModel):
    analysis_id: Optional[str] = None
    component: Optional[str] = None
    status: ProcessStatus
    summary: str
    model_outputs: dict[str, Any] = Field(default_factory=dict)
    model_version: Optional[str] = None
    visualisation: dict[str, Any] = Field(default_factory=dict)
    follow_up_actions: list[dict[str, str]] = Field(default_factory=list)
    validation: Optional[dict[str, Any]] = None
    limitations: list[str] = Field(default_factory=list)
    routing: dict[str, Any] = Field(default_factory=dict)
    result_id: Optional[str] = None
    error: Optional[str] = None


class ChatFollowUpRequest(BaseModel):
    action: str = Field(min_length=1, max_length=80)
    result_payload: dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = Field(default=None, max_length=500)


class ChatFollowUpResponse(BaseModel):
    status: Literal["success", "unsupported"]
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    follow_up_actions: list[dict[str, str]] = Field(default_factory=list)


class ChatIntentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class ChatIntentResponse(BaseModel):
    matched: bool
    intent: Optional[str] = None
    message: str
    follow_up_actions: list[dict[str, str]] = Field(default_factory=list)
