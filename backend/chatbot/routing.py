"""
Image-to-component routing abstraction.

This version does not perform automatic image-type classification.
A future trained router can plug into ImageComponentRouter without
changing the chatbot dispatch contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RoutingStatus(str, Enum):
    SELECTED = "selected"
    NEEDS_SELECTION = "needs_selection"
    SUGGESTED = "suggested"
    UNCERTAIN = "uncertain"
    UNSUPPORTED = "unsupported"
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNAVAILABLE = "unavailable"


SUPPORTED_COMPONENTS = {
    "radar": {
        "id": "radar",
        "label": "Radar Objects",
        "description": "Classify a radar screenshot as bird, ship, or unknown.",
    },
    "vessel": {
        "id": "vessel",
        "label": "Drone Vessel Detection",
        "description": "Detect boats/ships and Sri Lankan flag cues in a drone image.",
    },
    "sea": {
        "id": "sea",
        "label": "Sea State",
        "description": "Classify sea-surface condition from a marine surface image.",
    },
    "hull": {
        "id": "hull",
        "label": "Hull Condition",
        "description": "Assess underwater hull condition or reject non-hull images.",
    },
}


@dataclass
class RoutingDecision:
    """
    Separate from component prediction results.

    A radar class of "unknown" is a model output, not a routing rejection.
    """

    status: RoutingStatus
    selected_component: Optional[str] = None
    suggested_component: Optional[str] = None
    automatic_routing_available: bool = False
    requires_user_confirmation: bool = False
    message: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "selected_component": self.selected_component,
            "suggested_component": self.suggested_component,
            "automatic_routing_available": self.automatic_routing_available,
            "requires_user_confirmation": self.requires_user_confirmation,
            "message": self.message,
            "details": self.details,
        }


class ImageComponentRouter:
    """
    Manual-selection router with a stable hook for a future classifier.

    Future integration sketch:
      1. Load router checkpoint + preprocessing metadata (means/std, class map).
      2. Override ``classify_image`` to return RoutingDecision with
         status SUGGESTED / UNCERTAIN / UNSUPPORTED / NEEDS_CONFIRMATION.
      3. Chat UI confirms before dispatch when confirmation is required.
    """

    def __init__(self, automatic_available: bool = False):
        self.automatic_available = automatic_available

    def classify_image(self, _image_bytes: bytes | None = None) -> RoutingDecision:
        """
        Placeholder for a future image-type classifier.

        Must not invent confidence, use filenames as type detection, or
        compare unrelated component-model scores.
        """
        return RoutingDecision(
            status=RoutingStatus.UNAVAILABLE,
            automatic_routing_available=False,
            message=(
                "Automatic image-type routing is not available in this version. "
                "Please choose which analysis to run."
            ),
            details={"reason": "router_checkpoint_not_configured"},
        )

    def resolve(
        self,
        selected_component: Optional[str],
        image_bytes: bytes | None = None,
        confirm_suggestion: bool = False,
    ) -> RoutingDecision:
        if selected_component:
            key = selected_component.strip().lower()
            aliases = {
                "boat": "vessel",
                "drone": "vessel",
                "vessel_detection": "vessel",
                "hull_defect": "hull",
                "hull_condition": "hull",
                "sea_state": "sea",
                "radar_object": "radar",
                "radar_objects": "radar",
            }
            key = aliases.get(key, key)
            if key not in SUPPORTED_COMPONENTS:
                return RoutingDecision(
                    status=RoutingStatus.UNSUPPORTED,
                    automatic_routing_available=self.automatic_available,
                    message=(
                        f'"{selected_component}" is not a supported analysis. '
                        "Choose Radar Objects, Drone Vessel Detection, Sea State, or Hull Condition."
                    ),
                    details={"requested": selected_component},
                )
            return RoutingDecision(
                status=RoutingStatus.SELECTED,
                selected_component=key,
                automatic_routing_available=self.automatic_available,
                message=f"Using your selection: {SUPPORTED_COMPONENTS[key]['label']}.",
            )

        auto = self.classify_image(image_bytes)
        if auto.status == RoutingStatus.SUGGESTED and confirm_suggestion:
            return RoutingDecision(
                status=RoutingStatus.SELECTED,
                selected_component=auto.suggested_component,
                suggested_component=auto.suggested_component,
                automatic_routing_available=auto.automatic_routing_available,
                message=f"Confirmed suggested analysis: {auto.suggested_component}.",
            )
        if auto.status in {
            RoutingStatus.SUGGESTED,
            RoutingStatus.UNCERTAIN,
            RoutingStatus.NEEDS_CONFIRMATION,
            RoutingStatus.UNSUPPORTED,
        }:
            return auto

        return RoutingDecision(
            status=RoutingStatus.NEEDS_SELECTION,
            automatic_routing_available=False,
            message=(
                "Automatic image-type routing is unavailable. "
                "Please select which analysis to run for this image."
            ),
            details={"reason": "no_component_selected"},
        )


DEFAULT_ROUTER = ImageComponentRouter(automatic_available=False)
