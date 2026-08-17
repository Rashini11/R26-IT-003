from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AISObservation(BaseModel):
    mmsi: int = Field(gt=0)
    timestamp: datetime
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    speed_knots: float = Field(ge=0.0, le=80.0)
    course_degrees: float | None = Field(default=None, ge=0.0, lt=360.0)
    observed: bool = True
    course_missing: bool | None = None


class MotionPredictionRequest(BaseModel):
    observations: list[AISObservation]

    @model_validator(mode="after")
    def validate_history(self):
        if len(self.observations) != 10:
            raise ValueError("Exactly 10 one-minute observations are required.")
        mmsi_values = {item.mmsi for item in self.observations}
        if len(mmsi_values) != 1:
            raise ValueError("All observations must belong to the same MMSI.")
        timestamps = [item.timestamp for item in self.observations]
        if timestamps != sorted(timestamps):
            raise ValueError("Observations must be in chronological order.")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("Duplicate timestamps are not allowed.")
        gaps = [
            (timestamps[index] - timestamps[index - 1]).total_seconds()
            for index in range(1, len(timestamps))
        ]
        if any(gap < 30 or gap > 90 for gap in gaps):
            raise ValueError("Observations must be approximately one minute apart.")
        return self


class VesselState(BaseModel):
    mmsi: int = Field(gt=0)
    timestamp: datetime
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    speed_knots: float = Field(ge=0.0, le=80.0)
    course_degrees: float = Field(ge=0.0, lt=360.0)


class CollisionRiskRequest(BaseModel):
    own_vessel: VesselState
    target_vessel: VesselState


class SimulationStartRequest(BaseModel):
    sar_source: Literal["ship", "bird", "unknown", "all"] = "ship"
    mode: Literal["constructed", "actual"] = "constructed"
    real_interval_seconds: float = Field(default=10.0, ge=0.2, le=60.0)
    simulated_interval_seconds: int = Field(default=10, ge=10, le=60)
    loop: bool = True
    seed: int = 42
    scenario_minutes: int = Field(default=20, ge=15, le=60)

    @model_validator(mode="after")
    def validate_simulated_interval(self):
        if 60 % self.simulated_interval_seconds != 0:
            raise ValueError("The simulated interval must divide evenly into one minute.")
        return self
