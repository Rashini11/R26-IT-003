from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .ais_controller import AISScenario, AISSimulationController
from .config import (
    AIS_CSV_PATH,
    MOTION_METADATA_PATH,
    MOTION_MODEL_PATH,
    SAR_ROOT,
)
from .encounter_controller import aggregate_minute_observations
from .motion_inference import MotionPredictor
from .risk_engine import calculate_collision_risk
from .environment_service import get_environment_snapshot
from .sar_streamer import SARImageStreamer
from .schemas import (
    CollisionRiskRequest,
    MotionPredictionRequest,
    SimulationStartRequest,
)
from .state import SimulationState


LOGGER = logging.getLogger("maritime-simulation")

# IMPORTANT:
# This file is now a ROUTER only.
# It does not create a second FastAPI application and does not add CORS.
# backend/main.py owns the single application and includes this router.
router = APIRouter(tags=["Maritime Simulation"])

STATE = SimulationState()
AIS_CONTROLLER = AISSimulationController()
MOTION_PREDICTOR = MotionPredictor()

# MongoDB collections are injected by backend.main so the
# simulation reuses OceanIQ's existing MongoClient.
SIMULATION_RUNS_COLLECTION = None
SIMULATION_EVENTS_COLLECTION = None


def configure_simulation_persistence(
    runs_collection,
    events_collection,
) -> None:
    global SIMULATION_RUNS_COLLECTION
    global SIMULATION_EVENTS_COLLECTION

    SIMULATION_RUNS_COLLECTION = runs_collection
    SIMULATION_EVENTS_COLLECTION = events_collection


def _mongo_safe(value):
    """
    Convert common ML/numpy values into Mongo-safe Python types.
    """
    if isinstance(value, dict):
        return {
            str(key): _mongo_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _mongo_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _serialise_state(state: dict[str, Any]) -> dict[str, Any]:
    payload = dict(state)
    timestamp = payload.get("timestamp")
    if hasattr(timestamp, "isoformat"):
        payload["timestamp"] = timestamp.isoformat()
    return payload


def _ensure_motion_predictor_ready() -> None:
    """
    Lazily load the selected GRU model.

    This allows the complete OceanIQ backend to start even if the motion
    model has not yet been used. When a simulation or direct motion
    prediction is requested, the model is loaded and validated.
    """
    if not MOTION_PREDICTOR.ready:
        MOTION_PREDICTOR.load()

    if not MOTION_PREDICTOR.ready:
        raise RuntimeError(
            MOTION_PREDICTOR.load_error
            or "AIS motion predictor could not be loaded."
        )


def _safe_motion(
    history: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    if len(history) < 10:
        return None, None

    try:
        _ensure_motion_predictor_ready()
        return MOTION_PREDICTOR.predict(history[-10:]), None
    except Exception as error:
        return None, str(error)


def _classification_payload(
    result: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:

    if error:
        return {
            "error": error
        }

    result = result or {}

    return {
        "classification":
            result.get(
                "final_prediction"
            ),

        "binary_prediction":
            result.get(
                "binary_prediction"
            ),

        "confidence":
            result.get(
                "confidence"
            ),

        "bird_probability":
            result.get(
                "bird_probability"
            ),

        "ship_probability":
            result.get(
                "ship_probability"
            ),

        "unknown_probability":
            result.get(
                "unknown_probability"
            ),
        "unknown_threshold":
            result.get(
                "unknown_threshold"
            ),

        "decision_status":
            result.get(
                "decision_status"
            ),

        # Compatibility with the existing UI
        # until all simulation display code
        # has migrated.
        "agreement_status":
            result.get(
                "decision_status"
            ),

        "model_name":
            result.get(
                "model_name"
            ),

        "model_version":
            result.get(
                "model_version"
            ),
    }

def _run_simulation(
    request: SimulationStartRequest,
    scenario: AISScenario,
    sar_streamer: SARImageStreamer,
) -> None:
    """
    Background simulation loop.

    SAR:
        One image is replayed for each simulation tick and classified by
        the existing radar classification pipeline.

    AIS:
        Own-vessel and target-vessel states advance at the configured
        simulated interval.

    GRU:
        Six 10-second ticks are aggregated into one one-minute state.
        Once 10 minutes of history are available, the selected GRU
        forecasts 5 minutes ahead.

    Risk:
        Current relative motion is evaluated using DCPA/TCPA.
    """
    ticks_per_minute = 60 // request.simulated_interval_seconds

    own_tick_history: list[dict[str, Any]] = []
    target_tick_history: list[dict[str, Any]] = []

    tick_index = 0

    try:
        while not STATE.stop_event.is_set():

            if tick_index >= scenario.length:
                if not request.loop:
                    break

                tick_index = 0
                own_tick_history.clear()
                target_tick_history.clear()

            own_state = scenario.own_states[tick_index]
            target_state = scenario.target_states[tick_index]

            own_tick_history.append(own_state)
            target_tick_history.append(target_state)

            # ---------------------------------------------------------
            # SAR IMAGE CLASSIFICATION
            # ---------------------------------------------------------
            image_path = sar_streamer.next_image()

            classification = None
            classification_error = None

            try:
                classification = sar_streamer.classify(image_path)
            except Exception as error:
                classification_error = str(error)
                LOGGER.warning("SAR classification failed: %s", error)

            # ---------------------------------------------------------
            # AIS -> ONE-MINUTE HISTORY -> GRU
            # ---------------------------------------------------------
            own_minute_history = aggregate_minute_observations(
                own_tick_history,
                ticks_per_minute,
                max_minutes=10,
            )

            target_minute_history = aggregate_minute_observations(
                target_tick_history,
                ticks_per_minute,
                max_minutes=10,
            )

            own_prediction, own_motion_error = _safe_motion(
                own_minute_history
            )
            target_prediction, target_motion_error = _safe_motion(
                target_minute_history
            )

            # ---------------------------------------------------------
            # CURRENT ENCOUNTER RISK
            # ---------------------------------------------------------
            encounter = calculate_collision_risk(
                own_state,
                target_state,
            )

            # ---------------------------------------------------------
            # ENVIRONMENTAL CONTEXT
            #
            # Use the midpoint of the two vessels as the encounter
            # location. Results are cached by environment_service so
            # an external API call is not made for every frame.
            # ---------------------------------------------------------
            encounter_latitude = (
                float(own_state["latitude"])
                + float(target_state["latitude"])
            ) / 2.0

            encounter_longitude = (
                float(own_state["longitude"])
                + float(target_state["longitude"])
            ) / 2.0

            try:
                environment = get_environment_snapshot(
                    encounter_latitude,
                    encounter_longitude,
                )
            except Exception as environment_error:
                LOGGER.warning(
                    "Environmental lookup failed: %s",
                    environment_error,
                )

                environment = {
                    "available": False,
                    "error": str(environment_error),
                    "latitude": encounter_latitude,
                    "longitude": encounter_longitude,
                    "used_for_model_inference": False,
                    "used_for_collision_risk": False,
                }

            # ---------------------------------------------------------
            # SAVE CURRENT EVENT
            # ---------------------------------------------------------
            with STATE.lock:
                STATE.frame_number += 1
                frame_number = STATE.frame_number

                STATE.current_image_path = image_path

                event = {
                    "simulation_id": scenario.scenario_id,
                    "scenario_mode": scenario.mode,
                    "scenario_description": scenario.description,
                    "running": True,
                    "frame_number": frame_number,
                    "real_timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "simulated_timestamp": own_state[
                        "timestamp"
                    ].isoformat(),
                    "sar": {
                        "filename": image_path.name,
                        "image_url": (
                            "/simulation/current-image"
                            f"?frame={frame_number}"
                        ),
                        **_classification_payload(
                            classification,
                            classification_error,
                        ),
                    },
                    "own_vessel": {
                        "current_state": _serialise_state(
                            own_state
                        ),
                        "history_available_minutes": len(
                            own_minute_history
                        ),
                        "motion_prediction": own_prediction,
                        "motion_error": own_motion_error,
                    },
                    "target_vessel": {
                        "current_state": _serialise_state(
                            target_state
                        ),
                        "history_available_minutes": len(
                            target_minute_history
                        ),
                        "motion_prediction": target_prediction,
                        "motion_error": target_motion_error,
                    },
                    "encounter": encounter,
                    "environment": environment,
                    "simulation_notice": (
                        "SAR appearance and AIS target motion are linked "
                        "by this simulation scenario; random SAR images "
                        "are not used to derive vessel speed."
                    ),
                }

                STATE.latest = event
                STATE.history.append(event)

            # -------------------------------------------------
            # SAVE SIMULATION EVENT TO MONGODB
            # Database failure is intentionally non-fatal.
            # -------------------------------------------------
            if SIMULATION_EVENTS_COLLECTION is not None:
                try:
                    SIMULATION_EVENTS_COLLECTION.insert_one(
                        _mongo_safe(event)
                    )
                except Exception as db_error:
                    LOGGER.warning(
                        "Simulation event MongoDB save failed: %s",
                        db_error,
                    )

            tick_index += 1

            if STATE.stop_event.wait(
                request.real_interval_seconds
            ):
                break

    except Exception as error:
        LOGGER.exception(
            "Simulation stopped after an error"
        )

        with STATE.lock:
            STATE.last_error = str(error)

    finally:
        with STATE.lock:
            STATE.running = False

            if STATE.latest:
                STATE.latest["running"] = False

            final_frame_count = STATE.frame_number
            final_error = STATE.last_error

        if SIMULATION_RUNS_COLLECTION is not None:
            try:
                SIMULATION_RUNS_COLLECTION.update_one(
                    {
                        "simulation_id":
                            scenario.scenario_id
                    },
                    {
                        "$set": {
                            "status":
                                (
                                    "error"
                                    if final_error
                                    else "stopped"
                                ),
                            "ended_at":
                                datetime.now(
                                    timezone.utc
                                ),
                            "frame_count":
                                final_frame_count,
                            "last_error":
                                final_error,
                        }
                    },
                )
            except Exception as db_error:
                LOGGER.warning(
                    "Simulation completion MongoDB save failed: %s",
                    db_error,
                )



# =========================================================
# DATABASE HISTORY
# =========================================================

def _serialise_mongo_document(
    document: dict[str, Any],
) -> dict[str, Any]:

    document = dict(document)

    document.pop(
        "_id",
        None,
    )

    for key, value in list(
        document.items()
    ):
        if hasattr(
            value,
            "isoformat",
        ):
            document[key] = (
                value.isoformat()
            )

    return document


@router.get(
    "/simulation/database/runs"
)
def database_simulation_runs(
    limit: int = Query(
        default=20,
        ge=1,
        le=200,
    ),
):
    if (
        SIMULATION_RUNS_COLLECTION
        is None
    ):
        return {
            "database_available":
                False,
            "records": [],
        }

    try:
        records = list(
            SIMULATION_RUNS_COLLECTION
            .find({})
            .sort(
                "started_at",
                -1,
            )
            .limit(limit)
        )

        records = [
            _serialise_mongo_document(
                record
            )
            for record in records
        ]

        return {
            "database_available":
                True,
            "count":
                len(records),
            "records":
                records,
        }

    except Exception as error:
        LOGGER.warning(
            "Simulation run DB read failed: %s",
            error,
        )

        return {
            "database_available":
                False,
            "records": [],
            "error":
                str(error),
        }


@router.get(
    "/simulation/database/events"
)
def database_simulation_events(
    simulation_id: str | None = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),
):
    if (
        SIMULATION_EVENTS_COLLECTION
        is None
    ):
        return {
            "database_available":
                False,
            "records": [],
        }

    try:
        query = {}

        if simulation_id:
            query[
                "simulation_id"
            ] = simulation_id

        records = list(
            SIMULATION_EVENTS_COLLECTION
            .find(
                query,
                {"_id": 0},
            )
            .sort(
                "real_timestamp",
                -1,
            )
            .limit(limit)
        )

        return {
            "database_available":
                True,
            "count":
                len(records),
            "records":
                records,
        }

    except Exception as error:
        LOGGER.warning(
            "Simulation event DB read failed: %s",
            error,
        )

        return {
            "database_available":
                False,
            "records": [],
            "error":
                str(error),
        }


@router.get(
    "/simulation/database/summary"
)
def database_simulation_summary():

    if (
        SIMULATION_RUNS_COLLECTION
        is None
        or
        SIMULATION_EVENTS_COLLECTION
        is None
    ):
        return {
            "database_available":
                False,
        }

    try:
        latest_run = (
            SIMULATION_RUNS_COLLECTION
            .find_one(
                {},
                sort=[
                    (
                        "started_at",
                        -1,
                    )
                ],
            )
        )

        latest_event = (
            SIMULATION_EVENTS_COLLECTION
            .find_one(
                {},
                {"_id": 0},
                sort=[
                    (
                        "real_timestamp",
                        -1,
                    )
                ],
            )
        )

        return {
            "database_available":
                True,

            "run_count":
                SIMULATION_RUNS_COLLECTION
                .count_documents({}),

            "event_count":
                SIMULATION_EVENTS_COLLECTION
                .count_documents({}),

            "latest_run":
                (
                    _serialise_mongo_document(
                        latest_run
                    )
                    if latest_run
                    else None
                ),

            "latest_event":
                latest_event,
        }

    except Exception as error:
        LOGGER.warning(
            "Simulation summary DB read failed: %s",
            error,
        )

        return {
            "database_available":
                False,
            "error":
                str(error),
        }


# =========================================================
# SIMULATION MODULE HEALTH
# =========================================================
@router.get("/simulation/health")
def simulation_health():
    MOTION_PREDICTOR.load()

    return {
        "status": "ok",
        "service": "Maritime AI Simulation Module",
        "assets": {
            "ais_csv_exists": AIS_CSV_PATH.exists(),
            "sar_test_root_exists": SAR_ROOT.exists(),
            "motion_checkpoint_exists": MOTION_MODEL_PATH.exists(),
            "motion_metadata_exists": MOTION_METADATA_PATH.exists(),
        },
        "motion_model_ready": MOTION_PREDICTOR.ready,
        "motion_model_error": MOTION_PREDICTOR.load_error,
    }


# =========================================================
# START SIMULATION
# =========================================================
@router.post("/simulation/start")
def start_simulation(
    request: SimulationStartRequest,
):
    with STATE.lock:
        if STATE.running or STATE.starting:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A simulation is already running "
                    "or starting."
                ),
            )

        STATE.starting = True

    try:
        # Validate/load GRU before starting the background loop.
        _ensure_motion_predictor_ready()

        scenario = AIS_CONTROLLER.create_scenario(
            mode=request.mode,
            interval_seconds=request.simulated_interval_seconds,
            duration_minutes=request.scenario_minutes,
            seed=request.seed,
        )

        sar_streamer = SARImageStreamer(
            request.sar_source,
            seed=request.seed,
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:
        with STATE.lock:
            STATE.starting = False

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        LOGGER.exception(
            "Unexpected simulation start failure"
        )

        with STATE.lock:
            STATE.starting = False

        raise HTTPException(
            status_code=500,
            detail=(
                "Simulation setup failed unexpectedly. "
                "Check backend logs."
            ),
        ) from error

    STATE.reset()

    with STATE.lock:
        STATE.starting = False
        STATE.running = True
        STATE.simulation_id = scenario.scenario_id
        STATE.mode = scenario.mode

        STATE.thread = threading.Thread(
            target=_run_simulation,
            args=(
                request,
                scenario,
                sar_streamer,
            ),
            daemon=True,
            name="maritime-simulation",
        )

        STATE.thread.start()

    # -----------------------------------------------------
    # SAVE SIMULATION RUN
    # -----------------------------------------------------
    if SIMULATION_RUNS_COLLECTION is not None:
        try:
            SIMULATION_RUNS_COLLECTION.update_one(
                {
                    "simulation_id":
                        scenario.scenario_id
                },
                {
                    "$set": {
                        "simulation_id":
                            scenario.scenario_id,
                        "scenario_mode":
                            scenario.mode,
                        "scenario_description":
                            scenario.description,
                        "status":
                            "running",
                        "started_at":
                            datetime.now(
                                timezone.utc
                            ),
                        "real_interval_seconds":
                            request.real_interval_seconds,
                        "simulated_interval_seconds":
                            request.simulated_interval_seconds,
                        "scenario_minutes":
                            request.scenario_minutes,
                        "loop":
                            request.loop,
                        "seed":
                            request.seed,
                        "sar_source":
                            request.sar_source,
                        "sar_root":
                            str(SAR_ROOT),
                        "motion_model_path":
                            str(MOTION_MODEL_PATH),
                    }
                },
                upsert=True,
            )

        except Exception as db_error:
            LOGGER.warning(
                "Simulation run MongoDB save failed: %s",
                db_error,
            )

    return {
        "message": "Simulation started",
        "simulation_id": scenario.scenario_id,
        "mode": scenario.mode,
        "description": scenario.description,
        "real_interval_seconds": (
            request.real_interval_seconds
        ),
        "simulated_interval_seconds": (
            request.simulated_interval_seconds
        ),
        "frames_per_simulated_minute": (
            60 // request.simulated_interval_seconds
        ),
        "scenario_ticks": scenario.length,
    }


# =========================================================
# STOP SIMULATION
# =========================================================
@router.post("/simulation/stop")
def stop_simulation():
    with STATE.lock:
        if not STATE.running:
            return {
                "message": (
                    "Simulation is already stopped."
                )
            }

        STATE.stop_event.set()

    return {
        "message": "Simulation stopping."
    }


# =========================================================
# SIMULATION STATUS
# =========================================================
@router.get("/simulation/status")
def simulation_status():
    with STATE.lock:
        return {
            "running": STATE.running,
            "starting": STATE.starting,
            "simulation_id": STATE.simulation_id,
            "mode": STATE.mode,
            "frame_number": STATE.frame_number,
            "processed_events": len(
                STATE.history
            ),
            "last_error": STATE.last_error,
        }


# =========================================================
# LATEST SIMULATION EVENT
# =========================================================
@router.get("/simulation/latest")
def simulation_latest():
    with STATE.lock:
        if STATE.latest is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No simulation event is "
                    "available yet."
                ),
            )

        return STATE.latest


# =========================================================
# SIMULATION HISTORY
# =========================================================
@router.get("/simulation/history")
def simulation_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=500,
    ),
):
    with STATE.lock:
        results = list(
            STATE.history
        )[-limit:]

        return {
            "count": len(results),
            "results": results,
        }


# =========================================================
# CURRENT SAR IMAGE
# =========================================================
@router.get("/simulation/current-image")
def current_image():
    with STATE.lock:
        image_path = STATE.current_image_path

    if (
        image_path is None
        or not image_path.exists()
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "No current SAR image "
                "is available."
            ),
        )

    return FileResponse(image_path)


# =========================================================
# DIRECT GRU MOTION PREDICTION
# =========================================================
@router.post("/predict-vessel-motion")
def predict_vessel_motion(
    request: MotionPredictionRequest,
):
    history = [
        item.model_dump()
        for item in request.observations
    ]

    try:
        _ensure_motion_predictor_ready()
        return MOTION_PREDICTOR.predict(history)

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error


# =========================================================
# DIRECT COLLISION-RISK PREDICTION
# =========================================================
@router.post("/predict-collision-risk")
def predict_collision_risk(
    request: CollisionRiskRequest,
):
    return calculate_collision_risk(
        request.own_vessel.model_dump(),
        request.target_vessel.model_dump(),
    )
