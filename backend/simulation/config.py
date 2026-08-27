from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAR_ROOT = (PROJECT_ROOT / "ml" / "live_simulation_radar").resolve()
AIS_CSV_PATH = PROJECT_ROOT / "ml" / "external_datasets" / "ais_motion" / "processed_AIS_dataset.csv"
MOTION_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "final" / "ais_motion_gru_under90_deploy.pth"
MOTION_METADATA_PATH = PROJECT_ROOT / "ml" / "ais_motion" / "sequences_5min" / "metadata.json"
MOTION_TEST_NPZ_PATH = PROJECT_ROOT / "ml" / "ais_motion" / "sequences_5min" / "test_motion_sequences.npz"

ALLOWED_FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EARTH_RADIUS_METRES = 6_371_000.0
METRES_PER_NAUTICAL_MILE = 1852.0
KNOT_TO_METRES_PER_SECOND = METRES_PER_NAUTICAL_MILE / 3600.0


@dataclass(frozen=True)
class RiskThresholds:
    critical_dcpa_nm: float = 0.5
    critical_tcpa_minutes: float = 10.0
    high_dcpa_nm: float = 1.0
    high_tcpa_minutes: float = 20.0
    medium_dcpa_nm: float = 2.0
    medium_tcpa_minutes: float = 30.0
    assessment_horizon_minutes: float = 30.0


RISK_THRESHOLDS = RiskThresholds()
