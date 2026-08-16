from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.simulation.ais_controller import AISSimulationController
from backend.simulation.config import (
    METRES_PER_NAUTICAL_MILE,
    MOTION_TEST_NPZ_PATH,
    RISK_THRESHOLDS,
    SAR_ROOT,
)
from backend.simulation.encounter_controller import aggregate_minute_observations
from backend.simulation.geo import haversine_metres
from backend.simulation.motion_inference import MotionPredictor
from backend.simulation.risk_engine import calculate_collision_risk
from backend.simulation.sar_streamer import SARImageStreamer

OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation" / "complete_simulation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


def classify_speed(speed_knots: float) -> str:
    if speed_knots <= 2.0:
        return "Stopped"
    if speed_knots <= 6.0:
        return "Slow"
    if speed_knots <= 12.0:
        return "Moderate"
    return "Fast"


def ground_truth_risk(dcpa_metres: float, tcpa_minutes: float) -> str:
    dcpa_nm = dcpa_metres / METRES_PER_NAUTICAL_MILE
    if tcpa_minutes <= RISK_THRESHOLDS.critical_tcpa_minutes and dcpa_nm < RISK_THRESHOLDS.critical_dcpa_nm:
        return "Critical"
    if tcpa_minutes <= RISK_THRESHOLDS.high_tcpa_minutes and dcpa_nm < RISK_THRESHOLDS.high_dcpa_nm:
        return "High"
    if tcpa_minutes <= RISK_THRESHOLDS.medium_tcpa_minutes and dcpa_nm < RISK_THRESHOLDS.medium_dcpa_nm:
        return "Medium"
    return "Low"


def future_closest_approach(own_states, target_states, start_index, max_minutes=30):
    end_index = min(len(own_states), start_index + max_minutes + 1)
    distances = [
        haversine_metres(
            own_states[index]["latitude"], own_states[index]["longitude"],
            target_states[index]["latitude"], target_states[index]["longitude"],
        )
        for index in range(start_index, end_index)
    ]
    minimum_offset = int(np.argmin(distances))
    return float(distances[minimum_offset]), float(minimum_offset)


def mean_or_none(values):
    return None if not values else float(np.mean(values))


def median_or_none(values):
    return None if not values else float(np.median(values))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--mode", choices=["constructed", "actual", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-minutes", type=int, default=20)
    parser.add_argument("--classification-samples", type=int, default=5)
    parser.add_argument("--skip-classification", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MOTION_TEST_NPZ_PATH.exists():
        raise FileNotFoundError(
            "Test MMSI split is required for leakage-safe evaluation: "
            f"{MOTION_TEST_NPZ_PATH}"
        )
    with np.load(MOTION_TEST_NPZ_PATH) as split_data:
        test_mmsi = {int(value) for value in np.unique(split_data["mmsi"])}

    controller = AISSimulationController()
    predictor = MotionPredictor()
    predictor.load()
    if not predictor.ready:
        raise RuntimeError(f"Motion model unavailable: {predictor.load_error}")

    modes = ["constructed", "actual"] if args.mode == "both" else [args.mode]
    frame_rows = []
    scenario_rows = []
    actual_risk_labels = []
    predicted_risk_labels = []
    position_errors = []
    speed_errors = []
    motion_correct = []
    dcpa_errors = []
    tcpa_errors = []
    classification_correct = []
    classification_latencies = []
    uncertain_count = 0
    failed_frames = 0

    scenario_number = 0
    for mode in modes:
        for local_index in range(args.scenarios):
            seed = args.seed + scenario_number
            scenario_number += 1
            try:
                scenario = controller.create_scenario(
                    mode=mode,
                    interval_seconds=10,
                    duration_minutes=args.scenario_minutes,
                    seed=seed,
                    allowed_mmsi=test_mmsi,
                )
            except Exception as error:
                scenario_rows.append({
                    "scenario_id": f"failed-{mode}-{seed}",
                    "mode": mode,
                    "status": "failed",
                    "error": str(error),
                })
                continue

            own_minutes = aggregate_minute_observations(
                scenario.own_states,
                ticks_per_minute=6,
            )
            target_minutes = aggregate_minute_observations(
                scenario.target_states,
                ticks_per_minute=6,
            )
            scenario_position_errors = []
            scenario_speed_errors = []
            scenario_risk_correct = []
            scenario_start = time.perf_counter()
            classification_streamer = None
            classification_remaining = 0
            if not args.skip_classification:
                try:
                    classification_streamer = SARImageStreamer("all", seed=seed)
                    classification_remaining = args.classification_samples
                except Exception:
                    failed_frames += args.classification_samples

            for minute_index in range(9, min(len(own_minutes), len(target_minutes)) - 5):
                own_history = own_minutes[minute_index - 9:minute_index + 1]
                target_history = target_minutes[minute_index - 9:minute_index + 1]
                future_index = minute_index + 5
                own_actual = own_minutes[future_index]
                target_actual = target_minutes[future_index]
                tick_start = time.perf_counter()
                classification_attempted = False
                classification_latency_ms = None
                classification_prediction = None
                classification_expected = None
                classification_error = None
                if classification_streamer is not None and classification_remaining > 0:
                    classification_attempted = True
                    classification_remaining -= 1
                    image_path = classification_streamer.next_image()
                    classification_expected = image_path.relative_to(SAR_ROOT).parts[0].lower()
                    classification_start = time.perf_counter()
                    try:
                        classification_result = classification_streamer.classify(image_path)
                        classification_latency_ms = (
                            time.perf_counter() - classification_start
                        ) * 1000.0
                        classification_latencies.append(classification_latency_ms)
                        classification_prediction = str(
                            classification_result.get("final_prediction", "")
                        ).lower()
                        if classification_prediction == "uncertain":
                            uncertain_count += 1
                        classification_correct.append(
                            classification_prediction == classification_expected
                        )
                    except Exception as error:
                        failed_frames += 1
                        classification_error = str(error)

                own_prediction = predictor.predict(own_history)
                target_prediction = predictor.predict(target_history)
                risk = calculate_collision_risk(own_minutes[minute_index], target_minutes[minute_index])
                processing_ms = (time.perf_counter() - tick_start) * 1000.0

                own_position_error = haversine_metres(
                    own_prediction["predicted_latitude"], own_prediction["predicted_longitude"],
                    own_actual["latitude"], own_actual["longitude"],
                )
                target_position_error = haversine_metres(
                    target_prediction["predicted_latitude"], target_prediction["predicted_longitude"],
                    target_actual["latitude"], target_actual["longitude"],
                )
                own_speed_error = abs(own_prediction["predicted_speed_knots"] - own_actual["speed_knots"])
                target_speed_error = abs(target_prediction["predicted_speed_knots"] - target_actual["speed_knots"])
                own_motion_ok = own_prediction["predicted_motion_class"] == classify_speed(own_actual["speed_knots"])
                target_motion_ok = target_prediction["predicted_motion_class"] == classify_speed(target_actual["speed_knots"])

                true_dcpa, true_tcpa_minutes = future_closest_approach(
                    own_minutes, target_minutes, minute_index, max_minutes=30
                )
                true_risk = ground_truth_risk(true_dcpa, true_tcpa_minutes)
                predicted_risk = risk["risk_level"]
                predicted_dcpa = risk["dcpa_metres"]
                predicted_tcpa = risk["tcpa_minutes"]

                pair_position_error = (own_position_error + target_position_error) / 2.0
                pair_speed_error = (own_speed_error + target_speed_error) / 2.0
                pair_motion_ok = (int(own_motion_ok) + int(target_motion_ok)) / 2.0
                position_errors.extend([own_position_error, target_position_error])
                speed_errors.extend([own_speed_error, target_speed_error])
                motion_correct.extend([own_motion_ok, target_motion_ok])
                scenario_position_errors.append(pair_position_error)
                scenario_speed_errors.append(pair_speed_error)
                actual_risk_labels.append(true_risk)
                predicted_risk_labels.append(predicted_risk)
                scenario_risk_correct.append(predicted_risk == true_risk)
                dcpa_errors.append(abs(predicted_dcpa - true_dcpa))
                if predicted_tcpa is not None:
                    tcpa_errors.append(abs(float(predicted_tcpa) - true_tcpa_minutes))

                frame_rows.append({
                    "scenario_id": scenario.scenario_id,
                    "mode": mode,
                    "minute_index": minute_index,
                    "classification_attempted": classification_attempted,
                    "classification_expected": classification_expected,
                    "classification_prediction": classification_prediction,
                    "classification_latency_ms": classification_latency_ms,
                    "classification_error": classification_error,
                    "own_position_error_metres": own_position_error,
                    "target_position_error_metres": target_position_error,
                    "own_speed_error_knots": own_speed_error,
                    "target_speed_error_knots": target_speed_error,
                    "own_motion_correct": own_motion_ok,
                    "target_motion_correct": target_motion_ok,
                    "predicted_dcpa_metres": predicted_dcpa,
                    "ground_truth_dcpa_metres": true_dcpa,
                    "predicted_tcpa_minutes": predicted_tcpa,
                    "ground_truth_tcpa_minutes": true_tcpa_minutes,
                    "predicted_risk": predicted_risk,
                    "ground_truth_risk": true_risk,
                    "processing_latency_ms": processing_ms,
                })

            scenario_rows.append({
                "scenario_id": scenario.scenario_id,
                "mode": mode,
                "status": "completed",
                "motion_windows": len(scenario_position_errors),
                "mean_position_error_metres": mean_or_none(scenario_position_errors),
                "mean_speed_error_knots": mean_or_none(scenario_speed_errors),
                "risk_agreement": mean_or_none(scenario_risk_correct),
                "scenario_processing_seconds": time.perf_counter() - scenario_start,
            })

    by_mode = {}
    for mode in modes:
        mode_frames = [row for row in frame_rows if row["mode"] == mode]
        mode_scenarios = [row for row in scenario_rows if row.get("mode") == mode]
        mode_position_errors = [
            value
            for row in mode_frames
            for value in (
                row["own_position_error_metres"],
                row["target_position_error_metres"],
            )
        ]
        mode_speed_errors = [
            value
            for row in mode_frames
            for value in (
                row["own_speed_error_knots"],
                row["target_speed_error_knots"],
            )
        ]
        mode_motion_correct = [
            value
            for row in mode_frames
            for value in (
                row["own_motion_correct"],
                row["target_motion_correct"],
            )
        ]
        mode_tcpa_errors = [
            abs(float(row["predicted_tcpa_minutes"]) - row["ground_truth_tcpa_minutes"])
            for row in mode_frames
            if row["predicted_tcpa_minutes"] is not None
        ]
        by_mode[mode] = {
            "completed_scenarios": sum(row.get("status") == "completed" for row in mode_scenarios),
            "failed_scenarios": sum(row.get("status") == "failed" for row in mode_scenarios),
            "motion_windows": len(mode_frames),
            "gru_position_mae_metres": mean_or_none(mode_position_errors),
            "gru_speed_mae_knots": mean_or_none(mode_speed_errors),
            "gru_motion_state_accuracy": mean_or_none(mode_motion_correct),
            "dcpa_mae_metres": mean_or_none([
                abs(row["predicted_dcpa_metres"] - row["ground_truth_dcpa_metres"])
                for row in mode_frames
            ]),
            "tcpa_mae_minutes": mean_or_none(mode_tcpa_errors),
            "risk_level_agreement": mean_or_none([
                row["predicted_risk"] == row["ground_truth_risk"]
                for row in mode_frames
            ]),
            "motion_and_risk_latency_ms": mean_or_none([
                row["processing_latency_ms"]
                for row in mode_frames
                if not row["classification_attempted"]
            ]),
            "end_to_end_latency_ms": mean_or_none([
                row["processing_latency_ms"]
                for row in mode_frames
                if row["classification_attempted"]
                and row["classification_error"] is None
            ]),
        }

    metrics = {
        "evaluation_notice": (
            "Results are produced from historical AIS replay and constructed/actual research simulations, "
            "not operational Navy radar data."
        ),
        "evaluation_mmsi_policy": "Only MMSIs from test_motion_sequences.npz are eligible.",
        "test_mmsi_count": len(test_mmsi),
        "completed_scenarios": sum(row.get("status") == "completed" for row in scenario_rows),
        "failed_scenarios": sum(row.get("status") == "failed" for row in scenario_rows),
        "motion_windows": len(frame_rows),
        "gru_position_mae_metres": mean_or_none(position_errors),
        "gru_position_median_error_metres": median_or_none(position_errors),
        "gru_speed_mae_knots": mean_or_none(speed_errors),
        "gru_motion_state_accuracy": mean_or_none(motion_correct),
        "dcpa_mae_metres": mean_or_none(dcpa_errors),
        "tcpa_mae_minutes": mean_or_none(tcpa_errors),
        "risk_level_agreement": mean_or_none([
            actual == predicted for actual, predicted in zip(actual_risk_labels, predicted_risk_labels)
        ]),
        "sar_classification_samples_attempted": len(classification_correct) + failed_frames,
        "sar_classification_successful_responses": len(classification_correct),
        "sar_classification_correct_count": int(sum(classification_correct)),
        "sar_classification_accuracy": mean_or_none(classification_correct),
        "sar_classification_latency_ms": mean_or_none(classification_latencies),
        "failed_frames": failed_frames,
        "uncertain_classifications": uncertain_count,
        "motion_and_risk_processing_latency_ms": mean_or_none([
            row["processing_latency_ms"]
            for row in frame_rows
            if not row["classification_attempted"]
        ]),
        "end_to_end_processing_latency_ms": mean_or_none([
            row["processing_latency_ms"]
            for row in frame_rows
            if row["classification_attempted"]
            and row["classification_error"] is None
        ]),
        "by_mode": by_mode,
    }

    (OUTPUT_DIR / "metrics_summary.json").write_text(json.dumps(metrics, indent=2))
    if frame_rows:
        with (OUTPUT_DIR / "frame_results.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)
    if scenario_rows:
        fields = sorted({key for row in scenario_rows for key in row})
        with (OUTPUT_DIR / "scenario_results.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(scenario_rows)

    matrix = confusion_matrix(actual_risk_labels, predicted_risk_labels, labels=RISK_LEVELS) if actual_risk_labels else np.zeros((4, 4), dtype=int)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix)
    axis.set_xticks(range(len(RISK_LEVELS)), labels=RISK_LEVELS, rotation=30)
    axis.set_yticks(range(len(RISK_LEVELS)), labels=RISK_LEVELS)
    axis.set_xlabel("Predicted risk")
    axis.set_ylabel("Ground-truth future risk")
    axis.set_title("Collision-risk confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "risk_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(position_errors, bins=40)
    axis.set_xlabel("Position error (metres)")
    axis.set_ylabel("Frequency")
    axis.set_title("Five-minute GRU position-error distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "position_error_distribution.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
