# Maritime AI Simulation Backend

This service combines three separate research inputs without falsely claiming that random SAR images contain vessel speed:

1. Existing SAR test images are streamed to the radar classification endpoint.
2. Historical AIS trajectories provide sequential position, speed and course data.
3. The selected multi-task GRU forecasts vessel position, speed and motion state five minutes ahead.

A two-vessel controller then calculates relative motion, DCPA, TCPA and a configurable research risk level.

## Architecture

```text
SAR test images -> POST /predict-radar-object -> Bird / Ship / Unknown / Uncertain
Historical AIS -> 10-second replay -> aggregate six ticks per minute -> 10 one-minute observations -> GRU +5-minute forecast
Own AIS + target AIS -> relative motion -> DCPA/TCPA -> Low/Medium/High/Critical
```

The collision-risk level is an analytical decision-support result based on configurable DCPA/TCPA thresholds. It is not a separately trained collision-avoidance model and must not be represented as an autonomous navigation system.

## Required local assets

Runtime assets used by the integrated backend (`backend.main:app`):

```text
ml/live_simulation_radar/ship1 ... ship10/...
ml/external_datasets/ais_motion/processed_AIS_dataset.csv
ml/models/final/ais_motion_gru_under90_deploy.pth
ml/ais_motion/sequences_5min/metadata.json
ml/ais_motion/sequences_5min/test_motion_sequences.npz
ml/models/v6_runs/radar_unknown3/radar_unknown3_best.pth
```

Older experiment checkpoints may remain on disk but are not the active Live Simulation paths.

The GRU deploy checkpoint contains output-target normalisation. `metadata.json` contains the seven input-feature means and standard deviations created from the training split only.

## Run

Simulation routes are mounted on the single OceanIQ app. Do not start a second uvicorn process on port 8001.

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

## Start a fast constructed encounter

```bash
curl -X POST http://127.0.0.1:8000/simulation/start \
  -H 'Content-Type: application/json' \
  -d '{
    "sar_source": "ship",
    "mode": "constructed",
    "real_interval_seconds": 1,
    "simulated_interval_seconds": 10,
    "loop": true,
    "seed": 42,
    "scenario_minutes": 20
  }'
```

Status and live output:

```bash
curl http://127.0.0.1:8000/simulation/status
curl http://127.0.0.1:8000/simulation/latest
curl http://127.0.0.1:8000/simulation/history?limit=10
```

Stop:

```bash
curl -X POST http://127.0.0.1:8000/simulation/stop
```

Note: these simulation endpoints require an authenticated OceanIQ session cookie (and CSRF for POST).

A 1-second real interval advances simulation time by 10 seconds per frame. Use a 10-second real interval during the final viva to demonstrate six SAR frames per minute.

## API endpoints

- `GET /health`
- `POST /simulation/start`
- `POST /simulation/stop`
- `GET /simulation/status`
- `GET /simulation/latest`
- `GET /simulation/history?limit=20`
- `GET /simulation/current-image`
- `POST /predict-vessel-motion`
- `POST /predict-collision-risk`

## Scenario modes

### Constructed

Two real historical AIS motion patterns are selected. The target trajectory is rotated and geographically shifted into a controlled encounter. Speed patterns are retained. This mode is explicitly labelled as constructed.

### Actual

The controller searches for two MMSIs appearing in nearby spatial bins at overlapping times. Some seeds may not produce a suitable pair; constructed mode is the reliable viva mode.

## Security controls

- The integrated backend is intended to bind to `127.0.0.1` for local research use.
- CORS is restricted to the local Vite frontend (plus optional `FRONTEND_ORIGIN`).
- The API does not accept arbitrary filesystem paths.
- SAR input is restricted to the configured test dataset root.
- The GRU checkpoint path is fixed and loaded with `weights_only=True`.
- Mutable simulation state is protected by a lock.
- Only one simulation can run at a time.

## Evaluation

Run without real-time waiting:

```bash
./venv/bin/python ml/src/evaluate_complete_simulation.py \
  --mode both \
  --scenarios 4
```

Keep the integrated OceanIQ backend running unless using `--skip-classification`.

Generated outputs:

```text
ml/evaluation/complete_simulation/metrics_summary.json
ml/evaluation/complete_simulation/frame_results.csv
ml/evaluation/complete_simulation/scenario_results.csv
ml/evaluation/complete_simulation/risk_confusion_matrix.png
ml/evaluation/complete_simulation/position_error_distribution.png
```

## Limitations

- Independent SAR images do not provide valid geographic speed or direction by themselves.
- SAR appearance and target AIS motion are linked by a simulation scenario ID.
- Constructed encounters are simulations, not real Navy encounters.
- DCPA/TCPA thresholds are configurable research thresholds, not universal maritime operating rules.
- The prototype is decision support, not autonomous collision avoidance.

## Final-viva wording

> Because operational Navy radar streams were unavailable, the prototype uses existing SAR target images as a simulated radar-image stream and historical AIS trajectories as sequential vessel-motion input. SAR images are used for target classification, while AIS histories are used for five-minute motion forecasting. Two-vessel scenarios are then evaluated using relative-motion features, DCPA, TCPA and configurable research risk thresholds.
