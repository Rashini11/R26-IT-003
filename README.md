# R26-IT-003 — OceanIQ Maritime AI Prototype

The repository now contains a combined maritime simulation service that reuses the existing radar classifier and trained AIS GRU model.

## Run the complete prototype

From the project root:

```bash
./venv/bin/python -m pip install -r backend/requirements.txt
```

Terminal 1 — existing image-classification backend:

```bash
./venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — SAR/AIS simulation, GRU inference and DCPA/TCPA service:

```bash
./venv/bin/python -m uvicorn backend.simulation.app:app --host 127.0.0.1 --port 8001
```

Terminal 3 — dashboard:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` and select **Live Simulation**.

Full architecture, API examples, limitations and evaluation commands are documented in [`backend/simulation/README.md`](backend/simulation/README.md).

## Research boundary

SAR images are streamed for target appearance classification. Sequential AIS records provide latitude, longitude, speed and course for motion forecasting. Independent SAR images are not presented as a source of vessel speed. Constructed encounters are clearly labelled simulations.
