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

## Vessel Detection

Vessel Detection analyzes still images with `model/ship-detection-model.pt` and
can analyze a separate stern image with `model/Vessel-info-detector-model.pt`
for vessel name, port and IMO details. Ship Detection also accepts MP4, MOV, AVI,
MKV and WEBM videos up to 100 MB and 5 minutes, sampling one frame per second.
Video and sampled frames are temporary inference inputs only: they are not sent
to Cloudinary or saved in the database. Image-based results can be reviewed,
edited and submitted; saved records can be updated or deleted from the history
table.

Set valid Cloudinary credentials in `backend/.env` before submitting records so
their images can be stored. Use either `CLOUDINARY_URL` or the
`CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY` and `CLOUDINARY_API_SECRET`
variables. Set `MONGO_URI` to persist records in MongoDB; without it, the
backend uses its local JSON history fallback.

## Research boundary

SAR images are streamed for target appearance classification. Sequential AIS records provide latitude, longitude, speed and course for motion forecasting. Independent SAR images are not presented as a source of vessel speed. Constructed encounters are clearly labelled simulations.
