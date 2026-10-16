# R26-IT-003 — OceanIQ Maritime AI Prototype

OceanIQ runs as **one integrated FastAPI application** (`backend.main:app`) plus a React frontend.  
The Live Simulation / AIS / GRU / DCPA-TCPA module is mounted on the same app (not a second uvicorn process).

## Run the complete prototype (Windows)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
copy backend\.env.example backend\.env
# Set MONGO_URI in backend\.env (required for login)

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Or from the repo root (requires `venv` + root `npm install`):

```powershell
npm run dev
```

Open `http://127.0.0.1:5173`.

## Modules

- Hull Condition, Sea State, Radar Objects, Vessel Detection (manual screens)
- **Chat Assistant** — guided multi-component image analysis (deterministic, no LLM)
- Live Simulation — SAR stream + AIS replay + GRU + DCPA/TCPA
- Database viewer

Chat documentation: [`docs/CHAT_ASSISTANT.md`](docs/CHAT_ASSISTANT.md)  
Simulation details: [`backend/simulation/README.md`](backend/simulation/README.md)

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

SAR/radar images are used for appearance classification. Sequential AIS records provide latitude, longitude, speed and course for motion forecasting. Independent still images are not a source of vessel speed or collision risk. Constructed encounters are labelled simulations.

## Docker

```powershell
docker build -t oceaniq .
docker run -p 8080:8080 --env-file backend\.env oceaniq
```

The container serves the API and the built React SPA from `frontend/dist` on port `8080`.
