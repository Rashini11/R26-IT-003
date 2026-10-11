# R26-IT-003 — OceanIQ Maritime AI Prototype

The repository now contains a combined maritime simulation service that reuses the existing radar classifier and trained AIS GRU model.

## Run the complete prototype

From the project root:

```bash
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Terminal 1 — existing image-classification backend:

```bash
npm run backend
```

Terminal 2 — SAR/AIS simulation, GRU inference and DCPA/TCPA service:

```bash
.\.venv\Scripts\python.exe -m uvicorn backend.simulation.app:app --host 127.0.0.1 --port 8001
```

Terminal 3 — dashboard:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` and select **Live Simulation**.

Full architecture, API examples, limitations and evaluation commands are documented in [`backend/simulation/README.md`](backend/simulation/README.md).

## Vessel inspections

The existing Vessel Detection model and `/predict-boat-detection` endpoint remain unchanged. The authenticated dashboard now adds AIS parsing and operator verification, IMO/MMSI-based vessel matching, separate vessel and inspection records, and searchable inspection history through `/vessel-inspections`.

Inspection images are stored in Cloudinary as authenticated (non-public) assets; MongoDB stores the Cloudinary public ID and the dashboard retrieves image bytes through the existing authenticated backend endpoint. Set these values in `backend/.env` or the backend process environment:

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

Get the cloud name, API key, and API secret from the Cloudinary Console. Keep the API secret private and never put it in frontend code or source control. Uploads use deterministic public IDs so retries do not create duplicate assets. The backend returns HTTP 503 when Cloudinary is not configured and does not save vessel or inspection records if image upload fails. Install backend dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

The Google Drive downloader remains only to support inspection records created before this storage change; new inspection images use Cloudinary.

## Research boundary

SAR images are streamed for target appearance classification. Sequential AIS records provide latitude, longitude, speed and course for motion forecasting. Independent SAR images are not presented as a source of vessel speed. Constructed encounters are clearly labelled simulations.
