# OceanIQ Chat Assistant (v1)

Deterministic, template-driven assistant that runs the **real** OceanIQ image models.  
**No LLM**, no external generative APIs, and no component retraining.

## How v1 works

1. User opens **Chat Assistant** in the React sidebar.
2. User selects an analysis type and uploads an image (either order).
3. Frontend calls authenticated `POST /chat/analyze` with CSRF + session cookie.
4. Backend validates the image bytes, resolves routing (manual selection only), and dispatches to the shared component runner registered from `backend.main`.
5. Response includes a friendly summary, structured model outputs, optional charts / Grad-CAM / detection overlay, and follow-up actions.
6. Follow-ups call `POST /chat/follow-up` against a **specific stored result payload** (not always “latest only” on the server — the client sends the selected result).

Chat UI state is **in-memory for the browser session**. It is not written to `localStorage`. Component prediction persistence may still occur through existing MongoDB/JSON history when those paths run.

## Supported components

| Chat analysis id | Manual module | Shared runner |
|------------------|---------------|---------------|
| `radar` | Radar Objects | `run_radar_object_analysis` → `classify_radar_image_path` (`radar_unknown3`) |
| `vessel` | Vessel Detection | `run_boat_detection_analysis` |
| `sea` | Sea State | `run_sea_state_analysis` |
| `hull` | Hull Defect | `run_hull_defect_analysis` |

AIS motion / GRU / DCPA-TCPA remain in **Live Simulation**. The assistant will not invent encounter risk from a still image.

## Follow-up actions

- Explain this result
- Show available probabilities
- Explain confidence
- Explain unknown / uncertain
- Analyse another image
- Change analysis type

Short text commands are matched with regex intents (`backend/chatbot/intents.py`). Unsupported free text receives a capability notice — the assistant does **not** claim open-ended NLU.

## Automatic routing (future)

`backend/chatbot/routing.py` defines `ImageComponentRouter`:

- Today: `classify_image()` reports unavailable; missing selection → `needs_selection`.
- Future: load a router checkpoint + preprocessing metadata, return `suggested` / `uncertain` / `unsupported` / `needs_confirmation`.
- Do not compare unrelated component-model confidences or treat filenames as ground truth.

## Required checkpoints

- `model/image_only_model.pth` + `model/label_map.json`
- `model/hull_model.keras`
- `backend/best.pt`
- `ml/models/v6_runs/radar_unknown3/radar_unknown3_best.pth`
- (Simulation only, not required for chat) AIS CSV + `ml/live_simulation_radar/` + GRU deploy weights

## Environment

See `backend/.env.example`. Authentication **requires** MongoDB (`MONGO_URI`).

## Windows local run

```powershell
cd "C:\path\to\R26-IT-003"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
copy backend\.env.example backend\.env
# Edit backend\.env with your MongoDB URI, then create an admin:
python -m backend.create_admin

# Terminal 1 — single integrated backend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open the app, sign in, choose **Chat Assistant**.

Root helper (after `npm install` at repo root and venv present):

```powershell
npm run dev
```

## Current limitations

- No automatic image-type router.
- No LLM explanations.
- Live Simulation SAR/AIS datasets may be missing on this laptop; chat image analyses do not depend on them.
- Chat transcripts are not permanently stored by the assistant itself.
