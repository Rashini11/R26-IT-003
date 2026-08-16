🌊 OceanIQ

AI-Powered Maritime Intelligence, Vessel Monitoring & Collision-Risk Decision Support Platform

OceanIQ is an integrated maritime AI research platform developed to improve situational awareness, vessel monitoring, target classification, motion forecasting, collision-risk assessment, and vessel-condition analysis through a unified web-based system.

Rather than operating as isolated AI models, OceanIQ brings multiple maritime intelligence components into a single FastAPI backend and unified React frontend, creating one platform for marine inspection, vessel awareness, and decision support.

Research Project: R26-IT-003
System: OceanIQ
Architecture: React + FastAPI + MongoDB + PyTorch + TensorFlow + Ultralytics YOLO
Deployment Mode: Single integrated backend

✨ Key Capabilities

OceanIQ currently integrates:

🌊 Sea-State Classification

🛰️ Radar / SAR Object Classification

🚢 AIS-Based Vessel Motion Forecasting

⚠️ DCPA / TCPA Collision-Risk Evaluation

🛥️ Vessel / Boat Detection

🔧 Underwater Hull Condition Assessment

📡 Live Maritime Simulation

👤 Secure User Authentication

📝 User Registration

🛡️ Admin Approval & Access Control

🗄️ MongoDB-Based User / Session / History Storage

📊 Prediction History & Decision Support

📄 Sea-State PDF Report Generation

🧠 System Overview

OceanIQ combines multiple maritime AI components into one operational platform.

                         O C E A N I Q
                Maritime Intelligence Platform

                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
     Environment          Vessel Awareness      Vessel Health
          │                    │                    │
    Sea-State AI       Radar / SAR AI       Hull Assessment
                               │
                               ├── AIS Motion Forecasting
                               │
                               ├── DCPA / TCPA
                               │
                               └── Collision Risk
                               │
                         Live Simulation
                               │
                         Human Operator

🛰️ Radar / SAR Object Classification

The radar component classifies maritime radar/SAR observations into:

Ship

Bird

Unknown

Uncertain — when the final models disagree

Final Radar Architecture

OceanIQ uses a dual-model verification strategy:

Radar / SAR Image
       │
       ├── YOLO11-Medium Classification
       │
       └── DeeperCNN Verification
                    │
                    ↓
              Model Agreement
                    │
          ┌─────────┴─────────┐
          │                   │
       Agree               Disagree
          │                   │
      Accept              Uncertain

Radar Results

Metric

Result

YOLO11-Medium Accuracy

99.92%

DeeperCNN Accuracy

94.58%

Final Evaluation Samples

2,400

Model Agreements

2,268

Automatic Coverage

94.50%

Manual Review / Uncertain

132 samples (5.50%)

Accepted-Prediction Accuracy

100%

Important: The system does not claim 100% overall radar accuracy.
The correct result is 100% accepted-prediction accuracy at 94.50% automatic coverage on the prepared internal evaluation dataset.

Final Radar Models

ml/models/final/yolo11_medium_best.pt
ml/models/final/deepercnn_best.pth

🚢 AIS Vessel Motion Forecasting

OceanIQ uses historical AIS vessel data to estimate short-term future vessel motion.

Dataset

Raw AIS records: 1,098,966

Historical observation window: 10 minutes

Prediction horizon: 5 minutes

Input features: 7

Test vessels: 768 unseen vessels

Sequence Dataset

Dataset

Number of Sequences

Training

377,084

Validation

81,670

Test

81,184

GRU vs LSTM Evaluation

Metric

GRU

LSTM

Motion-State Accuracy

93.73%

93.70%

Macro F1

0.8800

0.8783

Position MAE

82.34 m

96.42 m

Median Position Error

11.44 m

11.37 m

Speed MAE

0.427 knots

0.532 knots

The GRU was selected as the final vessel-motion forecasting model because it achieved better overall forecasting performance, particularly in mean position and speed error.

Final AIS Model

ml/models/final/ais_motion_gru_best.pth

⚠️ DCPA / TCPA Collision-Risk Evaluation

OceanIQ includes a collision-risk subsystem based on relative vessel motion.

DCPA

Distance at Closest Point of Approach represents the predicted minimum separation between the own vessel and a target vessel.

TCPA

Time to Closest Point of Approach represents the estimated time until the minimum separation occurs.

The system uses:

Own Vessel Position + Motion
            │
            +
Target Vessel Position + Motion
            │
            ↓
     Relative Motion
            │
            ↓
        DCPA / TCPA
            │
            ↓
    Collision-Risk Engine

Risk is classified into:

LOW

MEDIUM

HIGH

CRITICAL

DCPA/TCPA are deterministic mathematical calculations, not separately trained machine-learning models.

The GRU motion forecast can be used alongside the collision-risk workflow to estimate how vessel movement may develop over the prediction horizon.

🌊 Sea-State Classification

The Sea-State module processes maritime images and provides:

Sea-state prediction

Confidence scores

Image validation

Image quality analysis

Optional image enhancement

Operational risk indication

Marine-operation suitability

Recommendation generation

Prediction history

PDF report generation

Prediction history can be stored in MongoDB.

🔧 Underwater Hull Condition Assessment

OceanIQ includes an underwater hull inspection component for analyzing vessel hull imagery.

The integrated workflow supports:

Hull Image
    ↓
AI Inspection
    ↓
Hull Condition / Defect Analysis
    ↓
Decision Support

Integrated model files include:

model/hull_model.h5
model/hull_model.keras

🛥️ Vessel / Boat Detection

A vessel detection model is included for detecting boats/vessels from visual imagery.

Integrated model:

model/boat_detection.pt

📡 Live Maritime Simulation

OceanIQ contains an integrated simulation environment under:

backend/simulation/

The simulation supports maritime encounter testing and collision-risk analysis.

Capabilities include:

Start / stop simulation

Live vessel-state retrieval

Simulation history

Current image retrieval

Vessel-motion prediction

Collision-risk analysis

Configurable DCPA / TCPA thresholds

Constructed and actual-data modes

The simulation is integrated into the same FastAPI backend — there is no separate simulation backend.

🔐 Authentication & Access Control

OceanIQ uses a backend-integrated authentication system.

Authentication is not only a frontend restriction. Every protected backend endpoint validates the authenticated session.

Authentication Features

MongoDB-backed user accounts

Argon2 password hashing

Server-side session storage

HttpOnly authentication cookies

Session expiry

Login / logout

/auth/me session verification

User registration

Failed-login protection

Temporary account lockout

Password-change support

Session revocation

Signup & Admin Approval

Create Profile
      ↓
MongoDB User
      ↓
Status = Pending
      ↓
Admin Review
      ↓
┌──────────────┬──────────────┐
│              │              │
Read Only   Read / Write    Reject

New users cannot assign their own access level.

Access Levels

Administrator

Manage registered users

Approve / reject accounts

Assign access levels

Enable / disable accounts

Full system access

Read Only

Can view dashboards, prediction history, system status and permitted read endpoints.

Read / Write

Can additionally upload images, run permitted analyses/predictions and control permitted simulation operations.

Access control is enforced by the FastAPI backend, not only by React.

🗄️ MongoDB

MongoDB is used for application data such as:

User profiles

Authentication sessions

Access-control information

Sea-State prediction history

Typical collections include:

users
auth_sessions
sea_state_predictions
hull_defect_predictions
boat_detections
radar_predictions

🏗️ Architecture

                        React Frontend
                      http://localhost:5173
                              │
                              │ HTTP API
                              ▼
                     FastAPI Backend :8000
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    Authentication        AI Models             Simulation
        │                     │                     │
      MongoDB          ┌──────┼──────┐        DCPA / TCPA
                       │      │      │
                     Radar   AIS   Sea/Hull

OceanIQ intentionally uses one integrated FastAPI backend.

🛠️ Technology Stack

Frontend

React

Vite

JavaScript

CSS

Lucide React

jsPDF

Backend

Python

FastAPI

Uvicorn

PyMongo

python-dotenv

pwdlib / Argon2

AI / ML

PyTorch

TorchVision

Ultralytics YOLO

TensorFlow / Keras

Scikit-learn

OpenCV

NumPy

Pandas

Database

MongoDB

📁 Project Structure

R26-IT-003/
│
├── backend/
│   ├── main.py
│   ├── auth.py
│   ├── create_admin.py
│   ├── migrate_auth_access.py
│   ├── requirements.txt
│   ├── simulation/
│   ├── model/
│   ├── utils/
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── LiveSimulation.jsx
│   │   ├── components/
│   │   └── context/
│   ├── package.json
│   └── vite.config.*
│
├── ml/
│   ├── models/
│   │   └── final/
│   │       ├── yolo11_medium_best.pt
│   │       ├── deepercnn_best.pth
│   │       └── ais_motion_gru_best.pth
│   └── src/
│
├── model/
│   ├── hull_model.keras
│   ├── image_only_model.pth
│   └── boat_detection.pt
│
└── README.md

🚀 Getting Started

1. Clone the repository

git clone <YOUR_REPOSITORY_URL>
cd R26-IT-003

2. Create a Python virtual environment

python3.11 -m venv venv

macOS / Linux

source venv/bin/activate

Windows

venv\Scripts\activate

3. Install backend dependencies

pip install -r backend/requirements.txt

4. Configure environment variables

Create:

backend/.env

using backend/.env.example as the template.

MONGO_URI=your_mongodb_connection_string
MONGO_DB_NAME=marine_ai_db

AUTH_COOKIE_NAME=oceaniq_session
AUTH_SESSION_HOURS=8
AUTH_COOKIE_SECURE=false
AUTH_MAX_FAILED_ATTEMPTS=5
AUTH_LOCK_MINUTES=10

Never commit the real .env file or MongoDB credentials.

For localhost HTTP development:

AUTH_COOKIE_SECURE=false

For HTTPS deployment:

AUTH_COOKIE_SECURE=true

👤 Create the Initial Administrator

After MongoDB is configured:

python -m backend.create_admin

Passwords are stored as hashes rather than plaintext.

▶️ Run the Backend

From the repository root:

python -m uvicorn backend.main:app   --host 127.0.0.1   --port 8000   --reload

Backend:

http://localhost:8000

FastAPI documentation:

http://localhost:8000/docs

💻 Run the Frontend

cd frontend
npm install
npm run dev

Frontend:

http://localhost:5173

During local development, use localhost consistently for browser/API URLs to avoid authentication-cookie host mismatches.

🔌 Main API Endpoints

System

GET  /
GET  /health

Authentication

POST /auth/signup
POST /auth/login
POST /auth/logout
GET  /auth/me
POST /auth/change-password

Administration

GET  /auth/admin/users

Additional admin endpoints are used to approve users, assign access levels, reject profiles and manage accounts.

AI / Prediction

POST /predict-hull-defect
POST /predict-sea-state
POST /predict-boat-detection
POST /predict-radar-object
POST /predict-vessel-motion
POST /predict-collision-risk

Simulation

GET  /simulation/health
POST /simulation/start
POST /simulation/stop
GET  /simulation/status
GET  /simulation/latest
GET  /simulation/history
GET  /simulation/current-image

🔬 Research Contributions

Dual-model radar classification with disagreement-based uncertainty handling

AIS-based short-term vessel motion forecasting

GRU vs LSTM comparative evaluation

DCPA/TCPA-based collision-risk evaluation

Integrated maritime encounter simulation

Sea-state operational analysis

AI-supported hull-condition assessment

Vessel detection

Unified maritime web platform

Backend-enforced authentication and access control

📊 Key Research Results

Component

Result

YOLO11-Medium Radar Accuracy

99.92%

DeeperCNN Radar Accuracy

94.58%

Radar Automatic Coverage

94.50%

Radar Accepted-Prediction Accuracy

100%

Radar Manual Review Rate

5.50%

GRU Motion-State Accuracy

93.73%

GRU Macro F1

0.8800

GRU Position MAE

82.34 m

GRU Median Position Error

11.44 m

GRU Speed MAE

0.427 kn

AIS Test Sequences

81,184

Unseen Test Vessels

768

🧪 Research Integrity

Some OceanIQ components operate on separate datasets and do not necessarily represent observations of the same physical vessel at the same real-world timestamp.

The simulation environment is therefore used for controlled integration and collision-risk scenarios where required.

Model accuracy values should be interpreted within the context of their respective prepared datasets and evaluation procedures.

100% accepted-prediction accuracy for radar classification does not mean 100% overall accuracy.

It represents the accuracy of the subset automatically accepted by the dual-model agreement mechanism.

🔮 Future Development

Multi-Sensor Fusion Engine

Unified vessel target association

Radar + AIS + drone data fusion

COLREG-aware encounter interpretation

Environmental adjustment of collision risk

Maritime digital twin

Prediction uncertainty visualization

Cross-sensor anomaly detection

Incident replay / maritime black-box logging

Explainable AI dashboards

Research ablation and evaluation dashboard

Radar Target
     +
AIS Vessel
     +
Drone Detection
     ↓
Unified Maritime Target
     ↓
Motion Forecasting
     ↓
Collision-Risk Decision Support

⚠️ Intended Use

OceanIQ is a research and decision-support prototype.

It is not intended to replace certified marine navigation equipment, COLREG-compliant professional judgement, qualified bridge personnel, approved collision-avoidance systems, or certified vessel inspection procedures.

The human operator remains the final decision-maker.

🔒 Security Notes

Never commit backend/.env

Never expose MongoDB connection strings

Use strong administrator passwords

Use HTTPS for deployed environments

Set AUTH_COOKIE_SECURE=true when using HTTPS

Do not commit node_modules

Do not commit temporary runtime prediction files unless intentionally required

📜 License

This repository was developed for academic research under R26-IT-003.

Add the appropriate project license here if the repository is intended for public distribution.

👥 Research Team

Developed as part of the R26-IT-003 Research Project.

Add individual researcher names, student IDs, supervisors, and component responsibilities here before final publication.

<div align="center">

🌊 OceanIQ

From isolated maritime AI models to integrated maritime intelligence.

Radar • AIS • Sea State • Hull • Vessel Detection • Simulation • Collision Risk

</div>
