# ============================================================
# FRONTEND BUILD
# ============================================================

FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./

RUN npm ci

COPY frontend/ ./

RUN npm run build


# ============================================================
# OCEANIQ BACKEND
# ============================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY model/ /app/model/
COPY ml/ /app/ml/

COPY --from=frontend-build \
    /app/frontend/dist \
    /app/frontend/dist

EXPOSE 8080

CMD [
  "python",
  "-m",
  "uvicorn",
  "backend.main:app",
  "--host",
  "0.0.0.0",
  "--port",
  "8080"
]
