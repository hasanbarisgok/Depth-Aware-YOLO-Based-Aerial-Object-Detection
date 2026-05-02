FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    MODEL_WEIGHTS=models/obb_ha_hb_best.pt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-railway.txt ./requirements-railway.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-railway.txt

COPY scripts/api_server.py ./scripts/api_server.py
COPY scripts/depth_overlay.py ./scripts/depth_overlay.py
COPY OBB_HA_HB/results/best.pt ./models/obb_ha_hb_best.pt
COPY OBB_HA_HB/results/last.pt ./models/obb_ha_hb_last.pt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn scripts.api_server:app --host 0.0.0.0 --port ${PORT}"]
