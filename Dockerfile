FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download InsightFace models at build time so the container doesn't
# need internet access or a slow first-request download at runtime.
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_s'); app.prepare(ctx_id=-1, det_size=(320,320))"

EXPOSE 7860

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}