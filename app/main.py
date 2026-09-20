"""
Step 8: API Layer.
"""

import sys
import os
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import FaceVerificationPipeline
from src.detector import NoFaceDetectedError
from src.config import INSIGHTFACE_MODEL_NAME, DATA_DIR

app = FastAPI(
    title="Face Verification API",
    description="Upload two face images, get back a same/different person decision.",
    version="1.0.0",
)

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

pipeline: FaceVerificationPipeline = None


@app.on_event("startup")
def load_pipeline():
    global pipeline
    print("Loading face verification pipeline...")
    pipeline = FaceVerificationPipeline()
    print("Pipeline ready.")


class VerifyResponse(BaseModel):
    same_person: bool
    similarity: float
    threshold: float
    aligned1: str
    aligned2: str


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    path1 = _save_upload_to_temp(image1)
    path2 = _save_upload_to_temp(image2)

    try:
        result = pipeline.verify(path1, path2)
    except NoFaceDetectedError as e:
        raise HTTPException(status_code=422, detail=f"No face detected: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        os.remove(path1)
        os.remove(path2)

    return VerifyResponse(**result)


@app.get("/model-info")
async def model_info():
    info = {"model_pack": INSIGHTFACE_MODEL_NAME}
    threshold_path = os.path.join(DATA_DIR, "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            data = json.load(f)
        info.update({
            "eer": data.get("eer"),
            "auc": data.get("auc"),
            "n_pairs": data.get("n_pairs"),
        })
    return info


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))