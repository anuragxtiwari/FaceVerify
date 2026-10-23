
"""
Step 8: API Layer.
"""

import sys
import os
import json
import tempfile
import logging

import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import FaceVerificationPipeline
from src.detector import NoFaceDetectedError
from src.config import INSIGHTFACE_MODEL_NAME, DATA_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Save an uploaded file to a temporary file."""
    suffix = os.path.splitext(upload.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload.file.seek(0)

        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

        return tmp.name


def _remove_temp_file(path: str) -> None:
    """Attempt to remove a temporary file without hiding another error."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        logger.warning("Could not delete temporary file: %s", path)


@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded yet.",
        )

    temp_paths = []

    try:
        path1 = _save_upload_to_temp(image1)
        temp_paths.append(path1)

        path2 = _save_upload_to_temp(image2)
        temp_paths.append(path2)

        # Check that both uploads contain readable image data.
        if cv2.imread(path1) is None or cv2.imread(path2) is None:
            raise HTTPException(
                status_code=422,
                detail="One or both uploaded files are not valid images.",
            )

        result = pipeline.verify(path1, path2)
        return VerifyResponse(**result)

    except HTTPException:
        raise

    except NoFaceDetectedError as e:
        raise HTTPException(
            status_code=422,
            detail=f"No face detected: {e}",
        ) from e

    except Exception:
        logger.exception("Face verification failed.")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during verification.",
        ) from None

    finally:
        for path in temp_paths:
            _remove_temp_file(path)

        for upload in (image1, image2):
            try:
                await upload.close()
            except OSError:
                logger.warning("Could not close an uploaded file.")


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
