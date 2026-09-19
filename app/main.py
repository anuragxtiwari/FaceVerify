"""
Step 8: API Layer.

Exposes the verification pipeline over HTTP:
    POST /verify   (multipart form: image1, image2 file uploads)
    -> { same_person, similarity, threshold }

Run with:
    uvicorn app.main:app --reload
"""

import sys
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from main import FaceVerificationPipeline
from src.detector import NoFaceDetectedError

app = FastAPI(
    title="Face Verification API",
    description="Upload two face images, get back a same/different person decision.",
    version="1.0.0",
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ... (keep everything else in the file as-is) ...

# Add this right after `app = FastAPI(...)`:
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Loaded ONCE at server startup, reused across every request.
pipeline: FaceVerificationPipeline = None


@app.on_event("startup")
def load_pipeline():
    global pipeline
    print("Loading face verification pipeline (this may take a few seconds)...")
    pipeline = FaceVerificationPipeline()
    print("Pipeline ready.")


class VerifyResponse(BaseModel):
    same_person: bool
    similarity: float
    threshold: float


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

    return VerifyResponse(
        same_person=result["same_person"],
        similarity=result["similarity"],
        threshold=result["threshold"],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}