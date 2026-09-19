"""
Step 3: Embedding Extraction.

Responsibility of this module ONLY:
    input:  a 112x112 aligned face crop (from detector.py)
    output: a 512-D L2-normalized embedding vector

Loads the recognition model from whichever InsightFace pack is configured
(buffalo_l, buffalo_s, etc.) — we scan the pack's folder and ask InsightFace
which file is the recognition model, instead of hardcoding a filename that
only exists in one specific pack.
"""

import os
import glob
import numpy as np
from insightface.model_zoo import get_model
from src import ort_memory_patch  # noqa: F401 — must run before any InsightFace model loads
from src.config import EMBEDDING_DIM, INSIGHTFACE_MODEL_NAME, CTX_ID, ALIGNED_FACE_SIZE


def _find_recognition_model_path() -> str:
    """
    Scan the downloaded model pack folder and return the path to whichever
    .onnx file InsightFace identifies as the 'recognition' task model.
    """
    home = os.path.expanduser("~")
    pack_dir = os.path.join(home, ".insightface", "models", INSIGHTFACE_MODEL_NAME)
    if not os.path.isdir(pack_dir):
        raise FileNotFoundError(
            f"Model pack folder not found at {pack_dir}. "
            f"Did test_setup.py run successfully for '{INSIGHTFACE_MODEL_NAME}'?"
        )

    for onnx_path in sorted(glob.glob(os.path.join(pack_dir, "*.onnx"))):
        try:
            candidate = get_model(onnx_path)
        except Exception:
            continue
        if getattr(candidate, "taskname", None) == "recognition":
            return onnx_path

    raise FileNotFoundError(
        f"No recognition model found inside {pack_dir}. "
        f"Contents: {os.listdir(pack_dir)}"
    )


class FaceEmbedder:
    def __init__(self, model_path: str = None):
        model_path = model_path or _find_recognition_model_path()
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=CTX_ID)

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        aligned_face: a (112, 112, 3) BGR numpy array, as produced by
        FaceDetector.align_face() / detect_and_align_largest().

        Returns: a (EMBEDDING_DIM,) float32 numpy array, L2-normalized.
        Note: buffalo_s's embedding dimension may differ from buffalo_l's
        512 — we read the actual output size from the model rather than
        assuming, and just report whatever it is.
        """
        if aligned_face is None:
            raise ValueError("aligned_face is None.")
        h, w = aligned_face.shape[:2]
        if (h, w) != (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE):
            raise ValueError(
                f"Expected a {ALIGNED_FACE_SIZE}x{ALIGNED_FACE_SIZE} aligned crop, "
                f"got {h}x{w}. Did you pass a raw (unaligned) image by mistake?"
            )

        raw_embedding = self.model.get_feat([aligned_face])
        embedding = np.asarray(raw_embedding).flatten().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise RuntimeError("Embedding norm is zero — something went wrong upstream.")
        normalized_embedding = embedding / norm

        return normalized_embedding