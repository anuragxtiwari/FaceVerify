"""
Step 3: Embedding Extraction.

Responsibility of this module ONLY:
    input:  a 112x112 aligned face crop (from detector.py)
    output: a 512-D L2-normalized embedding vector

Uses the ArcFace recognition model (w600k_r50.onnx) that ships inside the
buffalo_l pack you already downloaded in Step 1. We load it directly via
insightface.model_zoo instead of going through FaceAnalysis, because we
already have the aligned crop — no need to re-run detection.
"""

import os
import numpy as np
from insightface.model_zoo import get_model

from src.config import EMBEDDING_DIM, INSIGHTFACE_MODEL_NAME, CTX_ID, ALIGNED_FACE_SIZE


def _default_recognition_model_path() -> str:
    """Locate w600k_r50.onnx inside the buffalo_l pack InsightFace already downloaded."""
    home = os.path.expanduser("~")
    path = os.path.join(home, ".insightface", "models", INSIGHTFACE_MODEL_NAME, "w600k_r50.onnx")
    return path


class FaceEmbedder:
    def __init__(self, model_path: str = None):
        model_path = model_path or _default_recognition_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Recognition model not found at {model_path}. "
                f"Did Step 1's test_setup.py run successfully (it downloads this file)?"
            )
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=CTX_ID)

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        aligned_face: a (112, 112, 3) BGR numpy array, as produced by
        FaceDetector.align_face() / detect_and_align_largest().

        Returns: a (512,) float32 numpy array, L2-normalized (unit length).
        """
        if aligned_face is None:
            raise ValueError("aligned_face is None.")
        h, w = aligned_face.shape[:2]
        if (h, w) != (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE):
            raise ValueError(
                f"Expected a {ALIGNED_FACE_SIZE}x{ALIGNED_FACE_SIZE} aligned crop, "
                f"got {h}x{w}. Did you pass a raw (unaligned) image by mistake?"
            )

        # get_feat expects a list/batch of aligned images, returns (N, 512)
        raw_embedding = self.model.get_feat([aligned_face])
        embedding = np.asarray(raw_embedding).flatten().astype(np.float32)

        if embedding.shape[0] != EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected embedding dimension: got {embedding.shape[0]}, expected {EMBEDDING_DIM}."
            )

        # L2 normalize -> unit vector. This is what makes cosine similarity
        # in Step 4 equivalent to a simple dot product.
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise RuntimeError("Embedding norm is zero — something went wrong upstream.")
        normalized_embedding = embedding / norm

        return normalized_embedding