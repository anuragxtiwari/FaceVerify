"""
Step 2: Face Detection + Alignment.

Responsibility of this module ONLY:
    input:  a raw image (any size, containing 0+ faces)
    output: a 112x112 aligned face crop, ready to feed into the embedder (Step 3)

It does NOT compute embeddings here — that's embedder.py's job.
We deliberately load only the 'detection' model (not recognition/genderage/etc.)
so this stays fast and single-purpose.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from src.config import INSIGHTFACE_MODEL_NAME, CTX_ID, DETECTION_SIZE, ALIGNED_FACE_SIZE


class NoFaceDetectedError(Exception):
    """Raised when zero faces are found in an image."""
    pass


class FaceDetector:
    def __init__(self):
        # allowed_modules=['detection'] -> only loads det_10g.onnx (SCRFD detector).
        # This model already outputs 5-point landmarks (eyes, nose, mouth corners)
        # alongside each bounding box, which is all we need for alignment.
        self.app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            allowed_modules=["detection"],
        )
        self.app.prepare(ctx_id=CTX_ID, det_size=DETECTION_SIZE)

    def detect_faces(self, image: np.ndarray):
        """
        Run detection on a BGR image (as loaded by cv2.imread).
        Returns a list of insightface Face objects (bbox, kps, det_score).
        """
        if image is None:
            raise ValueError("Received None as image input.")
        faces = self.app.get(image)
        return faces

    def align_face(self, image: np.ndarray, face) -> np.ndarray:
        """
        Given the original image and a detected Face object, crop + align
        it to a 112x112 image using the standard ArcFace 5-point template.
        """
        aligned = face_align.norm_crop(image, landmark=face.kps, image_size=ALIGNED_FACE_SIZE)
        return aligned

    def detect_and_align_largest(self, image: np.ndarray) -> np.ndarray:
        """
        Convenience method for the verification pipeline (Step 5):
        detect all faces, pick the LARGEST one (by bbox area) — this is the
        standard assumption for verification photos (one subject per image) —
        and return its 112x112 aligned crop.

        Raises NoFaceDetectedError if no face is found.
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            raise NoFaceDetectedError("No face detected in image.")

        def bbox_area(f):
            x1, y1, x2, y2 = f.bbox
            return (x2 - x1) * (y2 - y1)

        largest_face = max(faces, key=bbox_area)
        return self.align_face(image, largest_face)

    def detect_and_align_from_path(self, image_path: str) -> np.ndarray:
        """Load an image from disk, then detect+align the largest face."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")
        return self.detect_and_align_largest(image)