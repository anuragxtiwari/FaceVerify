"""
Step 5: End-to-End Pipeline.

verify(image1_path, image2_path) -> result dict, including base64-encoded
aligned face crops so callers (like the frontend) can show exactly what
was compared, not just the raw uploaded photos.
"""

import sys
import base64
import cv2

from src.detector import FaceDetector, NoFaceDetectedError
from src.embedder import FaceEmbedder
from src.similarity import verify_match


def _encode_aligned_face(aligned_image) -> str:
    """Encode a 112x112 aligned face crop as a base64 JPEG data URI."""
    success, buffer = cv2.imencode(".jpg", aligned_image)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class FaceVerificationPipeline:
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

    def _load_align_and_embed(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        aligned = self.detector.detect_and_align_largest(image)
        embedding = self.embedder.get_embedding(aligned)
        return embedding, aligned

    def verify(self, image1_path: str, image2_path: str, threshold: float = None) -> dict:
        embedding1, aligned1 = self._load_align_and_embed(image1_path)
        embedding2, aligned2 = self._load_align_and_embed(image2_path)

        result = verify_match(embedding1, embedding2, threshold=threshold)

        return {
            "same_person": result["is_same_person"],
            "similarity": round(result["similarity"], 4),
            "threshold": result["threshold"],
            "image1": image1_path,
            "image2": image2_path,
            "aligned1": _encode_aligned_face(aligned1),
            "aligned2": _encode_aligned_face(aligned2),
        }


def verify(image1_path: str, image2_path: str, threshold: float = None) -> dict:
    pipeline = FaceVerificationPipeline()
    return pipeline.verify(image1_path, image2_path, threshold=threshold)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <image1> <image2>")
        sys.exit(1)

    image1_path, image2_path = sys.argv[1], sys.argv[2]

    print("Loading pipeline (this loads detection + recognition models)...")
    pipeline = FaceVerificationPipeline()

    print(f"\nVerifying:\n  Image 1: {image1_path}\n  Image 2: {image2_path}\n")
    try:
        result = pipeline.verify(image1_path, image2_path)
    except (NoFaceDetectedError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("--- RESULT ---")
    print(f"Same person:  {result['same_person']}")
    print(f"Similarity:   {result['similarity']}")
    print(f"Threshold:    {result['threshold']}")