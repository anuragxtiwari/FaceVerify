"""
Step 5: End-to-End Pipeline.

This is the single entry point for the whole system:
    verify(image1_path, image2_path) -> result dict

Internally: detect+align both images -> embed both -> compare -> return result.
Everything else (detector.py, embedder.py, similarity.py) stays independently
testable, but this is what the API (Step 8) and evaluation script (Step 6)
will actually call.
"""

import sys
import cv2

from src.detector import FaceDetector, NoFaceDetectedError
from src.embedder import FaceEmbedder
from src.similarity import verify_match


class FaceVerificationPipeline:
    """
    Loads all models ONCE. Reuse the same instance across many verify() calls
    instead of constructing a new one each time — model loading is the
    expensive part (~1-2s), inference on an aligned face is fast (~50-100ms).
    """

    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

    def _load_and_embed(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        aligned = self.detector.detect_and_align_largest(image)
        embedding = self.embedder.get_embedding(aligned)
        return embedding

    def verify(self, image1_path: str, image2_path: str, threshold: float = None) -> dict:
        """
        Full pipeline: image path -> image path -> verification result.

        Returns:
            {
                "same_person": bool,
                "similarity": float,
                "threshold": float,
                "image1": str,
                "image2": str,
            }

        Raises:
            FileNotFoundError       - if either image path is invalid
            NoFaceDetectedError     - if either image has no detectable face
        """
        embedding1 = self._load_and_embed(image1_path)
        embedding2 = self._load_and_embed(image2_path)

        result = verify_match(embedding1, embedding2, threshold=threshold)

        return {
            "same_person": result["is_same_person"],
            "similarity": round(result["similarity"], 4),
            "threshold": result["threshold"],
            "image1": image1_path,
            "image2": image2_path,
        }


def verify(image1_path: str, image2_path: str, threshold: float = None) -> dict:
    """
    Convenience function for one-off use (e.g. quick scripts, notebooks).
    For repeated calls (evaluation, API server), instantiate
    FaceVerificationPipeline() once and reuse it instead — this function
    reloads all models on every call, which is wasteful in a loop.
    """
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
    except NoFaceDetectedError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("--- RESULT ---")
    print(f"Same person:  {result['same_person']}")
    print(f"Similarity:   {result['similarity']}")
    print(f"Threshold:    {result['threshold']}")