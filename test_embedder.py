"""
Step 3 sanity check.

Usage:
    python test_embedder.py path/to/photo.jpg

Runs the full detect -> align -> embed chain on ONE image and confirms:
- embedding shape is (512,)
- embedding is unit-normalized (L2 norm ~= 1.0)
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import FaceDetector, NoFaceDetectedError
from src.embedder import FaceEmbedder


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_embedder.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: could not read image at {image_path}")
        sys.exit(1)

    print("Detecting + aligning...")
    detector = FaceDetector()
    try:
        aligned = detector.detect_and_align_largest(image)
    except NoFaceDetectedError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"  Aligned crop shape: {aligned.shape}")

    print("\nLoading embedder (w600k_r50.onnx)...")
    embedder = FaceEmbedder()

    print("Extracting embedding...")
    embedding = embedder.get_embedding(aligned)

    print(f"\n  Embedding shape: {embedding.shape}  (expect (512,))")
    print(f"  Embedding dtype: {embedding.dtype}")
    print(f"  L2 norm: {np.linalg.norm(embedding):.6f}  (expect ~1.000000)")
    print(f"  First 5 values: {embedding[:5]}")

    print("\nStep 3 verified. Ready for Step 4 (Similarity Module).")


if __name__ == "__main__":
    main()