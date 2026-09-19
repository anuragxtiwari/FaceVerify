"""
Step 2 sanity check.

Usage:
    python test_detector.py path/to/some_photo_with_a_face.jpg

What it does:
1. Loads the image
2. Runs detection -> prints how many faces found, bbox, confidence, landmarks
3. Aligns the largest face -> saves the 112x112 result to data/aligned_debug.jpg
   so you can open it and visually confirm the crop looks like a proper,
   centered, upright face.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `from src...` when run directly

from src.detector import FaceDetector, NoFaceDetectedError
from src.config import DATA_DIR


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_detector.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Loading: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: could not read image at {image_path}")
        sys.exit(1)
    print(f"  Image shape: {image.shape}")

    print("\nInitializing detector (loads det_10g.onnx)...")
    detector = FaceDetector()

    print("\nRunning detection...")
    faces = detector.detect_faces(image)
    print(f"  Faces found: {len(faces)}")
    for i, f in enumerate(faces):
        print(f"  Face {i}: bbox={f.bbox.astype(int).tolist()}, det_score={f.det_score:.3f}")
        print(f"           kps (5-point landmarks)=\n{f.kps}")

    if len(faces) == 0:
        print("\nNo face detected — try a clearer / more frontal photo.")
        sys.exit(1)

    print("\nAligning largest face...")
    try:
        aligned = detector.detect_and_align_largest(image)
    except NoFaceDetectedError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Aligned crop shape: {aligned.shape}  (expect 112x112x3)")

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "aligned_debug.jpg")
    cv2.imwrite(out_path, aligned)
    print(f"\nSaved aligned face to: {out_path}")
    print("Open that file and confirm: face is centered, eyes roughly level, 112x112.")
    print("\nStep 2 verified. Ready for Step 3 (Embedding Extraction) once you confirm the crop looks correct.")


if __name__ == "__main__":
    main()