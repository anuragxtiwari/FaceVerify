"""
Step 4 sanity check.

Usage:
    python test_similarity.py path/to/photoA.jpg path/to/photoB.jpg

Runs the full chain (detect -> align -> embed -> compare) on TWO images
and prints the similarity score + SAME/DIFFERENT decision.

Test this with:
  - two photos of the SAME person -> similarity should be noticeably higher
  - two photos of DIFFERENT people -> similarity should be noticeably lower
to sanity-check the model is actually discriminating, before we build the
full pipeline in Step 5.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import FaceDetector, NoFaceDetectedError
from src.embedder import FaceEmbedder
from src.similarity import verify_match


def get_embedding_for(path, detector, embedder):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    aligned = detector.detect_and_align_largest(image)
    return embedder.get_embedding(aligned)


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_similarity.py <image1> <image2>")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]

    print("Loading models...")
    detector = FaceDetector()
    embedder = FaceEmbedder()

    print(f"\nProcessing {path1}...")
    try:
        emb1 = get_embedding_for(path1, detector, embedder)
    except NoFaceDetectedError as e:
        print(f"ERROR on {path1}: {e}")
        sys.exit(1)

    print(f"Processing {path2}...")
    try:
        emb2 = get_embedding_for(path2, detector, embedder)
    except NoFaceDetectedError as e:
        print(f"ERROR on {path2}: {e}")
        sys.exit(1)

    result = verify_match(emb1, emb2)

    print("\n--- RESULT ---")
    print(f"Similarity score: {result['similarity']:.4f}")
    print(f"Threshold used:   {result['threshold']}")
    print(f"Decision:         {'SAME PERSON' if result['is_same_person'] else 'DIFFERENT PERSON'}")

    print("\nStep 4 verified. Ready for Step 5 (End-to-End Pipeline) once scores look reasonable.")


if __name__ == "__main__":
    main()