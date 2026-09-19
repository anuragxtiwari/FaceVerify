"""
Step 1 sanity check.

Run this AFTER `pip install -r requirements.txt` to confirm:
1. All core libraries import correctly.
2. InsightFace can download and load the buffalo_l model pack.
3. The FaceAnalysis app initializes without errors.

This does NOT do detection/alignment/embedding yet — that's Step 2+.
It only proves your environment is correctly set up.
"""

import sys


def check_imports():
    print("Checking imports...")
    try:
        import cv2
        import numpy as np
        import insightface
        from insightface.app import FaceAnalysis
        import onnxruntime
        print(f"  OK - opencv-python {cv2.__version__}")
        print(f"  OK - numpy {np.__version__}")
        print(f"  OK - insightface {insightface.__version__}")
        print(f"  OK - onnxruntime {onnxruntime.__version__}")
        return True
    except ImportError as e:
        print(f"  FAILED - {e}")
        print("  -> Run: pip install -r requirements.txt")
        return False


def check_model_load():
    print("\nInitializing InsightFace FaceAnalysis (buffalo_l)...")
    print("(First run downloads ~300MB of model weights — this may take a while.)")
    try:
        from insightface.app import FaceAnalysis
        from src.config import INSIGHTFACE_MODEL_NAME, CTX_ID, DETECTION_SIZE

        app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME)
        app.prepare(ctx_id=CTX_ID, det_size=DETECTION_SIZE)
        print("  OK - FaceAnalysis initialized successfully.")
        print(f"  Loaded modules: {list(app.models.keys())}")
        return True
    except Exception as e:
        print(f"  FAILED - {e}")
        return False


if __name__ == "__main__":
    ok_imports = check_imports()
    if not ok_imports:
        sys.exit(1)

    ok_model = check_model_load()
    if not ok_model:
        sys.exit(1)

    print("\nSetup verified. You're ready for Step 2 (Face Detection + Alignment).")
