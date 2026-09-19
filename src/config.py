"""
Central configuration for the face verification system.
Nothing in here should require code changes later — just edit values.
"""

import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# --- InsightFace model settings ---
INSIGHTFACE_MODEL_NAME = "buffalo_s"
# ctx_id = -1 forces CPU. Set to 0 if you have a working CUDA GPU + onnxruntime-gpu installed.
CTX_ID = -1
DETECTION_SIZE = (320, 320)   # detector input size, standard default

# --- Alignment ---
ALIGNED_FACE_SIZE = 112  # ArcFace expects 112x112 aligned crops

# --- Embedding ---
EMBEDDING_DIM = 512

# --- Verification ---
# Placeholder default — Step 7 will replace this with a data-driven,
# EER-optimized threshold instead of this hardcoded guess.
DEFAULT_SIMILARITY_THRESHOLD = 0.35

# --- Auto-load evaluated threshold, if evaluate.py has been run ---
# This overrides DEFAULT_SIMILARITY_THRESHOLD above with the data-driven,
# EER-optimized value, so nothing needs to be hand-edited after evaluation.
import json as _json

_threshold_file = os.path.join(DATA_DIR, "threshold.json")
if os.path.exists(_threshold_file):
    try:
        with open(_threshold_file, "r") as _f:
            _threshold_data = _json.load(_f)
        DEFAULT_SIMILARITY_THRESHOLD = _threshold_data["threshold"]
    except (_json.JSONDecodeError, KeyError):
        pass  # fall back to the hardcoded default above if the file is malformed