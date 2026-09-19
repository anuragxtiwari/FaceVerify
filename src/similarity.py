"""
Step 4: Similarity Module.

Responsibility of this module ONLY:
    input:  two embeddings (each a (512,) L2-normalized numpy array from embedder.py)
    output: a cosine similarity score, and a SAME/DIFFERENT decision vs. a threshold

Because both embeddings are already L2-normalized (unit length), cosine
similarity reduces to a plain dot product:
    cos_sim(a, b) = (a . b) / (||a|| * ||b||) = a . b   when ||a||=||b||=1
"""

import numpy as np

from src.config import DEFAULT_SIMILARITY_THRESHOLD


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Returns cosine similarity in range [-1, 1] (in practice, ArcFace embeddings
    for real face pairs typically fall somewhere in [-0.2, 0.8]).
    Higher = more similar.
    """
    embedding1 = np.asarray(embedding1).flatten()
    embedding2 = np.asarray(embedding2).flatten()

    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}"
        )

    # Dot product of two unit vectors = cosine similarity.
    # Still divide by norms explicitly in case either wasn't perfectly normalized
    # (e.g. float32 rounding) — makes this function correct even if reused
    # standalone with non-normalized embeddings.
    dot = np.dot(embedding1, embedding2)
    norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    if norm_product == 0:
        raise ValueError("One of the embeddings has zero norm.")

    similarity = float(dot / norm_product)
    return similarity


def verify_match(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = None) -> dict:
    """
    Full similarity decision for a pair of embeddings.

    threshold: if None, uses DEFAULT_SIMILARITY_THRESHOLD from config.py.
               Step 7 will replace this default with a properly computed,
               EER-optimized value — this param exists so evaluate.py can
               sweep thresholds without touching this function.

    Returns:
        {
            "similarity": float,      # raw cosine similarity
            "threshold": float,       # threshold used for this decision
            "is_same_person": bool,
        }
    """
    if threshold is None:
        threshold = DEFAULT_SIMILARITY_THRESHOLD

    similarity = cosine_similarity(embedding1, embedding2)
    is_same_person = similarity >= threshold

    return {
        "similarity": similarity,
        "threshold": threshold,
        "is_same_person": is_same_person,
    }