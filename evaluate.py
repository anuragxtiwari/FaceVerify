"""
Step 6: Dataset + Evaluation System.

Expects a dataset laid out as:
    data/dataset/<person_name>/<image>.jpg   (2+ images per person, 2+ people)

Builds positive pairs (same person) and negative pairs (different people),
computes similarity for every pair, then:
    - plots an ROC curve
    - computes EER (Equal Error Rate)
    - reports the threshold at the EER point

Usage:
    python evaluate.py
    python evaluate.py --dataset_dir data/dataset --max_negatives_per_person 20
"""

import os
import json
import argparse
import itertools
import random

import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from src.detector import FaceDetector, NoFaceDetectedError
from src.embedder import FaceEmbedder
from src.similarity import cosine_similarity
from src.config import DATA_DIR


def collect_dataset(dataset_dir):
    """Returns {person_name: [image_path, ...]}"""
    people = {}
    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        images = [
            os.path.join(person_dir, f)
            for f in sorted(os.listdir(person_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(images) >= 1:
            people[person_name] = images
    return people


def build_pairs(people, max_negatives_per_person, seed=42):
    """
    Returns (pairs, labels):
        pairs  = list of (path1, path2)
        labels = list of 1 (same person) / 0 (different person), same order
    """
    rng = random.Random(seed)
    pairs = []
    labels = []

    # Positive pairs: every 2-combination within each person's images
    for person, images in people.items():
        if len(images) < 2:
            continue
        for img1, img2 in itertools.combinations(images, 2):
            pairs.append((img1, img2))
            labels.append(1)

    # Negative pairs: sample across different people (capped, so this doesn't
    # explode combinatorially with a larger dataset)
    person_names = list(people.keys())
    for i, person_a in enumerate(person_names):
        others = person_names[i + 1:]
        for person_b in others:
            imgs_a = people[person_a]
            imgs_b = people[person_b]
            possible = list(itertools.product(imgs_a, imgs_b))
            rng.shuffle(possible)
            for img1, img2 in possible[:max_negatives_per_person]:
                pairs.append((img1, img2))
                labels.append(0)

    return pairs, labels


def compute_embeddings_cache(image_paths, detector, embedder):
    """Compute each unique image's embedding ONCE, reused across all pairs."""
    cache = {}
    failed = set()
    unique_paths = sorted(set(image_paths))
    for i, path in enumerate(unique_paths):
        print(f"  [{i+1}/{len(unique_paths)}] Embedding {path}...")
        image = cv2.imread(path)
        if image is None:
            print(f"    WARNING: could not read {path}, skipping.")
            failed.add(path)
            continue
        try:
            aligned = detector.detect_and_align_largest(image)
            embedding = embedder.get_embedding(aligned)
            cache[path] = embedding
        except NoFaceDetectedError:
            print(f"    WARNING: no face detected in {path}, skipping.")
            failed.add(path)
    return cache, failed


def compute_eer(fpr, tpr, thresholds):
    """
    EER = point where False Positive Rate == False Negative Rate (1 - TPR).
    Returns (eer, threshold_at_eer).
    """
    fnr = 1 - tpr
    # Find the index where fpr and fnr are closest (crossover point)
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return float(eer), float(eer_threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=os.path.join(DATA_DIR, "dataset"))
    parser.add_argument("--max_negatives_per_person", type=int, default=20,
                         help="Cap on negative pairs sampled per person-pair, to avoid combinatorial blowup.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"ERROR: dataset dir not found: {args.dataset_dir}")
        print("Expected structure: data/dataset/<person_name>/<image>.jpg")
        return

    print(f"Scanning dataset at {args.dataset_dir}...")
    people = collect_dataset(args.dataset_dir)
    print(f"Found {len(people)} people: {list(people.keys())}")
    for name, imgs in people.items():
        print(f"  {name}: {len(imgs)} images")

    if len(people) < 2:
        print("\nERROR: need at least 2 people in the dataset to build negative pairs.")
        return

    pairs, labels = build_pairs(people, args.max_negatives_per_person)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"\nBuilt {len(pairs)} pairs -> {n_pos} positive, {n_neg} negative")

    if n_pos == 0:
        print("ERROR: 0 positive pairs — need at least one person with 2+ images.")
        return

    print("\nLoading models...")
    detector = FaceDetector()
    embedder = FaceEmbedder()

    all_image_paths = [p for pair in pairs for p in pair]
    print(f"\nComputing embeddings for {len(set(all_image_paths))} unique images...")
    embedding_cache, failed = compute_embeddings_cache(all_image_paths, detector, embedder)

    # Drop any pair that touches a failed image
    scores = []
    clean_labels = []
    for (img1, img2), label in zip(pairs, labels):
        if img1 in failed or img2 in failed:
            continue
        sim = cosine_similarity(embedding_cache[img1], embedding_cache[img2])
        scores.append(sim)
        clean_labels.append(label)

    scores = np.array(scores)
    clean_labels = np.array(clean_labels)
    print(f"\nUsable pairs after dropping failures: {len(scores)} "
          f"({sum(clean_labels)} positive, {len(clean_labels) - sum(clean_labels)} negative)")

    # --- ROC + EER ---
    fpr, tpr, thresholds = roc_curve(clean_labels, scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)

    print("\n=== EVALUATION RESULTS ===")
    print(f"AUC:              {roc_auc:.4f}")
    print(f"EER:              {eer:.4f}  ({eer*100:.2f}%)")
    print(f"Threshold at EER: {eer_threshold:.4f}")

    threshold_path = os.path.join(DATA_DIR, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({
            "threshold": eer_threshold,
            "eer": eer,
            "auc": roc_auc,
            "n_pairs": len(scores),
            "n_positive": int(sum(clean_labels)),
            "n_negative": int(len(clean_labels) - sum(clean_labels)),
        }, f, indent=2)
    print(f"\nSaved computed threshold to: {threshold_path}")
    print("main.py / similarity.py will now use this threshold automatically.")

    # --- Plot ROC curve ---
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    plt.scatter(fpr[eer_idx], tpr[eer_idx], color="red", zorder=5,
                label=f"EER point ({eer*100:.2f}%)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Face Verification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    out_path = os.path.join(DATA_DIR, "roc_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nROC curve saved to: {out_path}")


if __name__ == "__main__":
    main()