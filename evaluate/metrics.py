from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def cosine_scores(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = _as_2d(reference)
    query = _as_2d(query)
    numerator = np.sum(reference * query, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(query, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def negative_euclidean_scores(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = _as_2d(reference)
    query = _as_2d(query)
    return -np.linalg.norm(reference - query, axis=1)


def eer_from_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def verification_report(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("labels must be binary: 1 for genuine, 0 for forgery")

    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    eer, eer_threshold = (
        eer_from_scores(labels, scores)
        if len(np.unique(labels)) == 2
        else (float("nan"), float("nan"))
    )
    threshold = eer_threshold if threshold is None else threshold

    predictions = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    fp = np.logical_and(predictions, negatives).sum()
    tn = np.logical_and(~predictions, negatives).sum()
    fn = np.logical_and(~predictions, positives).sum()
    tp = np.logical_and(predictions, positives).sum()
    far = float(fp / max(1, fp + tn))
    frr = float(fn / max(1, fn + tp))

    return {
        "auc": auc,
        "eer": eer,
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "accuracy": float((tp + tn) / max(1, len(labels))),
    }


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    return array
