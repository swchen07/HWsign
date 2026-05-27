from __future__ import annotations

import numpy as np

from evaluate.metrics import cosine_scores, verification_report


def test_cosine_scores() -> None:
    reference = np.array([[1.0, 0.0], [1.0, 0.0]])
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = cosine_scores(reference, query)
    assert np.allclose(scores, [1.0, 0.0])


def test_verification_report() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    report = verification_report(labels, scores)
    assert report["auc"] == 1.0
    assert report["far"] == 0.0
    assert report["frr"] == 0.0
