from __future__ import annotations

import torch

from src.ms_signet.loss import CoTupletLoss
from src.ms_signet.model import MSSigNet


def test_ms_signet_forward_shape() -> None:
    model = MSSigNet(embedding_dim=32)
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 1, 150, 220))
    assert outputs["embedding"].shape == (1, 32 * 7)
    assert outputs["global"].shape == (1, 32)
    assert len(outputs["regions"]) == 6


def test_co_tuplet_loss_is_finite() -> None:
    loss_fn = CoTupletLoss(epsilon=0.2)
    anchor = torch.randn(2, 16)
    positives = torch.randn(2, 5, 16)
    negatives = torch.randn(2, 5, 16)
    loss = loss_fn(anchor, positives, negatives)
    assert torch.isfinite(loss)
