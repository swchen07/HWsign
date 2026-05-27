from __future__ import annotations

import torch

from src.common.utils import resolve_device, seed_everything
from src.ms_signet.loss import CoTupletLoss
from src.ms_signet.model import MSSigNet


def main() -> None:
    seed_everything(11)
    device = resolve_device("auto")
    model = MSSigNet(embedding_dim=128).to(device)
    model.eval()

    batch_size = 2
    count = 5
    anchor = torch.randn(batch_size, 1, 150, 220, device=device)
    positives = torch.randn(batch_size * count, 1, 150, 220, device=device)
    negatives = torch.randn(batch_size * count, 1, 150, 220, device=device)

    with torch.inference_mode():
        anchor_embedding = model(anchor)["embedding"]
        positive_embedding = model(positives)["embedding"].view(batch_size, count, -1)
        negative_embedding = model(negatives)["embedding"].view(batch_size, count, -1)

    loss = CoTupletLoss()(anchor_embedding, positive_embedding, negative_embedding)
    print(
        {
            "device": str(device),
            "embedding_shape": tuple(anchor_embedding.shape),
            "loss": float(loss.item()),
        }
    )


if __name__ == "__main__":
    main()
