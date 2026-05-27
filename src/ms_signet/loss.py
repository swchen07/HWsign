from __future__ import annotations

import torch
from torch import nn


class CoTupletLoss(nn.Module):
    """Co-Tuplet style loss for one anchor, K positives, and K negatives."""

    def __init__(self, epsilon: float = 0.2) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        if positives.ndim != 3 or negatives.ndim != 3:
            raise ValueError("positives and negatives must have shape [batch, k, dim]")
        if anchor.ndim != 2:
            raise ValueError("anchor must have shape [batch, dim]")

        pos_dist = (anchor[:, None, :] - positives).pow(2).sum(dim=-1)
        neg_dist = (anchor[:, None, :] - negatives).pow(2).sum(dim=-1)
        hard_pos = pos_dist.max(dim=1, keepdim=True).values
        hard_neg = neg_dist.min(dim=1, keepdim=True).values

        pos_loss = torch.relu(pos_dist - hard_neg + self.epsilon)
        neg_loss = torch.relu(hard_pos - neg_dist + self.epsilon)
        return 0.5 * (pos_loss.mean() + neg_loss.mean())


class MultiBranchCoTupletLoss(nn.Module):
    """Apply Co-Tuplet loss to the concatenated embedding and all MS-SigNet branches."""

    def __init__(self, epsilon: float = 0.2, branch_weight: float = 1.0) -> None:
        super().__init__()
        self.base_loss = CoTupletLoss(epsilon=epsilon)
        self.branch_weight = branch_weight

    def forward(
        self,
        anchor_outputs: dict[str, torch.Tensor | list[torch.Tensor]],
        positive_outputs: dict[str, torch.Tensor | list[torch.Tensor]],
        negative_outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> torch.Tensor:
        loss = self.base_loss(
            anchor_outputs["embedding"],
            positive_outputs["embedding"],
            negative_outputs["embedding"],
        )

        branch_losses = [
            self.base_loss(
                anchor_outputs["global"],
                positive_outputs["global"],
                negative_outputs["global"],
            )
        ]

        anchor_regions = anchor_outputs["regions"]
        positive_regions = positive_outputs["regions"]
        negative_regions = negative_outputs["regions"]
        if not isinstance(anchor_regions, list):
            raise TypeError("Expected regions to be a list of tensors")

        for anchor_region, positive_region, negative_region in zip(
            anchor_regions,
            positive_regions,
            negative_regions,
            strict=True,
        ):
            branch_losses.append(self.base_loss(anchor_region, positive_region, negative_region))

        return loss + self.branch_weight * torch.stack(branch_losses).mean()
