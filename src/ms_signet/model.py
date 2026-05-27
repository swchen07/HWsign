from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MSSigNet(nn.Module):
    """Multi-scale signature network for 150x220 grayscale signatures."""

    def __init__(self, embedding_dim: int = 1024) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
        )
        self.conv2f = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.conv3f = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.conv51 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv52 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.ca_reduce = nn.Sequential(nn.Conv2d(256, 32, kernel_size=1), nn.ReLU(inplace=True))
        self.ca_expand = nn.Sequential(nn.Conv2d(32, 256, kernel_size=1), nn.Sigmoid())
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.shape[-2:] != (150, 220):
            raise ValueError(f"MSSigNet expects input size 150x220, got {tuple(x.shape[-2:])}")

        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        global_map = self.conv51(out4)
        regional_map = self.conv52(out4)
        fused2 = self.conv2f(out2)
        fused3 = self.conv3f(out3)

        global_map = fused2 * fused3 * global_map
        regional_map = fused2 * fused3 * regional_map

        global_attention = self.ca_reduce(self.gap(global_map))
        regional_attention = self.ca_reduce(self.gap(regional_map))
        attention = self.ca_expand(global_attention * regional_attention)

        global_map = attention * global_map
        regional_map = attention * regional_map

        global_embedding = self._project(global_map)
        region_embeddings = self._regional_embeddings(regional_map)
        embedding = torch.cat([global_embedding, *region_embeddings], dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)

        return {
            "embedding": embedding,
            "global": global_embedding,
            "regions": region_embeddings,
        }

    def _project(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = self.gap(feature_map).flatten(1)
        pooled = F.normalize(pooled, p=2, dim=1)
        return self.fc(pooled)

    def _regional_embeddings(self, feature_map: torch.Tensor) -> list[torch.Tensor]:
        horizontal = torch.nn.functional.unfold(feature_map, kernel_size=(16, 13), stride=6)
        horizontal = horizontal.view(feature_map.size(0), 256, 16, 13, 3).permute(0, 4, 1, 2, 3)

        vertical = torch.nn.functional.unfold(feature_map, kernel_size=(8, 25), stride=4)
        vertical = vertical.view(feature_map.size(0), 256, 8, 25, 3).permute(0, 4, 1, 2, 3)

        regions = [horizontal[:, idx] for idx in range(3)]
        regions.extend(vertical[:, idx] for idx in range(3))
        return [self._project(region) for region in regions]
