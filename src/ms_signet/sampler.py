from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.common.transforms import build_signature_transform


class HanSigCoTupletDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        num_positive: int = 5,
        num_negative: int = 5,
        input_size: tuple[int, int] = (150, 220),
        augment: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root)
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.transform = build_signature_transform(input_size=input_size, augment=augment)

        self.frame = pd.read_csv(self.manifest_path)
        required = {"image_path", "label", "signer_id"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        self.genuine = self.frame[self.frame["label"] == "genuine"].reset_index(drop=True)
        self.forgery = self.frame[self.frame["label"] == "forgery"].reset_index(drop=True)
        if self.genuine.empty or self.forgery.empty:
            raise ValueError("Co-tuplet training needs both genuine and forgery samples")

        self.genuine_by_signer = self._group_indices(self.genuine)
        self.forgery_by_signer = self._group_indices(self.forgery)
        self.anchor_indices = [
            idx
            for idx, row in self.genuine.iterrows()
            if row["signer_id"] in self.forgery_by_signer
            and len(self.genuine_by_signer[row["signer_id"]]) >= 2
        ]
        if not self.anchor_indices:
            raise ValueError("No signer has enough genuine samples and matching forgeries")

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_idx = self.anchor_indices[index]
        anchor_row = self.genuine.iloc[anchor_idx]
        signer_id = anchor_row["signer_id"]

        positive_pool = [idx for idx in self.genuine_by_signer[signer_id] if idx != anchor_idx]
        negative_pool = self.forgery_by_signer[signer_id]

        positive_rows = self.genuine.iloc[
            self._sample_indices(positive_pool, self.num_positive)
        ]
        negative_rows = self.forgery.iloc[
            self._sample_indices(negative_pool, self.num_negative)
        ]

        anchor = self._load_image(anchor_row["image_path"])
        positives = torch.stack([self._load_image(path) for path in positive_rows["image_path"]])
        negatives = torch.stack([self._load_image(path) for path in negative_rows["image_path"]])

        return {
            "anchor": anchor,
            "positives": positives,
            "negatives": negatives,
            "signer_id": signer_id,
        }

    @staticmethod
    def _group_indices(frame: pd.DataFrame) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, row in frame.iterrows():
            grouped[str(row["signer_id"])].append(idx)
        return grouped

    @staticmethod
    def _sample_indices(pool: list[int], count: int) -> list[int]:
        if len(pool) >= count:
            return random.sample(pool, count)
        return [random.choice(pool) for _ in range(count)]

    def _load_image(self, relative_path: str) -> torch.Tensor:
        image_path = self.root / relative_path
        with Image.open(image_path) as image:
            return self.transform(image)


def flatten_tuplet_batch(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError("Expected tensor shape [batch, k, channels, height, width]")
    batch, count, channels, height, width = images.shape
    return images.view(batch * count, channels, height, width)


def unflatten_embeddings(embeddings: torch.Tensor, batch_size: int, count: int) -> torch.Tensor:
    return embeddings.view(batch_size, count, -1)
