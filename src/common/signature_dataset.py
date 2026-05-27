from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.common.transforms import build_signature_transform

STANDARD_MANIFEST_COLUMNS = ["image_id", "image_path", "label", "authenticity"]


def read_signature_manifests(manifest_paths: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    paths = _as_path_list(manifest_paths)
    frames: list[pd.DataFrame] = []
    for manifest_path in paths:
        frame = pd.read_csv(manifest_path, dtype=str)
        _validate_manifest_columns(frame, manifest_path)
        frame = frame.copy()
        frame["label"] = frame["label"].astype(str)
        frame["authenticity"] = frame["authenticity"].astype(str).str.lower()
        frame["source_manifest"] = Path(manifest_path).as_posix()
        frames.append(frame)

    if not frames:
        raise ValueError("At least one manifest path is required")

    merged = pd.concat(frames, ignore_index=True)
    if merged["image_id"].duplicated().any():
        duplicates = merged.loc[merged["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"Duplicate image_id values found: {duplicates}")
    return merged


class SignatureImageDataset(Dataset):
    """Generic single-image dataset backed by standard signature manifests."""

    def __init__(
        self,
        manifest_paths: str | Path | Sequence[str | Path],
        project_root: str | Path = ".",
        input_size: tuple[int, int] = (150, 220),
        augment: bool = False,
    ) -> None:
        self.frame = read_signature_manifests(manifest_paths)
        self.project_root = Path(project_root)
        self.transform = build_signature_transform(input_size=input_size, augment=augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = resolve_image_path(row["image_path"], self.project_root)
        image = load_signature_image(image_path, self.transform)
        return {
            "image": image,
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "label": row["label"],
            "authenticity": row["authenticity"],
        }


def load_signature_image(image_path: str | Path, transform: Any) -> torch.Tensor:
    with Image.open(image_path) as image:
        return transform(image)


def resolve_image_path(image_path: str | Path, project_root: str | Path = ".") -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _validate_manifest_columns(frame: pd.DataFrame, manifest_path: Path) -> None:
    missing = set(STANDARD_MANIFEST_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")


def _as_path_list(manifest_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(manifest_paths, str | Path):
        return [Path(manifest_paths)]
    return [Path(path) for path in manifest_paths]
