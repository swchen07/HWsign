from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.common.signature_dataset import SignatureImageDataset, read_signature_manifests


def test_read_signature_manifests_and_load_image(tmp_path) -> None:
    image_path = tmp_path / "processed" / "sample.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (10, 8), "white").save(image_path)

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "image_id": "sample_001",
                "image_path": "processed/sample.jpg",
                "label": "22571030",
                "authenticity": "TRUE",
            }
        ],
    )

    frame = read_signature_manifests(manifest_path)
    assert frame.loc[0, "authenticity"] == "true"
    assert frame.loc[0, "label"] == "22571030"

    dataset = SignatureImageDataset(manifest_path, project_root=tmp_path)
    sample = dataset[0]
    assert sample["image"].shape == (1, 150, 220)
    assert sample["image_id"] == "sample_001"
    assert sample["authenticity"] == "true"


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "image_path", "label", "authenticity"],
        )
        writer.writeheader()
        writer.writerows(rows)
