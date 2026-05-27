from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.ms_signet.sampler import MSSigNetCoTupletDataset


def test_ms_signet_cotuplet_dataset_samples_by_label_and_authenticity(tmp_path) -> None:
    true_manifest = tmp_path / "true.csv"
    fake_manifest = tmp_path / "fake.csv"
    image_root = tmp_path / "processed" / "csw_0116"

    true_rows = []
    for index in range(3):
        path = image_root / "true" / f"true_{index}.jpg"
        _write_image(path)
        true_rows.append(
            {
                "image_id": f"t{index}",
                "image_path": path.relative_to(tmp_path).as_posix(),
                "label": "22571030",
                "authenticity": "true",
            }
        )

    fake_rows = []
    for index in range(2):
        path = image_root / "fake" / f"fake_{index}.jpg"
        _write_image(path)
        fake_rows.append(
            {
                "image_id": f"f{index}",
                "image_path": path.relative_to(tmp_path).as_posix(),
                "label": "22571030",
                "authenticity": "fake",
            }
        )

    _write_manifest(true_manifest, true_rows)
    _write_manifest(fake_manifest, fake_rows)

    dataset = MSSigNetCoTupletDataset(
        manifest_paths=[true_manifest, fake_manifest],
        project_root=tmp_path,
        num_positive=2,
        num_negative=2,
        augment=False,
    )

    sample = dataset[0]
    assert sample["anchor"].shape == (1, 150, 220)
    assert sample["positives"].shape == (2, 1, 150, 220)
    assert sample["negatives"].shape == (2, 1, 150, 220)
    assert sample["label"] == "22571030"


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), "white").save(path)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "image_path", "label", "authenticity"],
        )
        writer.writeheader()
        writer.writerows(rows)
