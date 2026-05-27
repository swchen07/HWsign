from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.common.prepare_annotation_v2 import prepare_annotation_v2


def test_prepare_annotation_v2_builds_processed_dataset_and_manifests(tmp_path) -> None:
    source_root = tmp_path / "raw" / "annotation_v2"
    _write_side_fixture(
        source_root / "true" / "processed_data_true",
        [
            ("22571030_bbbb.jpg", "22571030"),
            ("22571030_aaaa.jpg", "22571030"),
        ],
    )
    _write_side_fixture(
        source_root / "fake" / "processed_data_fake",
        [
            ("22571031_dddd.jpg", "22571031"),
            ("22571031_cccc.jpg", "22571031"),
        ],
    )

    summaries = prepare_annotation_v2(
        source_root=source_root,
        dataset_id="csw_0116",
        output_root=tmp_path / "processed",
        manifest_dir=tmp_path / "manifests",
        overwrite=False,
        project_root=tmp_path,
    )

    assert summaries["true"].rows == 2
    assert summaries["fake"].rows == 2
    assert (tmp_path / "processed" / "csw_0116" / "true" / "22571030_aaaa.jpg").exists()
    assert (tmp_path / "processed" / "csw_0116" / "fake" / "22571031_cccc.jpg").exists()

    true_rows = _read_manifest(tmp_path / "manifests" / "csw_0116_true.csv")
    fake_rows = _read_manifest(tmp_path / "manifests" / "csw_0116_fake.csv")

    assert true_rows[0] == {
        "image_id": "csw0116_t000001",
        "image_path": "processed/csw_0116/true/22571030_aaaa.jpg",
        "label": "22571030",
        "authenticity": "true",
    }
    assert true_rows[1]["image_id"] == "csw0116_t000002"
    assert fake_rows[0]["image_id"] == "csw0116_f000001"
    assert fake_rows[0]["label"] == "22571031"
    assert fake_rows[0]["authenticity"] == "fake"


def _write_side_fixture(side_dir: Path, rows: list[tuple[str, str]]) -> None:
    side_dir.mkdir(parents=True)
    with (side_dir / "patch_mapping_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patch_id",
                "patch_filename",
                "original_label",
                "source_image",
                "source_csv",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
            ],
        )
        writer.writeheader()
        for index, (filename, label) in enumerate(rows, start=1):
            Image.new("RGB", (8, 8), "white").save(side_dir / filename)
            writer.writerow(
                {
                    "patch_id": f"uuid-{index}",
                    "patch_filename": filename,
                    "original_label": label,
                    "source_image": "source.jpg",
                    "source_csv": "source.csv",
                    "bbox_x": 0,
                    "bbox_y": 0,
                    "bbox_w": 8,
                    "bbox_h": 8,
                }
            )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
