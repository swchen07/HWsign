from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

MANIFEST_COLUMNS = ["image_id", "image_path", "label", "authenticity"]
SIDE_CONFIG = {
    "true": ("true/processed_data_true", "t"),
    "fake": ("fake/processed_data_fake", "f"),
}


@dataclass(frozen=True)
class PrepareSummary:
    authenticity: str
    source_dir: Path
    output_dir: Path
    manifest_path: Path
    rows: int
    copied: int
    skipped: int


def prepare_annotation_v2(
    source_root: str | Path = "datasets/raw/annotation_v2",
    dataset_id: str = "csw_0116",
    output_root: str | Path = "datasets/processed",
    manifest_dir: str | Path = "datasets/manifests",
    overwrite: bool = False,
    project_root: str | Path = ".",
) -> dict[str, PrepareSummary]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    manifest_dir = Path(manifest_dir)
    project_root = Path(project_root).resolve()

    summaries: dict[str, PrepareSummary] = {}
    for authenticity, (relative_source, id_prefix) in SIDE_CONFIG.items():
        summaries[authenticity] = _prepare_side(
            source_dir=source_root / relative_source,
            output_dir=output_root / dataset_id / authenticity,
            manifest_path=manifest_dir / f"{dataset_id}_{authenticity}.csv",
            dataset_id=dataset_id,
            authenticity=authenticity,
            id_prefix=id_prefix,
            overwrite=overwrite,
            project_root=project_root,
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare annotation_v2 as a processed dataset.")
    parser.add_argument("--source-root", default="datasets/raw/annotation_v2")
    parser.add_argument("--dataset-id", default="csw_0116")
    parser.add_argument("--output-root", default="datasets/processed")
    parser.add_argument("--manifest-dir", default="datasets/manifests")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = prepare_annotation_v2(
        source_root=args.source_root,
        dataset_id=args.dataset_id,
        output_root=args.output_root,
        manifest_dir=args.manifest_dir,
        overwrite=args.overwrite,
    )
    for summary in summaries.values():
        print(
            {
                "authenticity": summary.authenticity,
                "rows": summary.rows,
                "copied": summary.copied,
                "skipped": summary.skipped,
                "output_dir": summary.output_dir.as_posix(),
                "manifest_path": summary.manifest_path.as_posix(),
            }
        )


def _prepare_side(
    source_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    dataset_id: str,
    authenticity: str,
    id_prefix: str,
    overwrite: bool,
    project_root: Path,
) -> PrepareSummary:
    summary_path = source_dir / "patch_mapping_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    rows = _read_summary_rows(summary_path)
    missing = [
        row["patch_filename"]
        for row in rows
        if not (source_dir / row["patch_filename"]).is_file()
    ]
    if missing:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} images listed in {summary_path} are missing: {sample}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    copied = 0
    skipped = 0
    compact_dataset_id = _compact_dataset_id(dataset_id)

    for index, row in enumerate(rows, start=1):
        source_path = source_dir / row["patch_filename"]
        target_path = output_dir / row["patch_filename"]

        if target_path.exists() and not overwrite:
            skipped += 1
        else:
            shutil.copy2(source_path, target_path)
            copied += 1

        manifest_rows.append(
            {
                "image_id": f"{compact_dataset_id}_{id_prefix}{index:06d}",
                "image_path": _relative_to_project(target_path, project_root),
                "label": str(row["original_label"]),
                "authenticity": authenticity,
            }
        )

    _write_manifest(manifest_path, manifest_rows)
    return PrepareSummary(
        authenticity=authenticity,
        source_dir=source_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        rows=len(manifest_rows),
        copied=copied,
        skipped=skipped,
    )


def _read_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"patch_filename", "original_label"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{summary_path} missing required columns: {sorted(missing_columns)}")
        return sorted(reader, key=lambda row: row["patch_filename"])


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _compact_dataset_id(dataset_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]+", "", dataset_id).lower()
    if not compact:
        raise ValueError("dataset_id must contain at least one alphanumeric character")
    return compact


def _relative_to_project(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
