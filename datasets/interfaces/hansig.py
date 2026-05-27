from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from datasets.interfaces.base import SignatureRecord, ensure_relative_path

HANSIG_FILENAME_RE = re.compile(
    r"^(?P<label>original|forgery)_w(?P<writer>\d+)_(?P<name>\d+)_(?P<sample>\d+)"
    r"\.(?P<ext>jpg|jpeg|png|bmp)$",
    re.IGNORECASE,
)


def parse_hansig_filename(path: str | Path, root: str | Path | None = None) -> SignatureRecord:
    file_path = Path(path)
    match = HANSIG_FILENAME_RE.match(file_path.name)
    if match is None:
        raise ValueError(f"Not a HanSig filename: {file_path.name}")

    label = "genuine" if match.group("label").lower() == "original" else "forgery"
    writer_idx = int(match.group("writer"))
    name_id = int(match.group("name"))
    sample_id = int(match.group("sample"))
    writer_id = f"w{writer_idx:03d}"
    signer_id = f"{writer_id}_n{name_id:03d}"

    if root is not None:
        image_path = ensure_relative_path(file_path, Path(root))
    else:
        image_path = file_path.as_posix()

    return SignatureRecord(
        image_path=image_path,
        label=label,
        writer_id=writer_id,
        name_id=name_id,
        signer_id=signer_id,
        sample_id=sample_id,
    )


def discover_hansig_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in root_path.rglob("*") if path.suffix.lower() in supported)


def build_hansig_manifest(root: str | Path) -> list[SignatureRecord]:
    root_path = Path(root)
    records: list[SignatureRecord] = []
    skipped: list[Path] = []

    for image_path in discover_hansig_images(root_path):
        try:
            records.append(parse_hansig_filename(image_path, root=root_path))
        except ValueError:
            skipped.append(image_path)

    if not records:
        raise ValueError(
            f"No HanSig images found under {root_path}. Expected names like "
            "original_w1_2_3.jpg and forgery_w1_2_3.jpg."
        )

    if skipped:
        skipped_names = ", ".join(path.name for path in skipped[:5])
        print(f"Skipped {len(skipped)} files with unexpected names: {skipped_names}")

    return records


def split_records(
    records: list[SignatureRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 11,
    group_by: str = "writer_id",
) -> dict[str, list[SignatureRecord]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    groups: dict[str, list[SignatureRecord]] = defaultdict(list)
    for record in records:
        key = getattr(record, group_by)
        groups[str(key)].append(record)

    group_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = max(1, int(n_groups * train_ratio))
    n_val = max(1, int(n_groups * val_ratio)) if n_groups >= 3 else 0
    n_train = min(n_train, n_groups)
    n_val = min(n_val, max(0, n_groups - n_train))

    split_keys = {
        "train": set(group_keys[:n_train]),
        "val": set(group_keys[n_train : n_train + n_val]),
        "test": set(group_keys[n_train + n_val :]),
    }

    splits: dict[str, list[SignatureRecord]] = {"train": [], "val": [], "test": []}
    for split, keys in split_keys.items():
        for key in sorted(keys):
            splits[split].extend(record.with_split(split) for record in groups[key])

    return splits
