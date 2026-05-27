from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from datasets.interfaces.base import records_to_rows
from datasets.interfaces.hansig import build_hansig_manifest, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create HanSig image manifests.")
    parser.add_argument("--root", required=True, help="HanSig root directory.")
    parser.add_argument("--out", default="datasets/manifests", help="Output manifest directory.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    records = build_hansig_manifest(args.root)
    splits = split_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    full_records = [record.with_split(split) for split, items in splits.items() for record in items]
    write_csv(out_dir / "hansig_full.csv", records_to_rows(full_records))
    for split, items in splits.items():
        write_csv(out_dir / f"hansig_{split}.csv", records_to_rows(items))


if __name__ == "__main__":
    main()
