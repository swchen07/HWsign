from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate.metrics import verification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pair-level signature verification scores."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest. Can be image or score manifest.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)

    if {"score", "label"}.issubset(frame.columns):
        report = verification_report(
            labels=frame["label"].to_numpy(),
            scores=frame["score"].to_numpy(),
            threshold=args.threshold,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    path = Path(args.manifest)
    summary = {
        "manifest": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "message": "No score column found; image manifest loaded successfully.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
