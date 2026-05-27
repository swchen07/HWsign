from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate.metrics import verification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a verification threshold from scores.")
    parser.add_argument("--scores", required=True, help="CSV with score and label columns.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scores)
    report = verification_report(frame["label"].to_numpy(), frame["score"].to_numpy())
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
