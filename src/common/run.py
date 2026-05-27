from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.common.utils import ensure_dir


def create_run_layout(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    run_config = config.get("run", {})
    run_name = _slugify(str(run_config.get("name", "experiment")))
    timestamp = str(run_config.get("timestamp", "auto"))
    if timestamp == "auto":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    output_dir = Path(run_config.get("output_dir", "outputs/runs"))
    run_dir = ensure_dir(output_dir / run_name / timestamp)

    paths = {
        "run_dir": run_dir,
        "checkpoints": ensure_dir(run_dir / "checkpoints"),
        "metrics": ensure_dir(run_dir / "metrics"),
        "predictions": ensure_dir(run_dir / "predictions"),
        "logs": ensure_dir(run_dir / "logs"),
        "embeddings": ensure_dir(run_dir / "embeddings"),
    }

    snapshot = dict(config)
    snapshot["resolved_run_dir"] = run_dir.as_posix()
    if config_path is not None:
        snapshot["source_config"] = str(config_path)

    write_yaml(run_dir / "config.yaml", snapshot)
    write_json(run_dir / "run.json", {"run_dir": run_dir.as_posix(), "name": run_name})
    return paths


def write_json(path: str | Path, payload: Any) -> None:
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(content, encoding="utf-8")


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def append_log_line(path: str | Path, line: str) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "experiment"
