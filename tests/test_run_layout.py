from __future__ import annotations

from src.common.run import create_run_layout


def test_create_run_layout(tmp_path) -> None:
    config = {
        "run": {
            "name": "MS SigNet HanSig",
            "output_dir": tmp_path.as_posix(),
            "timestamp": "fixed",
        }
    }

    paths = create_run_layout(config)

    assert paths["run_dir"] == tmp_path / "ms_signet_hansig" / "fixed"
    assert (paths["run_dir"] / "config.yaml").exists()
    assert (paths["run_dir"] / "run.json").exists()
    assert paths["checkpoints"].is_dir()
    assert paths["metrics"].is_dir()
    assert paths["predictions"].is_dir()
    assert paths["logs"].is_dir()
    assert paths["embeddings"].is_dir()
