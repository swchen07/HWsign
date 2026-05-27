from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common.run import append_log_line, create_run_layout, write_json
from src.common.utils import ensure_dir, load_yaml, resolve_device, seed_everything
from src.ms_signet.loss import MultiBranchCoTupletLoss
from src.ms_signet.model import MSSigNet
from src.ms_signet.sampler import (
    HanSigCoTupletDataset,
    MSSigNetCoTupletDataset,
    flatten_tuplet_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MS-SigNet on signature manifests.")
    parser.add_argument("--config", default="configs/ms_signet_hansig.yaml")
    return parser.parse_args()


def forward_tuplet(
    model: MSSigNet,
    anchor: torch.Tensor,
    grouped: torch.Tensor,
) -> dict[str, object]:
    batch_size, count = grouped.shape[:2]
    anchor_outputs = model(anchor)
    flat_outputs = model(flatten_tuplet_batch(grouped))
    flat_outputs["embedding"] = flat_outputs["embedding"].view(batch_size, count, -1)
    flat_outputs["global"] = flat_outputs["global"].view(batch_size, count, -1)
    flat_outputs["regions"] = [
        region.view(batch_size, count, -1) for region in flat_outputs["regions"]
    ]
    return {"anchor": anchor_outputs, "grouped": flat_outputs}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 11)))

    # initialize device, dataset, model, criterion, optimizer, and checkpoint paths
    device = resolve_device(str(config.get("device", "auto")))
    data_config = config["data"]
    train_config = config["train"]
    model_config = config["model"]

    # build dataset and dataloader
    dataset = build_dataset(data_config)
    loader = DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 18)),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    # build model, loss criterion, optimizer, and scaler
    model = MSSigNet(embedding_dim=int(model_config.get("embedding_dim", 1024))).to(device)
    criterion = MultiBranchCoTupletLoss(
        epsilon=float(train_config.get("epsilon", 0.2)),
        branch_weight=float(train_config.get("branch_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 0.0001)),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train_config.get("amp", False)) and device.type == "cuda"
    )

    # set up checkpoint paths and training history
    if train_config.get("checkpoint_dir"):
        checkpoint_dir = ensure_dir(train_config["checkpoint_dir"])
        run_paths = {
            "run_dir": checkpoint_dir.parent,
            "checkpoints": checkpoint_dir,
            "metrics": ensure_dir(checkpoint_dir.parent / "metrics"),
            "logs": ensure_dir(checkpoint_dir.parent / "logs"),
        }
    else:
        run_paths = create_run_layout(config, config_path=args.config)
        checkpoint_dir = run_paths["checkpoints"]

    save_best = bool(config.get("run", {}).get("save_best", True))
    train_log = run_paths["logs"] / "train.log"
    history: list[dict[str, float | int | str]] = []
    best_loss = math.inf
    print(f"run_dir={run_paths['run_dir']}")

    # training loop
    for epoch in range(1, int(train_config.get("epochs", 90)) + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            anchor = batch["anchor"].to(device, non_blocking=True)
            positives = batch["positives"].to(device, non_blocking=True)
            negatives = batch["negatives"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pos_outputs = forward_tuplet(model, anchor, positives)
                neg_outputs = forward_tuplet(model, anchor, negatives)
                loss = criterion(
                    pos_outputs["anchor"],
                    pos_outputs["grouped"],
                    neg_outputs["grouped"],
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            if step % int(train_config.get("log_interval", 20)) == 0:
                progress.set_postfix(loss=running_loss / step)

        mean_loss = running_loss / max(1, len(loader))
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "checkpoint": f"epoch_{epoch:03d}.pth",
        }
        history.append(epoch_metrics)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "metrics": epoch_metrics,
            "run_dir": str(run_paths["run_dir"]),
        }
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        torch.save(payload, checkpoint_path)
        torch.save(payload, checkpoint_dir / "latest.pth")

        if save_best and mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(payload, checkpoint_dir / "best.pth")
            epoch_metrics["best_checkpoint"] = "best.pth"

        write_json(run_paths["metrics"] / "latest_train.json", epoch_metrics)
        write_json(run_paths["metrics"] / "train_history.json", history)
        append_log_line(train_log, f"epoch={epoch} loss={mean_loss:.6f} saved={checkpoint_path}")
        print(f"epoch={epoch} loss={mean_loss:.6f} saved={checkpoint_path}")


def build_dataset(data_config: dict[str, object]) -> Dataset:
    common_kwargs = {
        "num_positive": int(data_config.get("num_positive", 5)),
        "num_negative": int(data_config.get("num_negative", 5)),
        "input_size": tuple(data_config.get("input_size", [150, 220])),
        "augment": True,
    }
    if data_config.get("manifests"):
        return MSSigNetCoTupletDataset(
            manifest_paths=data_config["manifests"],
            project_root=str(data_config.get("project_root", ".")),
            **common_kwargs,
        )
    return HanSigCoTupletDataset(
        manifest_path=data_config["train_manifest"],
        root=data_config["root"],
        **common_kwargs,
    )


if __name__ == "__main__":
    main()
