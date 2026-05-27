from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.common.transforms import build_signature_transform
from src.common.utils import resolve_device
from src.ms_signet.model import MSSigNet


def load_model(
    checkpoint_path: str | Path,
    embedding_dim: int = 1024,
    device: str = "auto",
) -> tuple[MSSigNet, torch.device]:
    target_device = resolve_device(device)
    model = MSSigNet(embedding_dim=embedding_dim).to(target_device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, target_device


def embed_image(
    model: MSSigNet,
    image_path: str | Path,
    device: torch.device,
    input_size: tuple[int, int] = (150, 220),
) -> torch.Tensor:
    transform = build_signature_transform(input_size=input_size, augment=False)
    with Image.open(image_path) as image:
        tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor)["embedding"]
    return F.normalize(embedding.squeeze(0).cpu(), p=2, dim=0)


def pair_score(reference: torch.Tensor, query: torch.Tensor) -> float:
    return float(F.cosine_similarity(reference[None, :], query[None, :]).item())
