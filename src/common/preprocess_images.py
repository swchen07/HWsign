from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - exercised only when optional package is absent.
    register_heif_opener = None


SUPPORTED_SUFFIXES = {".heic", ".heif", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ConvertSummary:
    input_dir: Path
    output_dir: Path
    converted: int
    skipped: int
    failed: int


def convert_folder_to_jpg(
    input_dir: str | Path,
    output_root: str | Path = "datasets/raw",
    quality: int = 95,
    overwrite: bool = False,
    recursive: bool = True,
    verbose: bool = False,
) -> ConvertSummary:
    source_dir = Path(input_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {source_dir}")

    _register_heic_support()

    output_dir = Path(output_root) / f"{source_dir.name}_jpg"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    failed = 0
    image_paths = _iter_image_paths(source_dir, recursive=recursive)

    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(source_dir)
        output_path = (output_dir / relative_path).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            skipped += 1
            if verbose:
                print(f"[{index}/{total}] skipped {relative_path}")
            continue

        try:
            _convert_one_image(image_path, output_path, quality=quality)
            converted += 1
            if verbose:
                print(f"[{index}/{total}] converted {relative_path} -> {output_path.name}")
        except Exception as exc:
            failed += 1
            print(f"failed {image_path}: {exc}")

    return ConvertSummary(
        input_dir=source_dir,
        output_dir=output_dir,
        converted=converted,
        skipped=skipped,
        failed=failed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert images in one folder to JPEG.")
    parser.add_argument("input_dir", help="Folder containing images to convert.")
    parser.add_argument(
        "--output-root",
        default="datasets/raw",
        help="Root folder for processed outputs.",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality from 1 to 100.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPEG files.")
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only convert files directly inside input_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_folder_to_jpg(
        input_dir=args.input_dir,
        output_root=args.output_root,
        quality=args.quality,
        overwrite=args.overwrite,
        recursive=not args.non_recursive,
        verbose=True,
    )
    print(
        {
            "input_dir": summary.input_dir.as_posix(),
            "output_dir": summary.output_dir.as_posix(),
            "converted": summary.converted,
            "skipped": summary.skipped,
            "failed": summary.failed,
        }
    )


def _iter_image_paths(input_dir: Path, recursive: bool) -> list[Path]:
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _convert_one_image(image_path: Path, output_path: Path, quality: int) -> None:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        rgb_image = _to_rgb(image)
        rgb_image.save(output_path, format="JPEG", quality=quality)


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        background = Image.new("RGB", image.size, "white")
        alpha = image.convert("RGBA").getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def _register_heic_support() -> None:
    if register_heif_opener is not None:
        register_heif_opener()


if __name__ == "__main__":
    main()
