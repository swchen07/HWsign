from __future__ import annotations

from PIL import Image

from src.common.preprocess_images import convert_folder_to_jpg


def test_convert_folder_to_jpg_from_png(tmp_path) -> None:
    input_dir = tmp_path / "sample_images"
    input_dir.mkdir()
    Image.new("RGBA", (12, 8), (255, 0, 0, 128)).save(input_dir / "signature.png")

    summary = convert_folder_to_jpg(input_dir, output_root=tmp_path / "processed")

    output_path = tmp_path / "processed" / "sample_images_jpg" / "signature.jpg"
    assert summary.converted == 1
    assert summary.failed == 0
    assert output_path.exists()

    with Image.open(output_path) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
