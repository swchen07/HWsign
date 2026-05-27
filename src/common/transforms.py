from __future__ import annotations

from torchvision import transforms


def build_signature_transform(
    input_size: tuple[int, int] = (150, 220),
    augment: bool = False,
) -> transforms.Compose:
    steps: list[object] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(input_size),
    ]

    if augment:
        steps.extend(
            [
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), shear=2)],
                    p=0.5,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            ]
        )

    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
