"""Shared CoRP image protocol; all image operators come from Torchvision/Pillow."""

from __future__ import annotations

import numpy as np
import math
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as TVF

IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SHIFTS = ("none", "grayscale", "color_jitter", "low_contrast", "hue_shift")


def image_contract() -> dict:
    return {
        "version": "torchvision_photometric_v1",
        "torchvision": torchvision.__version__,
        "resize": "PIL_RGB_uint8_bilinear_224",
        "shift_position": "after_resize_before_normalize",
        "color_jitter": {
            "brightness": 1.6,
            "contrast": 1.6,
            "saturation": 2.2,
            "order": ["brightness", "contrast", "saturation"],
        },
        "low_contrast": 0.3,
        "hue_shift": 0.5,
        "blur": "torchvision.gaussian_blur; kernel=2*ceil(3*sigma)+1; after_resize",
        "note": "fixed photometric stress conditions, not random ColorJitter",
    }


def apply_image_shift(image: torch.Tensor, shift: str) -> torch.Tensor:
    """Float RGB [...,3,H,W] in [0,1]; supports CPU and CUDA tensors."""
    if shift == "none":
        return image
    if shift == "grayscale":
        return TVF.rgb_to_grayscale(image, num_output_channels=3)
    if shift == "low_contrast":
        return TVF.adjust_contrast(image, 0.3)
    if shift == "color_jitter":
        return TVF.adjust_saturation(
            TVF.adjust_contrast(TVF.adjust_brightness(image, 1.6), 1.6), 2.2
        )
    if shift == "hue_shift":
        return TVF.adjust_hue(image, 0.5)
    raise ValueError(f"unknown shift {shift!r}")


def resize_rgb(
    image: np.ndarray | torch.Tensor, size: int = IMAGE_SIZE
) -> torch.Tensor:
    """Official CoRP evaluation uses PIL bilinear resize of uint8 RGB, without crop.

    The DP reader returns CHW float /255 on CPU; reconstruct its uint8 pixels
    before resize so offline samples and simulator HWC uint8 share one path.
    """
    if isinstance(image, torch.Tensor):
        if image.device.type != "cpu" or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("decoded image must be a CPU CHW RGB tensor")
        if image.is_floating_point():
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("decoded float image must be finite and in [0,1]")
            image = image.mul(255).round().to(torch.uint8)
        elif image.dtype != torch.uint8:
            raise ValueError("decoded image must be uint8 or float /255")
        image = image.permute(1, 2, 0).numpy()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("decoded image must be uint8 HWC RGB")
    resized = TVF.resize(
        Image.fromarray(image),
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return TVF.pil_to_tensor(resized).float().div(255)


def apply_corruption(
    image: torch.Tensor, shift: str = "none", blur_sigma: float = 0.0
) -> torch.Tensor:
    if not math.isfinite(blur_sigma) or blur_sigma < 0:
        raise ValueError("blur sigma must be finite and nonnegative")
    image = apply_image_shift(image, shift)
    if blur_sigma > 0:
        image = TVF.gaussian_blur(
            image, 2 * math.ceil(3 * blur_sigma) + 1, [blur_sigma, blur_sigma]
        )
    return image


def prepare_image(
    image, shift: str = "none", size: int = IMAGE_SIZE, blur_sigma: float = 0.0
) -> torch.Tensor:
    return TVF.normalize(
        apply_corruption(resize_rgb(image, size), shift, blur_sigma),
        IMAGENET_MEAN.flatten().tolist(),
        IMAGENET_STD.flatten().tolist(),
    )
