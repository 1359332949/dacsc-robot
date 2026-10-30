"""Train-only ColorJitter, directly delegated to Torchvision.

Recipe reference: timm transforms_imagenet_train(color_jitter=0.4) expands a
scalar into brightness/contrast/saturation, with no hue. This is a generic
photometric baseline, not the unreleased CoRP training augmentation recipe.
"""

from __future__ import annotations

import torch
import torchvision
from torchvision.transforms import ColorJitter, functional as TVF

from dacsc.corp_images import IMAGENET_MEAN, IMAGENET_STD

TRAIN_AUGMENTATIONS = ("none", "color_jitter")


def pixel_augmentation_contract(mode: str, seed: int) -> dict | None:
    if mode not in TRAIN_AUGMENTATIONS:
        raise ValueError("unknown training pixel augmentation")
    if mode == "none":
        return None
    if type(seed) is not int or seed < 0:
        raise ValueError("augmentation seed must be a nonnegative integer")
    return {
        "version": "torchvision_training_color_jitter_v1",
        "operator": "torchvision.transforms.ColorJitter",
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.4,
        "hue": 0.0,
        "seed": seed,
        "sampling": "independent_observations_shared_views",
        "position": "after_resize_before_imagenet_normalization",
        "rng": "private_cpu_generator_with_torch.random.fork_rng",
    }


class TrainingPixelAugmentation:
    """Main-process photometric baseline; not a policy/inference module.

    Calling ColorJitter once on [V,3,H,W] reuses its sampled transform across
    views. Call it once per observation, rather than sharing randomness across
    an entire batch. ColorJitter itself samples order and factors unchanged.
    """

    def __init__(self, seed: int):
        self.config = pixel_augmentation_contract("color_jitter", seed)
        self.transform = ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0
        )
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[2] != 3 or not rgb.is_floating_point():
            raise ValueError("augmentation requires unnormalized float RGB [B,V,3,H,W]")
        if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            raise ValueError(
                "training ColorJitter input must be finite RGB in [0,1] before normalization"
            )
        # ColorJitter.get_params uses CPU Torch RNG. Use upstream fork_rng to
        # restore global CPU state even on errors; pixel ops may run on CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            images = torch.stack([self.transform(observation) for observation in rgb])
            self.generator.set_state(torch.get_rng_state())
        return TVF.normalize(
            images, IMAGENET_MEAN.flatten().tolist(), IMAGENET_STD.flatten().tolist()
        )

    def state_dict(self) -> dict:
        return {"config": dict(self.config), "rng": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        if state["config"] != self.config:
            raise ValueError(
                "training augmentation recipe, library version or seed changed"
            )
        self.generator.set_state(state["rng"].cpu())
