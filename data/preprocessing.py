"""Fixed-resolution RGB-D tensor preprocessing and patch-label generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Mapping

import torch

if TYPE_CHECKING:
    from .dataset import RGBDSample


_PRODUCTION_IMAGE_SIZE = (480, 640)


def _validate_patch_label_parameters(
    patch_size: Any, background_max: Any, full_min: Any
) -> tuple[int, float, float]:
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, Integral)
        or patch_size <= 0
    ):
        raise ValueError("patch_size must be a positive integer")
    thresholds = (background_max, full_min)
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in thresholds
    ) or not 0.0 <= float(background_max) < float(full_min) <= 1.0:
        raise ValueError(
            "patch label thresholds must satisfy 0 <= background_max < full_min <= 1"
        )
    return int(patch_size), float(background_max), float(full_min)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Immutable normalization and patch-label settings for native RGB-D frames."""

    image_size: tuple[int, int] = _PRODUCTION_IMAGE_SIZE
    patch_size: int = 32
    depth_max_mm: float = 4000.0
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    background_max: float = 0.0
    full_min: float = 1.0

    def __post_init__(self) -> None:
        if self.image_size != _PRODUCTION_IMAGE_SIZE:
            raise ValueError("image_size must remain (480, 640)")
        _validate_patch_label_parameters(
            self.patch_size, self.background_max, self.full_min
        )
        height, width = self.image_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("patch_size must divide image_size")
        if self.depth_max_mm <= 0:
            raise ValueError("depth_max_mm must be positive")
        if len(self.rgb_mean) != 3 or len(self.rgb_std) != 3:
            raise ValueError("rgb_mean and rgb_std must each have three values")
        if any(std <= 0 for std in self.rgb_std):
            raise ValueError("rgb_std values must be positive")


def make_patch_labels(
    mask: torch.Tensor | Any,
    patch_size: int,
    background_max: float,
    full_min: float,
) -> torch.Tensor:
    """Classify row-major patches as background (0), full (1), or edge (2)."""
    patch_size, background_max, full_min = _validate_patch_label_parameters(
        patch_size, background_max, full_min
    )
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W]")
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide mask dimensions")
    ratios = mask.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    ratios = ratios.mean(dim=(-1, -2)).reshape(-1)
    labels = torch.full_like(ratios, 2, dtype=torch.long)
    labels[ratios <= background_max] = 0
    labels[ratios >= full_min] = 1
    return labels


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected} for image_size")


def _stable_augmentation_metadata(sample: RGBDSample) -> dict[str, bool | str]:
    """Return the fixed flat metadata schema required for default collation."""
    metadata = sample.augmentation_metadata or {}
    augmented = bool(metadata.get("augmented", False))

    def string_value(key: str, default: str) -> str:
        value = metadata.get(key, default)
        return value if isinstance(value, str) else default

    return {
        "augmented": augmented,
        "source_split": string_value("source_split", sample.split),
        "source_pig_id": string_value("source_pig_id", sample.pig_id),
        "source_frame_id": string_value("source_frame_id", sample.frame_id),
        "target_split": string_value("target_split", "") if augmented else "",
        "target_pig_id": string_value("target_pig_id", "") if augmented else "",
        "target_frame_id": string_value("target_frame_id", "") if augmented else "",
    }


def normalize_sample(sample: RGBDSample, config: PreprocessConfig) -> dict[str, Any]:
    """Convert one native-size sample into normalized RGB, depth, and RGB-D tensors."""
    height, width = config.image_size
    rgb = torch.tensor(sample.rgb)
    depth = torch.tensor(sample.depth)
    depth_seg = torch.tensor(sample.depth_seg)
    mask = torch.tensor(sample.mask)
    _require_shape("rgb", rgb, (height, width, 3))
    _require_shape("depth", depth, (height, width))
    _require_shape("depth_seg", depth_seg, (height, width))
    _require_shape("mask", mask, (height, width))

    rgb = rgb.to(dtype=torch.float32).permute(2, 0, 1).div(255.0)
    mean = torch.tensor(config.rgb_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config.rgb_std, dtype=torch.float32).view(3, 1, 1)
    rgb = (rgb - mean) / std
    depth = depth.to(dtype=torch.float32).clamp(0, config.depth_max_mm).div(config.depth_max_mm)
    depth = depth.unsqueeze(0)

    return {
        "rgb": rgb,
        "depth": depth,
        "rgbd": torch.cat((rgb, depth), dim=0),
        "patch_labels": make_patch_labels(
            mask, config.patch_size, config.background_max, config.full_min
        ),
        "weight_kg": torch.tensor(sample.weight_kg, dtype=torch.float32),
        "split": sample.split,
        "scene": sample.scene,
        "pig_id": sample.pig_id,
        "frame_id": sample.frame_id,
        "augmentation_metadata": _stable_augmentation_metadata(sample),
    }


def select_modality(batch: Mapping[str, Any], modality: str) -> torch.Tensor:
    """Return the requested normalized model-input tensor."""
    if modality not in {"rgb", "depth", "rgbd"}:
        raise ValueError(f"Unsupported modality: {modality}")
    return batch[modality]
