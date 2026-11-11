"""Configuration loading and validation for the two training stages."""

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


_MODALITIES = {"rgb", "depth", "rgbd"}
_PROFILES = {"tiny", "small", "base", "large"}


def load_config(path: Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load a YAML mapping and apply existing dotted-key overrides."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    for item in overrides:
        dotted_key, separator, raw_value = item.partition("=")
        if not separator or not dotted_key:
            raise ValueError(f"Override must be key=value: {item}")
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise KeyError(f"Unknown config path: {dotted_key}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise KeyError(f"Unknown config path: {dotted_key}")
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return config


def _validate_image_grid(config: Mapping[str, Any]) -> None:
    height, width = config["data"]["image_size"]
    patch = config["data"]["patch_size"]
    if (height, width) != (480, 640):
        raise ValueError("image_size must remain [480, 640]")
    if isinstance(patch, bool) or not isinstance(patch, int) or patch <= 0:
        raise ValueError("patch_size must be a positive integer")
    if height % patch or width % patch:
        raise ValueError("patch_size must divide 480 and 640")


def _validate_modality_and_profile(modality: Any, profile: Any) -> None:
    if modality not in _MODALITIES:
        raise ValueError(f"modality must be one of {sorted(_MODALITIES)}")
    if profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}")


def _validate_connectivity_config(config: Mapping[str, Any]) -> None:
    settings = config.get("connectivity", {})
    if not isinstance(settings, Mapping):
        raise ValueError("connectivity must be a mapping")
    weight = settings.get("loss_weight", 0.0)
    if (
        isinstance(weight, bool) or not isinstance(weight, Real)
        or not math.isfinite(float(weight)) or weight < 0
    ):
        raise ValueError("connectivity.loss_weight must be finite and nonnegative")
    neighborhood = settings.get("neighborhood", 8)
    if (
        isinstance(neighborhood, bool) or not isinstance(neighborhood, Integral)
        or neighborhood not in (4, 8)
    ):
        raise ValueError("connectivity.neighborhood must be 4 or 8")
    if not isinstance(settings.get("postprocess", False), bool):
        raise ValueError("connectivity.postprocess must be a boolean")


def validate_stage1_config(config: Mapping[str, Any]) -> None:
    """Validate the fixed image grid and stage-one segmentation settings."""
    _validate_image_grid(config)
    _validate_connectivity_config(config)
    _validate_modality_and_profile(
        config["model"]["modality"], config["model"]["profile"]
    )

    probability = config["augmentation"]["probability"]
    if not 0 <= probability <= 1:
        raise ValueError("augmentation probability must be in [0, 1]")

    background_max = config["patch_labels"]["background_max"]
    full_min = config["patch_labels"]["full_min"]
    if not 0 <= background_max < full_min <= 1:
        raise ValueError("patch label thresholds must satisfy 0 <= background_max < full_min <= 1")


def validate_stage2_config(config: Mapping[str, Any]) -> None:
    """Validate the fixed image grid and stage-two regression settings."""
    _validate_image_grid(config)
    _validate_connectivity_config(config)

    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training must be a mapping")
    if "evaluation_batch_size" in training:
        evaluation_batch_size = training["evaluation_batch_size"]
        if (
            isinstance(evaluation_batch_size, bool)
            or not isinstance(evaluation_batch_size, Integral)
            or evaluation_batch_size <= 0
        ):
            raise ValueError("training.evaluation_batch_size must be a positive integer")
    if "pin_memory" in training and not isinstance(training["pin_memory"], bool):
        raise ValueError("training.pin_memory must be a boolean")

    checkpoint = config.get("stage1_checkpoint")
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
        raise ValueError("stage1_checkpoint must be a non-empty path")

    _validate_modality_and_profile(
        config["model"]["regressor_modality"],
        config["model"]["regressor_profile"],
    )
    if not isinstance(config["model"].get("expert_routing", False), bool):
        raise ValueError("model.expert_routing must be a boolean")
    background_mode = config["model"].get("background_mode", "drop")
    if not isinstance(background_mode, str) or background_mode not in ("drop", "zero"):
        raise ValueError("model.background_mode must be 'drop' or 'zero'")
    if config["loss"]["name"] != "smooth_l1":
        raise ValueError("loss.name must be 'smooth_l1'")
    beta = config["loss"]["beta"]
    if (
        isinstance(beta, bool)
        or not isinstance(beta, Real)
        or not math.isfinite(float(beta))
        or float(beta) <= 0.0
    ):
        raise ValueError("loss.beta must be finite and positive")
