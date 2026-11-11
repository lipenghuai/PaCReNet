"""Frozen contribution gating and checkpoint contract for two-stage regression."""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .patch_contribution import PatchContributionNet
from .connectivity import decode_patch_classes, validate_connectivity
from .vit_regressor import PigWeightViT


STAGE1_SCHEMA_VERSION = 1
STAGE1_CLASS_NAMES = ("background", "full_pig", "pig_edge")
_STAGE1_MODALITY_CHANNELS = {"rgb": 3, "depth": 1, "rgbd": 4}
_STAGE1_PROFILES = {"tiny", "small", "base", "large"}
_CHECKPOINT_REQUIRED_FIELDS = {
    "schema_version",
    "metadata",
    "model_state",
    "epoch",
    "optimizer_state",
    "scheduler_state",
    "best_validation_accuracy",
}


class CheckpointCompatibilityError(ValueError):
    """Raised when a stage-one checkpoint cannot satisfy an expected contract."""


@dataclass(frozen=True, slots=True)
class Stage1Metadata:
    """Architecture and preprocessing contract stored with a stage-one model."""

    schema_version: int
    modality: str
    input_channels: int
    image_size: tuple[int, int]
    patch_size: int
    class_names: tuple[str, str, str]
    depth_max_mm: float
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    profile: str
    overrides: dict[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", tuple(self.image_size))
        object.__setattr__(self, "class_names", tuple(self.class_names))
        object.__setattr__(self, "rgb_mean", tuple(self.rgb_mean))
        object.__setattr__(self, "rgb_std", tuple(self.rgb_std))
        object.__setattr__(self, "overrides", dict(self.overrides))


@dataclass(slots=True)
class WeightForwardOutput:
    """Weight prediction plus token-selection diagnostics."""

    prediction: torch.Tensor
    keep_mask: torch.Tensor
    selected_counts: torch.Tensor
    fallback_count: int


def assert_compatible_metadata(
    checkpoint: Stage1Metadata, expected: Stage1Metadata
) -> None:
    """Report every incompatible stage-one metadata field in one error."""
    mismatches = []
    for metadata_field in dataclasses.fields(Stage1Metadata):
        name = metadata_field.name
        checkpoint_value = getattr(checkpoint, name)
        expected_value = getattr(expected, name)
        if checkpoint_value != expected_value:
            mismatches.append(
                f"{name}: checkpoint={checkpoint_value!r}, expected={expected_value!r}"
            )
    if mismatches:
        raise CheckpointCompatibilityError(
            "incompatible stage-1 checkpoint metadata: " + "; ".join(mismatches)
        )


def save_stage1_checkpoint(
    path: Path,
    model: PatchContributionNet,
    metadata: Stage1Metadata,
    *,
    epoch: int,
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any],
    best_validation_accuracy: float,
) -> None:
    """Atomically save a stage-one model and its complete input contract."""
    payload = {
        "schema_version": STAGE1_SCHEMA_VERSION,
        "metadata": dataclasses.asdict(metadata),
        "model_state": model.state_dict(),
        "epoch": epoch,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "best_validation_accuracy": best_validation_accuracy,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_numeric_sequence(
    values: Any,
    *,
    field_name: str,
    length: int,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise CheckpointCompatibilityError(
            f"stage-1 checkpoint metadata.{field_name} must contain {length} numbers"
        )
    normalized = tuple(values)
    if not all(_is_finite_number(value) for value in normalized) or (
        positive and any(float(value) <= 0.0 for value in normalized)
    ):
        qualifier = "finite positive" if positive else "finite"
        raise CheckpointCompatibilityError(
            f"stage-1 checkpoint metadata.{field_name} must contain {length} {qualifier} numbers"
        )
    return tuple(float(value) for value in normalized)


def _validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint payload must be a mapping"
        )
    missing = sorted(_CHECKPOINT_REQUIRED_FIELDS.difference(payload))
    if missing:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint is missing required top-level field(s): "
            + ", ".join(missing)
        )
    normalized = dict(payload)
    schema_version = normalized["schema_version"]
    if not _is_integer(schema_version) or schema_version != STAGE1_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "incompatible stage-1 checkpoint schema_version: "
            f"checkpoint={schema_version!r}, expected={STAGE1_SCHEMA_VERSION!r}"
        )
    if not isinstance(normalized["metadata"], Mapping):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata must be a mapping"
        )
    model_state = normalized["model_state"]
    if not isinstance(model_state, Mapping) or not model_state:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint model_state must be a nonempty mapping"
        )
    if any(
        not isinstance(key, str) or not key or not isinstance(value, torch.Tensor)
        for key, value in model_state.items()
    ):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint model_state must map nonempty string keys to tensors"
        )
    epoch = normalized["epoch"]
    if not _is_integer(epoch) or epoch < 0:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint epoch must be a nonnegative integer"
        )
    for field_name in ("optimizer_state", "scheduler_state"):
        if not isinstance(normalized[field_name], Mapping):
            raise CheckpointCompatibilityError(
                f"stage-1 checkpoint {field_name} must be a mapping"
            )
    best_accuracy = normalized["best_validation_accuracy"]
    if not _is_finite_number(best_accuracy) or not 0.0 <= float(best_accuracy) <= 1.0:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint best_validation_accuracy must be finite and within [0, 1]"
        )
    return normalized


def _metadata_from_mapping(values: Mapping[str, Any]) -> Stage1Metadata:
    required_fields = {field.name for field in dataclasses.fields(Stage1Metadata)}
    missing = sorted(required_fields.difference(values))
    unexpected = sorted(set(values).difference(required_fields))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata fields are malformed: " + "; ".join(details)
        )

    schema_version = values["schema_version"]
    if not _is_integer(schema_version) or schema_version != STAGE1_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.schema_version must equal "
            f"{STAGE1_SCHEMA_VERSION}"
        )
    modality = values["modality"]
    if modality not in _STAGE1_MODALITY_CHANNELS:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.modality is unsupported"
        )
    input_channels = values["input_channels"]
    if (
        not _is_integer(input_channels)
        or input_channels != _STAGE1_MODALITY_CHANNELS[modality]
    ):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.input_channels disagrees with modality"
        )
    image_size = values["image_size"]
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(not _is_integer(value) or value <= 0 for value in image_size)
    ):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.image_size must contain two positive integers"
        )
    patch_size = values["patch_size"]
    if not _is_integer(patch_size) or patch_size <= 0:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.patch_size must be a positive integer"
        )
    if any(int(size) % int(patch_size) for size in image_size):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.patch_size must divide metadata.image_size"
        )
    class_names = values["class_names"]
    if (
        not isinstance(class_names, (list, tuple))
        or len(class_names) != 3
        or any(not isinstance(name, str) or not name for name in class_names)
    ):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.class_names must contain three nonempty strings"
        )
    depth_max_mm = values["depth_max_mm"]
    if not _is_finite_number(depth_max_mm) or float(depth_max_mm) <= 0.0:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.depth_max_mm must be finite and positive"
        )
    rgb_mean = _require_numeric_sequence(
        values["rgb_mean"], field_name="rgb_mean", length=3
    )
    rgb_std = _require_numeric_sequence(
        values["rgb_std"], field_name="rgb_std", length=3, positive=True
    )
    profile = values["profile"]
    if profile not in _STAGE1_PROFILES:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.profile is unsupported"
        )
    overrides = values["overrides"]
    if not isinstance(overrides, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not _is_finite_number(value)
        for key, value in overrides.items()
    ):
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint metadata.overrides must map strings to finite numbers"
        )

    try:
        return Stage1Metadata(
            schema_version=int(schema_version),
            modality=modality,
            input_channels=int(input_channels),
            image_size=tuple(int(value) for value in image_size),
            patch_size=int(patch_size),
            class_names=tuple(class_names),
            depth_max_mm=float(depth_max_mm),
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            profile=profile,
            overrides=dict(overrides),
        )
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"stage-1 checkpoint metadata is malformed: {error}"
        ) from error


def load_frozen_contributor(
    path: Path,
    *,
    expected: Stage1Metadata | None,
    map_location: str | torch.device,
) -> tuple[PatchContributionNet, Stage1Metadata, dict[str, Any]]:
    """Load and strictly freeze a compatible contribution classifier."""
    payload = _validate_checkpoint_payload(
        torch.load(path, map_location=map_location, weights_only=False)
    )
    payload_schema = payload["schema_version"]
    metadata_values = payload["metadata"]
    metadata = _metadata_from_mapping(metadata_values)
    if metadata.schema_version != payload_schema:
        raise CheckpointCompatibilityError(
            "stage-1 checkpoint schema_version disagrees with metadata: "
            f"checkpoint={payload_schema!r}, metadata={metadata.schema_version!r}"
        )
    if expected is not None:
        assert_compatible_metadata(metadata, expected)

    try:
        model = PatchContributionNet(
            metadata.input_channels,
            metadata.image_size,
            metadata.patch_size,
            metadata.profile,
            overrides=metadata.overrides,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"stage-1 checkpoint metadata cannot construct contributor: {error}"
        ) from error
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"stage-1 checkpoint model_state is incompatible: {error}"
        ) from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata, payload


class TwoStagePigWeightModel(nn.Module):
    """Select and optionally route regression patches with a frozen contributor."""

    def __init__(
        self, contributor: PatchContributionNet, regressor: PigWeightViT,
        *, enforce_connectivity: bool = False, connectivity: int = 8,
    ) -> None:
        super().__init__()
        validate_connectivity(connectivity)
        if not isinstance(enforce_connectivity, bool):
            raise ValueError("enforce_connectivity must be a boolean")
        self.contributor = contributor
        self.regressor = regressor
        self.enforce_connectivity = enforce_connectivity
        self.connectivity = connectivity
        self._freeze_contributor()

    def _freeze_contributor(self) -> None:
        self.contributor.eval()
        for parameter in self.contributor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "TwoStagePigWeightModel":
        super().train(mode)
        self.contributor.eval()
        return self

    def forward(
        self, contributor_input: torch.Tensor, regression_input: torch.Tensor
    ) -> WeightForwardOutput:
        """Select body/edge tokens and preserve their semantic expert routes."""
        self.contributor.eval()
        with torch.no_grad():
            logits = self.contributor(contributor_input)
            predicted_classes = (
                decode_patch_classes(
                    logits, self.contributor.grid_size,
                    enforce_connectivity=True, connectivity=self.connectivity,
                )
                if self.enforce_connectivity else logits.argmax(dim=-1)
            )
            keep_mask = (predicted_classes == 1) | (predicted_classes == 2)
            # Dense zero grids remain valid through CLS even with no foreground.
            # Only sparse selection needs the legacy one-patch rescue.
            fallback_rows = (
                ~keep_mask.any(dim=1)
                if self.regressor.background_mode == "drop"
                else keep_mask.new_zeros(keep_mask.shape[0])
            )
            if torch.any(fallback_rows):
                pig_scores = logits.softmax(dim=-1)[..., 1:].sum(dim=-1)
                fallback_indices = pig_scores[fallback_rows].argmax(dim=1)
                keep_mask = keep_mask.clone()
                keep_mask[fallback_rows, fallback_indices] = True
                # The fallback still needs a body/edge route even though its
                # original argmax was background. Use its strongest pig class.
                predicted_classes = predicted_classes.clone()
                fallback_classes = (
                    logits[fallback_rows, fallback_indices, 1:].argmax(dim=-1) + 1
                )
                predicted_classes[fallback_rows, fallback_indices] = fallback_classes

        if self.regressor.expert_routing:
            regressor_output = self.regressor(
                regression_input, keep_mask, patch_classes=predicted_classes
            )
        else:
            regressor_output = self.regressor(regression_input, keep_mask)
        return WeightForwardOutput(
            prediction=regressor_output.prediction,
            keep_mask=keep_mask,
            selected_counts=regressor_output.selected_counts,
            fallback_count=int(fallback_rows.sum().item()),
        )
