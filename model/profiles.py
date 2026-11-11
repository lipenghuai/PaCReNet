"""Checkpoint-compatible model profile definitions."""

from dataclasses import dataclass, fields, replace
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ModelProfile:
    embed_dim: int
    depth: int
    heads: int
    mlp_ratio: float = 4.0


CONTRIBUTOR_PROFILES = {
    "tiny": ModelProfile(96, 2, 3),
    "small": ModelProfile(192, 4, 3),
    "base": ModelProfile(384, 6, 6),
    "large": ModelProfile(512, 8, 8),
}

REGRESSOR_PROFILES = {
    "tiny": ModelProfile(192, 12, 3),
    "small": ModelProfile(384, 12, 6),
    "base": ModelProfile(768, 12, 12),
    "large": ModelProfile(1024, 24, 16),
}

_PROFILE_FAMILIES = {
    "contributor": CONTRIBUTOR_PROFILES,
    "regressor": REGRESSOR_PROFILES,
}
_PROFILE_FIELDS = {field.name for field in fields(ModelProfile)}


def _validate_profile(profile: ModelProfile) -> None:
    if profile.embed_dim <= 0 or profile.depth <= 0 or profile.heads <= 0 or profile.mlp_ratio <= 0:
        raise ValueError("model profile values must be positive")
    if profile.embed_dim % profile.heads != 0:
        raise ValueError("embed_dim must be divisible by heads")


def get_profile(
    family: str,
    name: str,
    overrides: Mapping[str, int | float] | None = None,
) -> ModelProfile:
    """Return a validated profile copy, applying checkpointed overrides."""
    try:
        profile = _PROFILE_FAMILIES[family][name]
    except KeyError as error:
        raise ValueError(f"unknown model profile {family!r}/{name!r}") from error

    override_values = dict(overrides or {})
    invalid_keys = override_values.keys() - _PROFILE_FIELDS
    if invalid_keys:
        raise ValueError(f"unknown profile override(s): {', '.join(sorted(invalid_keys))}")

    resolved = replace(profile, **override_values)
    _validate_profile(resolved)
    return resolved
