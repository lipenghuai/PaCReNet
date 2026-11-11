"""Shared deterministic setup, device, and optimizer utilities."""

from __future__ import annotations

import math
import platform
import random
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without probing CUDA unnecessarily."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a data-loader worker from the seed assigned by PyTorch."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested: str | torch.device) -> torch.device:
    """Resolve an explicit CPU/CUDA device or automatically choose one."""
    requested_text = str(requested).strip().lower()
    if requested_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested_text)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Unsupported device: {requested}") from error
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError("device must be 'auto', 'cpu', or a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested_text}")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index is unavailable: {requested_text}")
    return device


def collect_environment() -> dict[str, Any]:
    """Return JSON-serializable runtime and accelerator metadata."""
    cuda_available = torch.cuda.is_available()
    gpus = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpus": gpus,
        "os": platform.platform(),
    }


def build_grad_scaler(device: torch.device, *, enabled: bool) -> Any | None:
    """Build a CUDA AMP scaler across supported PyTorch API generations."""
    if device.type != "cuda" or not enabled:
        return None
    modern_factory = getattr(torch.amp, "GradScaler", None)
    if modern_factory is not None:
        return modern_factory("cuda")
    return torch.cuda.amp.GradScaler()


def build_optimizer_scheduler(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    patch_count: int | None = None,
    profiles: str | Sequence[str] = (),
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR, int]:
    """Build AdamW and a per-step linear-warmup/cosine-decay schedule."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    _require_nonnegative_integer("warmup_epochs", warmup_epochs)
    _require_positive_integer("epochs", epochs)
    _require_positive_integer("steps_per_epoch", steps_per_epoch)

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    if warmup_steps >= total_steps:
        raise ValueError("warmup steps must be fewer than total training steps")
    _warn_for_patch_memory(patch_count, profiles)

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    def learning_rate_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        cosine_steps = total_steps - warmup_steps
        progress = min(max(step - warmup_steps, 0), cosine_steps) / cosine_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    return optimizer, scheduler, total_steps


def _warn_for_patch_memory(
    patch_count: int | None, profiles: str | Sequence[str]
) -> None:
    if patch_count is None:
        return
    _require_positive_integer("patch_count", patch_count)
    profile_values = (profiles,) if isinstance(profiles, str) else profiles
    normalized_profiles = {str(profile).lower() for profile in profile_values}
    if patch_count > 300 and normalized_profiles.intersection({"base", "large"}):
        print(
            "WARNING: patch count exceeds 300 with a base/large profile; "
            "GPU memory use may be high."
        )


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
