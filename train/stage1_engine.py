"""One-epoch training and evaluation for patch contribution classification."""

from __future__ import annotations

import math
from numbers import Real
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data.preprocessing import select_modality
from train.metrics import PatchMetricAccumulator
from train.topology_loss import topology_connectivity_loss
from model.connectivity import decode_patch_classes, validate_connectivity, validate_grid_size


PatchLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_patch_loss(
    name: str,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    *,
    grid_size: tuple[int, int] | None = None,
    connectivity_weight: float = 0.0,
    connectivity: int = 8,
) -> PatchLoss:
    """Build classification loss with an optional spatial connectivity prior."""
    if name not in {"weighted_cross_entropy", "focal"}:
        raise ValueError(f"Unsupported patch loss: {name}")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if (
        isinstance(connectivity_weight, bool)
        or not isinstance(connectivity_weight, Real)
        or not math.isfinite(float(connectivity_weight))
        or connectivity_weight < 0
    ):
        raise ValueError("connectivity_weight must be finite and nonnegative")
    validate_connectivity(connectivity)
    if connectivity_weight > 0 and grid_size is None:
        raise ValueError("grid_size is required when connectivity_weight is positive")
    if grid_size is not None:
        grid_size = validate_grid_size(grid_size)
    weights = torch.as_tensor(class_weights, dtype=torch.float32).detach().clone()
    if weights.ndim != 1 or weights.numel() != 3:
        raise ValueError("class_weights must contain exactly three values")
    if not torch.all(torch.isfinite(weights)) or torch.any(weights < 0):
        raise ValueError("class_weights must be finite and nonnegative")

    def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=weights.to(logits.device))

    def focal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_item_ce = F.cross_entropy(
            logits,
            labels,
            weight=weights.to(logits.device),
            reduction="none",
        )
        return (((1.0 - torch.exp(-per_item_ce)) ** gamma) * per_item_ce).mean()

    classification_loss = weighted_cross_entropy if name == "weighted_cross_entropy" else focal_loss
    if connectivity_weight == 0:
        return classification_loss

    patch_count = grid_size[0] * grid_size[1]

    def spatial_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[-1] != 3 or logits.shape[0] % patch_count:
            raise ValueError("flattened logits must contain complete patch grids")
        if labels.ndim != 1 or labels.numel() != logits.shape[0]:
            raise ValueError("flattened labels must match the patch grids")
        prior = topology_connectivity_loss(
            logits.reshape(-1, patch_count, 3),
            labels.reshape(-1, patch_count),
            grid_size,
            connectivity,
        )
        return classification_loss(logits, labels) + connectivity_weight * prior

    return spatial_loss


def train_stage1_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality: str,
    class_weights: torch.Tensor,
    scaler: Any | None,
    grad_clip: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    loss_function: PatchLoss | None = None,
    show_progress: bool = False,
    progress_description: str | None = None,
    connectivity: int = 8,
) -> dict[str, Any]:
    """Optimize the patch contributor for one loader pass."""
    _validate_grad_clip(grad_clip)
    model.train()
    metrics = PatchMetricAccumulator(
        num_classes=3, grid_size=getattr(model, "grid_size", None), connectivity=connectivity
    )
    loss_function = loss_function or build_patch_loss(
        "weighted_cross_entropy", class_weights
    )
    progress_bar = (
        tqdm(
            loader,
            desc=progress_description or "Stage 1 training",
            unit="iter",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    iterator = progress_bar if progress_bar is not None else loader
    running_loss_total = 0.0
    running_patch_count = 0

    for batch in iterator:
        inputs, labels = _move_stage1_batch(batch, modality, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, scaler):
            logits = model(inputs)
            flattened_logits, flattened_labels = _flatten_predictions(logits, labels)
            loss = loss_function(flattened_logits, flattened_labels)
        _require_finite_loss(loss)

        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        metrics.update(logits.detach(), labels.detach(), loss.detach())
        patch_count = flattened_labels.numel()
        running_loss_total += float(loss.detach().item()) * patch_count
        running_patch_count += patch_count
        if progress_bar is not None and running_patch_count > 0:
            progress_bar.set_postfix(
                {"avg_loss": running_loss_total / running_patch_count},
                refresh=True,
            )

    return metrics.compute()


def evaluate_stage1(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    modality: str,
    class_weights: torch.Tensor,
    loss_function: PatchLoss | None = None,
    *,
    enforce_connectivity: bool = False,
    connectivity: int = 8,
) -> dict[str, Any]:
    """Evaluate the patch contributor without gradients or dataset mutation."""
    model.eval()
    grid_size = getattr(model, "grid_size", None)
    if enforce_connectivity and grid_size is None:
        raise ValueError("model.grid_size is required for connectivity postprocessing")
    metrics = PatchMetricAccumulator(num_classes=3, grid_size=grid_size, connectivity=connectivity)
    processed_metrics = (
        PatchMetricAccumulator(num_classes=3, grid_size=grid_size, connectivity=connectivity)
        if enforce_connectivity else None
    )
    loss_function = loss_function or build_patch_loss(
        "weighted_cross_entropy", class_weights
    )

    with torch.no_grad():
        for batch in loader:
            inputs, labels = _move_stage1_batch(batch, modality, device)
            logits = model(inputs)
            flattened_logits, flattened_labels = _flatten_predictions(logits, labels)
            loss = loss_function(flattened_logits, flattened_labels)
            _require_finite_loss(loss)
            metrics.update(logits.detach(), labels.detach(), loss.detach())
            if processed_metrics is not None:
                predicted_classes = decode_patch_classes(
                    logits, grid_size, enforce_connectivity=True, connectivity=connectivity
                )
                processed_metrics.update(F.one_hot(predicted_classes, 3), labels, loss)

    report = metrics.compute()
    if processed_metrics is not None:
        report["postprocessed"] = processed_metrics.compute()
    return report


def _move_stage1_batch(
    batch: Mapping[str, Any], modality: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = select_modality(batch, modality).to(device=device, non_blocking=True)
    labels = batch["patch_labels"].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return inputs, labels


def _flatten_predictions(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim < 2 or logits.shape[-1] != 3:
        raise ValueError("model logits must end in exactly three class scores")
    flattened_logits = logits.reshape(-1, 3)
    flattened_labels = labels.reshape(-1)
    if flattened_logits.shape[0] != flattened_labels.numel():
        raise ValueError("logits and patch_labels must contain the same patch count")
    return flattened_logits, flattened_labels


def _autocast_context(device: torch.device, scaler: Any | None):
    if device.type == "cuda" and scaler is not None:
        return torch.autocast(device_type="cuda")
    return nullcontext()


def _require_finite_loss(loss: torch.Tensor) -> None:
    if loss.numel() != 1:
        raise ValueError("patch loss must be scalar")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("non-finite patch loss")


def _validate_grad_clip(grad_clip: float) -> None:
    try:
        numeric = float(grad_clip)
    except (TypeError, ValueError) as error:
        raise ValueError("grad_clip must be finite and positive") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("grad_clip must be finite and positive")
