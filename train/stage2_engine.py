"""One-epoch training and deterministic evaluation for weight regression."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data.preprocessing import select_modality
from model.two_stage_model import TwoStagePigWeightModel
from train.metrics import PredictionRow, regression_reports


def train_stage2_epoch(
    model: TwoStagePigWeightModel,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    contributor_modality: str,
    regressor_modality: str,
    scaler: Any | None,
    grad_clip: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    beta: float = 1.0,
    show_progress: bool = False,
    progress_description: str | None = None,
) -> dict[str, Any]:
    """Optimize only the regressor and report frame- and pig-level metrics."""
    _validate_grad_clip(grad_clip)
    beta = _validate_smooth_l1_beta(beta)
    _assert_contributor_frozen(model)
    model.train()
    rows: list[PredictionRow] = []
    loss_total = 0.0
    fallback_count = 0
    progress_bar = (
        tqdm(
            loader,
            desc=progress_description or "Stage 2 training",
            unit="iter",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    iterator = progress_bar if progress_bar is not None else loader

    for batch in iterator:
        contributor_input, regression_input, targets = _move_stage2_batch(
            batch,
            contributor_modality,
            regressor_modality,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, scaler):
            output = model(contributor_input, regression_input)
            predictions = _validate_predictions(output.prediction, targets)
            loss = F.smooth_l1_loss(predictions, targets, beta=beta)
        _require_finite_loss(loss)

        if scaler is None:
            loss.backward()
            _assert_contributor_frozen(model)
            torch.nn.utils.clip_grad_norm_(model.regressor.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            _assert_contributor_frozen(model)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.regressor.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch_rows = _prediction_rows(batch, targets, predictions)
        rows.extend(batch_rows)
        loss_total += float(loss.detach().item()) * len(batch_rows)
        fallback_count += int(output.fallback_count)
        if progress_bar is not None:
            progress_bar.set_postfix(
                {"avg_loss": loss_total / len(rows)},
                refresh=True,
            )

    optimizer.zero_grad(set_to_none=True)
    return _finalize_report(rows, loss_total, fallback_count)


def evaluate_stage2(
    model: TwoStagePigWeightModel,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    contributor_modality: str,
    regressor_modality: str,
    beta: float = 1.0,
) -> tuple[dict[str, Any], list[PredictionRow]]:
    """Evaluate every frame without gradients, shuffling, or dataset mutation."""
    _assert_contributor_frozen(model)
    beta = _validate_smooth_l1_beta(beta)
    model.eval()
    rows: list[PredictionRow] = []
    loss_total = 0.0
    fallback_count = 0

    with torch.no_grad():
        for batch in loader:
            contributor_input, regression_input, targets = _move_stage2_batch(
                batch,
                contributor_modality,
                regressor_modality,
                device,
            )
            output = model(contributor_input, regression_input)
            predictions = _validate_predictions(output.prediction, targets)
            loss = F.smooth_l1_loss(predictions, targets, beta=beta)
            _require_finite_loss(loss)

            batch_rows = _prediction_rows(batch, targets, predictions)
            rows.extend(batch_rows)
            loss_total += float(loss.detach().item()) * len(batch_rows)
            fallback_count += int(output.fallback_count)

    return _finalize_report(rows, loss_total, fallback_count), rows


def _move_stage2_batch(
    batch: Mapping[str, Any],
    contributor_modality: str,
    regressor_modality: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contributor_input = select_modality(batch, contributor_modality).to(
        device=device, non_blocking=True
    )
    if contributor_modality == regressor_modality:
        regression_input = contributor_input
    else:
        regression_input = select_modality(batch, regressor_modality).to(
            device=device, non_blocking=True
        )
    targets = batch["weight_kg"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return contributor_input, regression_input, targets.reshape(-1)


def _validate_predictions(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    predictions = predictions.reshape(-1)
    if predictions.numel() != targets.numel():
        raise ValueError("prediction and weight_kg must contain the same batch size")
    return predictions


def _prediction_rows(
    batch: Mapping[str, Any],
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> list[PredictionRow]:
    batch_size = targets.numel()
    splits = _string_batch(batch, "split", batch_size)
    pig_ids = _string_batch(batch, "pig_id", batch_size)
    frame_ids = _string_batch(batch, "frame_id", batch_size)
    target_values = targets.detach().to(device="cpu", dtype=torch.float32).tolist()
    prediction_values = (
        predictions.detach().to(device="cpu", dtype=torch.float32).tolist()
    )
    return [
        PredictionRow(
            split=split,
            pig_id=pig_id,
            frame_id=frame_id,
            target_kg=float(target),
            prediction_kg=float(prediction),
        )
        for split, pig_id, frame_id, target, prediction in zip(
            splits,
            pig_ids,
            frame_ids,
            target_values,
            prediction_values,
            strict=True,
        )
    ]


def _string_batch(
    batch: Mapping[str, Any], key: str, batch_size: int
) -> list[str]:
    values = batch[key]
    if isinstance(values, str):
        normalized = [values]
    elif isinstance(values, Sequence):
        normalized = list(values)
    else:
        raise ValueError(f"{key} must be a string sequence")
    if len(normalized) != batch_size or not all(
        isinstance(value, str) and value for value in normalized
    ):
        raise ValueError(f"{key} must contain one nonempty string per frame")
    return normalized


def _finalize_report(
    rows: list[PredictionRow], loss_total: float, fallback_count: int
) -> dict[str, Any]:
    report = regression_reports(rows)
    report["loss"] = loss_total / len(rows)
    report["fallback_count"] = fallback_count
    return report


def _autocast_context(device: torch.device, scaler: Any | None):
    if device.type == "cuda" and scaler is not None:
        return torch.autocast(device_type="cuda")
    return nullcontext()


def _assert_contributor_frozen(model: TwoStagePigWeightModel) -> None:
    parameters = tuple(model.contributor.parameters())
    if any(parameter.requires_grad for parameter in parameters) or any(
        parameter.grad is not None for parameter in parameters
    ):
        raise RuntimeError("contributor must remain fully frozen with no gradients")


def _require_finite_loss(loss: torch.Tensor) -> None:
    if loss.numel() != 1:
        raise ValueError("regression loss must be scalar")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("non-finite regression loss")


def _validate_grad_clip(grad_clip: float) -> None:
    try:
        numeric = float(grad_clip)
    except (TypeError, ValueError) as error:
        raise ValueError("grad_clip must be finite and positive") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("grad_clip must be finite and positive")


def _validate_smooth_l1_beta(beta: float) -> float:
    if (
        isinstance(beta, bool)
        or not isinstance(beta, Real)
        or not math.isfinite(float(beta))
        or float(beta) <= 0.0
    ):
        raise ValueError("Smooth-L1 beta must be finite and positive")
    return float(beta)
