"""Metrics shared by the patch-contribution and weight-regression stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from model.connectivity import component_counts, validate_connectivity, validate_grid_size


class PatchMetricAccumulator:
    """Accumulate loss and a fixed-size patch-class confusion matrix."""

    def __init__(
        self, num_classes: int, *, grid_size: tuple[int, int] | None = None, connectivity: int = 8
    ) -> None:
        if num_classes != 3:
            raise ValueError("num_classes must be exactly 3 for the background/full/edge taxonomy")
        self.num_classes = num_classes
        self._confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self._loss_total = 0.0
        self._patch_count = 0
        self.grid_size = validate_grid_size(grid_size) if grid_size is not None else None
        validate_connectivity(connectivity)
        self.connectivity = connectivity
        self._component_total = torch.zeros(2, dtype=torch.int64)
        self._connected_total = torch.zeros(2, dtype=torch.int64)
        self._empty_total = torch.zeros(2, dtype=torch.int64)
        self._both_connected_total = 0
        self._frame_count = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float | torch.Tensor) -> None:
        """Add one batch, weighting its mean loss by its valid patch count."""
        if logits.ndim < 2 or logits.shape[-1] != self.num_classes:
            raise ValueError(f"logits must end in {self.num_classes} class scores")
        flattened_logits = logits.detach().reshape(-1, self.num_classes)
        flattened_labels = labels.detach().reshape(-1).to(dtype=torch.long)
        if flattened_logits.shape[0] != flattened_labels.numel():
            raise ValueError("logits and labels must contain the same number of patches")
        if flattened_labels.numel() == 0:
            return
        if torch.any(flattened_labels < 0) or torch.any(flattened_labels >= self.num_classes):
            raise ValueError("labels must be in the configured class range")

        loss_value = _finite_float(loss, "loss")
        predictions = flattened_logits.argmax(dim=-1).to(dtype=torch.long)
        if self.grid_size is not None:
            patch_count = self.grid_size[0] * self.grid_size[1]
            if predictions.numel() % patch_count:
                raise ValueError("predictions must contain complete patch grids")
            counts = component_counts(
                predictions.reshape(-1, patch_count), self.grid_size, self.connectivity
            )
            self._component_total += counts.sum(dim=0)
            self._connected_total += (counts == 1).sum(dim=0)
            self._empty_total += (counts == 0).sum(dim=0)
            self._both_connected_total += int((counts == 1).all(dim=1).sum().item())
            self._frame_count += counts.shape[0]
        indices = flattened_labels * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )
        self._confusion += counts.to(device="cpu", dtype=torch.int64)
        patch_count = flattened_labels.numel()
        self._loss_total += loss_value * patch_count
        self._patch_count += patch_count

    def compute(self) -> dict[str, Any]:
        """Return accuracy, class-wise precision/recall/F1, macro-F1, and loss."""
        if self._patch_count == 0:
            raise ValueError("cannot compute patch metrics without patches")

        matrix = self._confusion.numpy().astype(np.float64, copy=False)
        true_positive = np.diag(matrix)
        predicted_total = matrix.sum(axis=0)
        actual_total = matrix.sum(axis=1)
        precision = np.divide(
            true_positive,
            predicted_total,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=predicted_total != 0,
        )
        recall = np.divide(
            true_positive,
            actual_total,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=actual_total != 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=(precision + recall) != 0,
        )
        report = {
            "loss": self._loss_total / self._patch_count,
            "accuracy": float(true_positive.sum() / self._patch_count),
            "macro_f1": float(f1.mean()),
            "per_class": [
                {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])}
                for index in range(self.num_classes)
            ],
            "confusion_matrix": self._confusion.tolist(),
        }
        if self._frame_count:
            for class_id, name in enumerate(("background", "full_pig")):
                report[f"{name}_mean_components"] = float(self._component_total[class_id]) / self._frame_count
                report[f"{name}_single_component_rate"] = float(self._connected_total[class_id]) / self._frame_count
                report[f"{name}_empty_rate"] = float(self._empty_total[class_id]) / self._frame_count
            report["both_single_component_rate"] = self._both_connected_total / self._frame_count
        return report


@dataclass(frozen=True)
class PredictionRow:
    """One frame-level estimate, with split retained for deterministic CSV export."""

    pig_id: str
    frame_id: str
    target_kg: float
    prediction_kg: float
    split: str = ""

    @property
    def absolute_error_kg(self) -> float:
        return abs(float(self.prediction_kg) - float(self.target_kg))


def regression_reports(rows: Sequence[PredictionRow]) -> dict[str, Any]:
    """Compute frame- and pig-level regression reports from frame predictions."""
    if not rows:
        raise ValueError("cannot compute regression metrics from empty prediction rows")

    _validate_prediction_rows(rows)
    frame_targets = np.asarray([row.target_kg for row in rows], dtype=np.float64)
    frame_predictions = np.asarray([row.prediction_kg for row in rows], dtype=np.float64)
    pig_predictions: list[dict[str, float | str]] = []
    grouped: dict[str, list[PredictionRow]] = {}
    for row in rows:
        grouped.setdefault(row.pig_id, []).append(row)
    for pig_id, pig_rows in grouped.items():
        target = float(pig_rows[0].target_kg)
        if any(float(row.target_kg) != target for row in pig_rows[1:]):
            raise ValueError(f"inconsistent target weights for pig {pig_id!r}")
        prediction = float(np.mean(np.asarray([row.prediction_kg for row in pig_rows], dtype=np.float64)))
        pig_predictions.append(
            {
                "pig_id": pig_id,
                "target_kg": target,
                "prediction_kg": prediction,
                "absolute_error_kg": abs(prediction - target),
            }
        )

    pig_targets = np.asarray([row["target_kg"] for row in pig_predictions], dtype=np.float64)
    pig_estimates = np.asarray([row["prediction_kg"] for row in pig_predictions], dtype=np.float64)
    return {
        "frame": _regression_metric_values(frame_targets, frame_predictions),
        "pig": _regression_metric_values(pig_targets, pig_estimates),
        "pig_predictions": pig_predictions,
    }


def _regression_metric_values(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    errors = predictions - targets
    squared_error = np.square(errors, dtype=np.float64)
    total_variance = float(np.sum(np.square(targets - np.mean(targets, dtype=np.float64)), dtype=np.float64))
    r2 = None if total_variance == 0.0 else float(1.0 - np.sum(squared_error, dtype=np.float64) / total_variance)
    return {
        "mae": float(np.mean(np.abs(errors), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(squared_error, dtype=np.float64))),
        "r2": r2,
        "mape": float(np.mean(np.abs(errors / targets), dtype=np.float64) * 100.0),
    }


def _validate_prediction_rows(rows: Sequence[PredictionRow]) -> None:
    for row in rows:
        target = _finite_float(row.target_kg, "target_kg")
        _finite_float(row.prediction_kg, "prediction_kg")
        if target <= 0.0:
            raise ValueError("target_kg must be positive for regression metrics")


def _finite_float(value: float | torch.Tensor, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        value = value.detach().item()
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric
