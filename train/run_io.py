"""Atomic, reproducible writers for training-run artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

from train.metrics import PredictionRow


_PREDICTION_FIELDS = (
    "split",
    "pig_id",
    "frame_id",
    "target_kg",
    "prediction_kg",
    "absolute_error_kg",
)

_PATCH_CLASS_FIELDS = (
    "class_index",
    "class_name",
    "precision",
    "recall",
    "f1",
)

_PIG_PREDICTION_FIELDS = (
    "split",
    "pig_id",
    "target_kg",
    "prediction_kg",
    "absolute_error_kg",
    "frame_count",
)


@dataclass(frozen=True)
class RunDirectory:
    """A timestamped run root with scoped, atomically-written artifacts."""

    root: Path

    @classmethod
    def create(
        cls,
        base_directory: Path,
        stage: str,
        modality: str,
        profile: str,
        *,
        now: datetime | None = None,
    ) -> "RunDirectory":
        timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        root = Path(base_directory) / f"{timestamp}_{stage}_{modality}_{profile}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "checkpoints").mkdir()
        (root / "predictions").mkdir()
        return cls(root=root)

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    def write_json(self, filename: str | Path, data: Any) -> Path:
        path = self._artifact_path(filename)
        _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path

    def write_yaml(self, filename: str | Path, data: Any) -> Path:
        path = self._artifact_path(filename)
        _atomic_write_text(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=True))
        return path

    def append_metrics(self, epoch: int, split: str, metrics: Mapping[str, Any]) -> None:
        if "epoch" in metrics or "split" in metrics:
            raise ValueError("metrics must not overwrite epoch or split")
        record = {"epoch": epoch, "split": split, **dict(metrics)}
        metrics_path = self.root / "metrics.jsonl"
        records = _read_jsonl(metrics_path)
        records.append(record)
        _atomic_write_text(
            metrics_path,
            "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in records),
        )
        _write_metrics_csv(self.root / "metrics.csv", records)

    def checkpoint_path(self, filename: str | Path) -> Path:
        return self._child_path(self.checkpoints_dir, filename)

    def prediction_path(self, filename: str | Path) -> Path:
        return self._child_path(self.predictions_dir, filename)

    def write_checkpoint(self, filename: str | Path, state: Any) -> Path:
        path = self.checkpoint_path(filename)
        atomic_torch_save(path, state)
        return path

    def write_predictions(
        self, rows: Iterable[PredictionRow], filename: str | Path = "predictions.csv"
    ) -> Path:
        path = self.prediction_path(filename)
        write_prediction_csv(path, rows)
        return path

    def write_pig_predictions(
        self, rows: Iterable[PredictionRow], filename: str | Path
    ) -> Path:
        """Aggregate frame estimates per pig and write a stable summary CSV."""
        path = self.prediction_path(filename)
        write_pig_prediction_csv(path, rows)
        return path

    def write_confusion_matrix(
        self,
        split: str,
        matrix: Iterable[Iterable[int]],
        class_names: Iterable[str],
    ) -> Path:
        """Write a labelled patch-confusion matrix for one evaluation split."""
        names = tuple(class_names)
        rows = tuple(tuple(int(value) for value in row) for row in matrix)
        if not names or len(rows) != len(names) or any(len(row) != len(names) for row in rows):
            raise ValueError("confusion matrix must be square and match class_names")
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("true_class", *names))
        for name, row in zip(names, rows, strict=True):
            writer.writerow((name, *row))
        path = self.prediction_path(f"{split}_confusion_matrix.csv")
        _atomic_write_text(path, output.getvalue())
        return path

    def write_patch_class_summary(
        self,
        split: str,
        per_class: Iterable[Mapping[str, Any]],
        class_names: Iterable[str],
    ) -> Path:
        """Write stable precision/recall/F1 rows for one patch split."""
        names = tuple(class_names)
        metric_rows = tuple(per_class)
        if len(metric_rows) != len(names):
            raise ValueError("per_class rows must match class_names")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=_PATCH_CLASS_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, (name, metrics) in enumerate(zip(names, metric_rows, strict=True)):
            writer.writerow(
                {
                    "class_index": index,
                    "class_name": name,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                }
            )
        path = self.prediction_path(f"{split}_per_class.csv")
        _atomic_write_text(path, output.getvalue())
        return path

    def _artifact_path(self, filename: str | Path) -> Path:
        return self._child_path(self.root, filename)

    @staticmethod
    def _child_path(parent: Path, filename: str | Path) -> Path:
        candidate = parent / Path(filename)
        parent_resolved = parent.resolve()
        candidate_resolved = candidate.resolve()
        if candidate_resolved != parent_resolved and parent_resolved not in candidate_resolved.parents:
            raise ValueError(f"artifact path escapes run directory: {filename}")
        return candidate


def atomic_torch_save(path: Path, state: Any) -> None:
    """Serialize a checkpoint to a sibling temporary file before replacing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    torch.save(state, temporary)
    temporary.replace(path)


def write_prediction_csv(path: Path, rows: Iterable[PredictionRow]) -> None:
    """Write frame predictions in the documented stable CSV field order."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_PREDICTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "split": row.split,
                "pig_id": row.pig_id,
                "frame_id": row.frame_id,
                "target_kg": float(row.target_kg),
                "prediction_kg": float(row.prediction_kg),
                "absolute_error_kg": row.absolute_error_kg,
            }
        )
    _atomic_write_text(path, output.getvalue())


def write_pig_prediction_csv(path: Path, rows: Iterable[PredictionRow]) -> None:
    """Write arithmetic-mean predictions for each split/pig pair."""
    grouped: dict[tuple[str, str], list[PredictionRow]] = {}
    for row in rows:
        grouped.setdefault((row.split, row.pig_id), []).append(row)
    if not grouped:
        raise ValueError("cannot write pig predictions without frame rows")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=_PIG_PREDICTION_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    for (split, pig_id), pig_rows in grouped.items():
        target = float(pig_rows[0].target_kg)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("pig target_kg must be finite and positive")
        if any(float(row.target_kg) != target for row in pig_rows[1:]):
            raise ValueError(f"inconsistent target weights for pig {pig_id!r}")
        predictions = [float(row.prediction_kg) for row in pig_rows]
        if any(not math.isfinite(value) for value in predictions):
            raise ValueError("pig prediction_kg must be finite")
        prediction = sum(predictions) / len(predictions)
        writer.writerow(
            {
                "split": split,
                "pig_id": pig_id,
                "target_kg": target,
                "prediction_kg": prediction,
                "absolute_error_kg": abs(prediction - target),
                "frame_count": len(pig_rows),
            }
        )
    _atomic_write_text(path, output.getvalue())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_metrics_csv(path: Path, records: list[dict[str, Any]]) -> None:
    flattened = [_flatten_scalars(record) for record in records]
    fieldnames = ["epoch", "split"] + sorted(
        {key for row in flattened for key in row if key not in {"epoch", "split"}}
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened)
    _atomic_write_text(path, output.getvalue())


def _flatten_scalars(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if str(key) == "confusion_matrix":
            continue
        if isinstance(item, Mapping):
            flattened.update(_flatten_scalars(item, name))
        elif str(key) == "per_class" and isinstance(item, (list, tuple)):
            for index, class_metrics in enumerate(item):
                if isinstance(class_metrics, Mapping):
                    flattened.update(
                        _flatten_scalars(class_metrics, f"{name}.{index}")
                    )
        elif item is None or isinstance(item, (str, int, float, bool)):
            flattened[name] = item
    return flattened


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(contents, encoding="utf-8", newline="")
    temporary.replace(path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")
