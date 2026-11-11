"""Dataset indexing, auditing, and raw RGB-D frame loading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from .augmentation import OverlayAugmenter
from .preprocessing import PreprocessConfig, normalize_sample


_REQUIRED_SPLITS = frozenset({"train", "validation", "cross_scene_test"})
_TRAIN_TEST_SPLITS = frozenset({"train", "test", "cross_scene_test"})


@dataclass(frozen=True, slots=True)
class FrameRecord:
    split: str
    scene: str
    pig_id: str
    frame_id: str
    weight_kg: float
    rgb_path: Path
    depth_path: Path
    depth_seg_path: Path
    mask_path: Path


@dataclass(slots=True)
class RGBDSample:
    rgb: np.ndarray
    depth: np.ndarray
    depth_seg: np.ndarray
    mask: np.ndarray
    weight_kg: float
    split: str
    scene: str
    pig_id: str
    frame_id: str
    augmentation_metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    dataset_root: Path
    by_split: dict[str, tuple[FrameRecord, ...]]

    def all_records(self) -> tuple[FrameRecord, ...]:
        return tuple(record for records in self.by_split.values() for record in records)


class DataAuditError(ValueError):
    """Raised when error-mode auditing finds one or more invalid records."""

    def __init__(self, record: FrameRecord, issue_count: int) -> None:
        self.record = record
        self.issue_count = issue_count
        super().__init__(
            f"Data audit found {issue_count} invalid record(s); first failing record "
            f"is {record.split}/{record.pig_id}/{record.frame_id}"
        )


class _ImageEncodingError(ValueError):
    """Internal error carrying the exact file and audit reason."""

    def __init__(self, path: Path, reason: str, message: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{message}: {path}")


def _mapping_value(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing {key!r} in {context}")
    return mapping[key]


def _resolve_under_root(dataset_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (dataset_root / candidate).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Path escapes dataset_root: {raw_path}") from error
    return resolved


def load_dataset_index(split_json: Path | str) -> DatasetIndex:
    """Parse a pig-level split manifest into immutable per-frame records."""
    split_path = Path(split_json).resolve()
    try:
        document = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read split JSON: {split_path}") from error
    if not isinstance(document, dict):
        raise ValueError("Split JSON root must be a mapping")

    raw_root = _mapping_value(document, "dataset_root", "split JSON")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError("dataset_root must be a non-empty string")
    dataset_root = Path(raw_root).expanduser().resolve()

    splits = _mapping_value(document, "splits", "split JSON")
    if not isinstance(splits, dict) or set(splits) not in (_REQUIRED_SPLITS, _TRAIN_TEST_SPLITS):
        raise ValueError(
            f"split keys must be exactly {sorted(_TRAIN_TEST_SPLITS)} "
            f"or legacy {sorted(_REQUIRED_SPLITS)}"
        )
    if "test" in splits:
        # Retain the training API/checkpoint schema: same-scene test is the
        # held-out split historically called validation.
        splits = {"train": splits["train"], "validation": splits["test"],
                  "cross_scene_test": splits["cross_scene_test"]}

    by_split: dict[str, tuple[FrameRecord, ...]] = {}
    for split_name, pigs in splits.items():
        if not isinstance(pigs, list):
            raise ValueError(f"Split {split_name!r} must be a list")
        records: list[FrameRecord] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pig in pigs:
            if not isinstance(pig, dict):
                raise ValueError(f"Split {split_name!r} contains a non-mapping pig")
            pig_context = f"pig in split {split_name!r}"
            pig_id = _mapping_value(pig, "pig_id", pig_context)
            scene = _mapping_value(pig, "scene", pig_context)
            relative_dir = _mapping_value(pig, "relative_dir", pig_context)
            weight_kg = _mapping_value(pig, "weight_kg", pig_context)
            frames = _mapping_value(pig, "frames", pig_context)
            if not all(isinstance(value, str) and value for value in (pig_id, scene, relative_dir)):
                raise ValueError(f"{pig_context} identifiers must be non-empty strings")
            if isinstance(weight_kg, bool) or not isinstance(weight_kg, (int, float)):
                raise ValueError(f"{pig_context} weight_kg must be a finite positive number")
            weight = float(weight_kg)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{pig_context} weight_kg must be finite and positive")
            if not isinstance(frames, list):
                raise ValueError(f"{pig_context} frames must be a list")

            pig_directory = _resolve_under_root(dataset_root, relative_dir)
            for frame in frames:
                if not isinstance(frame, dict):
                    raise ValueError(f"{pig_context} contains a non-mapping frame")
                frame_id = _mapping_value(frame, "frame_id", pig_context)
                rgb = _mapping_value(frame, "rgb", pig_context)
                depth = _mapping_value(frame, "depth", pig_context)
                if not all(isinstance(value, str) and value for value in (frame_id, rgb, depth)):
                    raise ValueError(f"{pig_context} frame fields must be non-empty strings")
                pair = (pig_id, frame_id)
                if pair in seen_pairs:
                    raise ValueError(f"duplicate (pig_id, frame_id) in split {split_name!r}: {pair}")
                seen_pairs.add(pair)
                records.append(
                    FrameRecord(
                        split=split_name,
                        scene=scene,
                        pig_id=pig_id,
                        frame_id=frame_id,
                        weight_kg=weight,
                        rgb_path=_resolve_under_root(dataset_root, rgb),
                        depth_path=_resolve_under_root(dataset_root, depth),
                        depth_seg_path=_resolve_under_root(
                            dataset_root, pig_directory.relative_to(dataset_root) / "DEPTH-SEG" / f"{frame_id}.png"
                        ),
                        mask_path=_resolve_under_root(
                            dataset_root, pig_directory.relative_to(dataset_root) / "MASK" / f"{frame_id}_color_mask.png"
                        ),
                    )
                )
        by_split[split_name] = tuple(records)
    return DatasetIndex(dataset_root=dataset_root, by_split=by_split)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise _ImageEncodingError(
                path,
                "invalid_rgb_encoding",
                "RGB image must use native three-channel RGB encoding",
            )
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise _ImageEncodingError(
                path,
                "invalid_rgb_encoding",
                "RGB image must use native three-channel RGB encoding",
            )
        return array


def _load_depth(path: Path, reason: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
        if (
            image.format not in {"PNG", "TIFF"}
            or array.ndim != 2
            or array.dtype.kind != "u"
            or array.dtype.itemsize != 2
        ):
            raise _ImageEncodingError(
                path,
                reason,
                "depth image must use lossless 16-bit single-channel encoding",
            )
        return array


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def load_rgbd_sample(record: FrameRecord) -> RGBDSample:
    """Load one record without resizing or normalizing its native arrays."""
    return RGBDSample(
        rgb=_load_rgb(record.rgb_path),
        depth=_load_depth(record.depth_path, "invalid_depth_encoding"),
        depth_seg=_load_depth(
            record.depth_seg_path, "invalid_depth_seg_encoding"
        ),
        mask=_load_mask(record.mask_path),
        weight_kg=record.weight_kg,
        split=record.split,
        scene=record.scene,
        pig_id=record.pig_id,
        frame_id=record.frame_id,
    )


def _audit_issue(record: FrameRecord) -> dict[str, str] | None:
    paths = (
        (record.rgb_path, "missing_rgb"),
        (record.depth_path, "missing_depth"),
        (record.depth_seg_path, "missing_depth_seg"),
        (record.mask_path, "missing_mask"),
    )
    for path, reason in paths:
        if not path.is_file():
            return {"path": str(path), "reason": reason}
    try:
        sample = load_rgbd_sample(record)
    except _ImageEncodingError as error:
        return {"path": str(error.path), "reason": error.reason}
    except (OSError, ValueError) as error:
        return {"path": str(record.rgb_path), "reason": f"unreadable_image: {error}"}
    if sample.rgb.ndim != 3 or sample.rgb.shape[2] != 3:
        return {"path": str(record.rgb_path), "reason": "invalid_rgb_channels"}
    if sample.depth.ndim != 2:
        return {"path": str(record.depth_path), "reason": "invalid_depth_shape"}
    if sample.depth_seg.ndim != 2:
        return {"path": str(record.depth_seg_path), "reason": "invalid_depth_seg_shape"}
    if sample.mask.ndim != 2:
        return {"path": str(record.mask_path), "reason": "invalid_mask_shape"}
    if not (sample.rgb.shape[:2] == sample.depth.shape == sample.depth_seg.shape == sample.mask.shape):
        return {"path": str(record.rgb_path), "reason": "shape_mismatch"}
    return None


def audit_records(
    records: Sequence[FrameRecord], invalid_sample_policy: str = "skip"
) -> tuple[tuple[FrameRecord, ...], dict[str, Any]]:
    """Validate records and either skip invalid frames or raise on the first issue."""
    if invalid_sample_policy not in {"skip", "error"}:
        raise ValueError("invalid_sample_policy must be 'skip' or 'error'")
    valid_records: list[FrameRecord] = []
    failures: list[tuple[FrameRecord, dict[str, str]]] = []
    for record in records:
        issue = _audit_issue(record)
        if issue is None:
            valid_records.append(record)
        else:
            failures.append((record, issue))
    if failures and invalid_sample_policy == "error":
        raise DataAuditError(failures[0][0], len(failures))
    report = {
        "input_count": len(records),
        "valid_count": len(valid_records),
        "skipped_count": len(failures),
        "issues": [issue for _, issue in failures],
    }
    return tuple(valid_records), report


class RGBDFrameDataset(torch.utils.data.Dataset[RGBDSample | dict[str, Any]]):
    """PyTorch dataset that can optionally return normalized, collatable batches."""

    def __init__(
        self,
        records: Sequence[FrameRecord],
        preprocess_config: PreprocessConfig | None = None,
        background_records: Sequence[FrameRecord] | None = None,
        augmentation_probability: float = 0.0,
        augmentation_seed: int = 0,
    ) -> None:
        self.records = tuple(records)
        self.preprocess_config = preprocess_config
        if (
            isinstance(augmentation_probability, bool)
            or not isinstance(augmentation_probability, (int, float))
            or not math.isfinite(augmentation_probability)
            or not 0.0 <= augmentation_probability <= 1.0
        ):
            raise ValueError("augmentation_probability must be between 0 and 1")
        if isinstance(augmentation_seed, bool) or not isinstance(augmentation_seed, (int, np.integer)):
            raise ValueError("augmentation_seed must be an integer")
        self.background_records = tuple(background_records or ())
        self.augmentation_probability = float(augmentation_probability)
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0
        self._augmenter = OverlayAugmenter()
        if self.augmentation_probability:
            for record in self.records:
                if not any(target.pig_id != record.pig_id for target in self.background_records):
                    raise ValueError("augmentation requires an eligible target from a different pig")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch component used by the deterministic per-index RNG."""
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        seed_sequence = np.random.SeedSequence([self.augmentation_seed, self.epoch, index])
        return np.random.default_rng(seed_sequence)

    def __getitem__(self, index: int) -> RGBDSample | dict[str, Any]:
        sample = load_rgbd_sample(self.records[index])
        rng = self._rng_for_index(index)
        if self.augmentation_probability and rng.random() < self.augmentation_probability:
            eligible_targets = tuple(
                record for record in self.background_records if record.pig_id != sample.pig_id
            )
            target_index = int(rng.integers(len(eligible_targets)))
            sample = self._augmenter(sample, load_rgbd_sample(eligible_targets[target_index]))
        if self.preprocess_config is None:
            return sample
        return normalize_sample(sample, self.preprocess_config)
