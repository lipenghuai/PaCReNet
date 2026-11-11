"""Deterministic pig-level split construction and balanced frame sampling."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from .dataset import DatasetIndex, FrameRecord


@dataclass(frozen=True, slots=True)
class Stage1Splits:
    """Records used for stage-1 training and evaluation."""

    train_records: tuple[FrameRecord, ...]
    validation_records: tuple[FrameRecord, ...]
    cross_scene_records: tuple[FrameRecord, ...]
    selected_cross_pig_ids: tuple[str, ...]


def _validate_original_split_isolation(index: DatasetIndex) -> None:
    pig_splits: dict[str, str] = {}
    for split_name, records in index.by_split.items():
        for record in records:
            prior_split = pig_splits.setdefault(record.pig_id, split_name)
            if prior_split != split_name:
                raise ValueError(
                    f"pig_id {record.pig_id!r} appears in multiple original splits: "
                    f"{prior_split!r} and {split_name!r}"
                )


def build_stage1_splits(index: DatasetIndex, fraction: float, seed: int) -> Stage1Splits:
    """Move a deterministic pig-level fraction of cross-scene records into training."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be within [0, 1]")
    _validate_original_split_isolation(index)

    cross = index.by_split["cross_scene_test"]
    pig_ids = sorted({record.pig_id for record in cross})
    count = math.floor(len(pig_ids) * fraction + 0.5)
    selected = tuple(sorted(random.Random(seed).sample(pig_ids, count)))
    selected_set = set(selected)
    added = tuple(record for record in cross if record.pig_id in selected_set)
    held_out = tuple(record for record in cross if record.pig_id not in selected_set)
    return Stage1Splits(
        train_records=index.by_split["train"] + added,
        validation_records=index.by_split["validation"],
        cross_scene_records=held_out,
        selected_cross_pig_ids=selected,
    )


class PigBalancedSampler(torch.utils.data.Sampler[int]):
    """Sample pigs uniformly, then sample one of each chosen pig's frames uniformly."""

    def __init__(self, records: Sequence[FrameRecord], num_samples: int, seed: int) -> None:
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        indices_by_pig: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            indices_by_pig[record.pig_id].append(index)
        if num_samples and not indices_by_pig:
            raise ValueError("records must not be empty when num_samples is positive")

        self._indices_by_pig = dict(indices_by_pig)
        self._pig_ids = tuple(sorted(self._indices_by_pig))
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        for _ in range(self.num_samples):
            pig_id = rng.choice(self._pig_ids)
            yield rng.choice(self._indices_by_pig[pig_id])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
