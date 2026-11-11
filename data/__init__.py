"""Data loading and preprocessing package."""

from .dataset import (
    DataAuditError,
    DatasetIndex,
    FrameRecord,
    RGBDFrameDataset,
    RGBDSample,
    audit_records,
    load_dataset_index,
    load_rgbd_sample,
)
from .sampling import PigBalancedSampler, Stage1Splits, build_stage1_splits
from .preprocessing import PreprocessConfig, make_patch_labels, normalize_sample, select_modality
from .augmentation import OverlayAugmenter

__all__ = [
    "DataAuditError",
    "DatasetIndex",
    "FrameRecord",
    "RGBDFrameDataset",
    "RGBDSample",
    "audit_records",
    "load_dataset_index",
    "load_rgbd_sample",
    "PigBalancedSampler",
    "Stage1Splits",
    "build_stage1_splits",
    "PreprocessConfig",
    "make_patch_labels",
    "normalize_sample",
    "select_modality",
    "OverlayAugmenter",
]
