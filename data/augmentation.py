"""In-memory, coordinate-preserving RGB-D augmentation for stage one."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataset import RGBDSample


def _has_matching_grid(sample: RGBDSample) -> bool:
    """Return whether every spatial modality shares the sample's RGB grid."""
    return sample.rgb.shape[:2] == sample.depth.shape == sample.depth_seg.shape == sample.mask.shape


class OverlayAugmenter:
    """Overlay a source pig onto a different pig's RGB-D background."""

    def __call__(self, source: RGBDSample, target: RGBDSample) -> RGBDSample:
        """Return a new source-authoritative sample composited at identical pixels."""
        if source.pig_id == target.pig_id:
            raise ValueError("source and target must be different pigs")
        if not _has_matching_grid(source) or not _has_matching_grid(target):
            raise ValueError("source and target sample grids must match internally")
        if source.rgb.shape[:2] != target.rgb.shape[:2]:
            raise ValueError("source and target dimensions must match")

        from .dataset import RGBDSample

        rgb = np.where(source.mask[..., None], source.rgb, target.rgb).astype(np.uint8, copy=False)
        depth = np.where(source.depth_seg > 0, source.depth_seg, target.depth).astype(np.uint16, copy=False)
        metadata: dict[str, bool | str] = {
            "augmented": True,
            "source_split": str(source.split),
            "source_pig_id": str(source.pig_id),
            "source_frame_id": str(source.frame_id),
            "target_split": str(target.split),
            "target_pig_id": str(target.pig_id),
            "target_frame_id": str(target.frame_id),
        }
        return RGBDSample(
            rgb=rgb,
            depth=depth,
            depth_seg=source.depth_seg.copy(),
            mask=source.mask.copy(),
            weight_kg=source.weight_kg,
            split=source.split,
            scene=source.scene,
            pig_id=source.pig_id,
            frame_id=source.frame_id,
            augmentation_metadata=metadata,
        )
