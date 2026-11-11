"""Display-only utility for inspecting one exact RGB-D overlay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .augmentation import OverlayAugmenter
from .dataset import RGBDSample, load_dataset_index, load_rgbd_sample


def build_demo_figure(source: RGBDSample, target: RGBDSample, result: RGBDSample):
    """Build, but never persist, a five-panel overlay inspection figure."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    panels = (
        (source.rgb, "Source RGB"),
        (target.rgb, "Target RGB"),
        (result.rgb, "Result RGB"),
        (result.depth, "Result depth"),
        (result.mask, "Result MASK"),
    )
    for axis, (image, title) in zip(axes, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    return figure


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Load two records, print overlay details, and optionally display the figure."""
    arguments = _parse_args(argv)
    records = load_dataset_index(arguments.split_json).all_records()
    source = load_rgbd_sample(records[arguments.source_index])
    target = load_rgbd_sample(records[arguments.target_index])
    result = OverlayAugmenter()(source, target)
    build_demo_figure(source, target, result)
    print(
        json.dumps(
            {
                "source_rgb_shape": list(source.rgb.shape),
                "target_rgb_shape": list(target.rgb.shape),
                "result_rgb_shape": list(result.rgb.shape),
                "result_depth_shape": list(result.depth.shape),
                "result_mask_shape": list(result.mask.shape),
                "augmentation_metadata": result.augmentation_metadata,
            },
            sort_keys=True,
        )
    )
    if arguments.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
