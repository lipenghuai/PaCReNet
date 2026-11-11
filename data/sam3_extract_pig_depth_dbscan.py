#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Use RGB-space MASKs and 3-D clustering to create pig-only depth images.

For every sibling group containing ``RGB``, ``MASK`` and ``DEPTH`` below the
input root, this script matches a mask to its raw depth image, reconstructs the
candidate depth pixels as 3-D points, and applies the same refinement order as
``segplyall.py``:

1. Erode the mask by distance transform to obtain reliable core seed points.
2. Dilate the mask slightly to obtain candidate 3-D points.
3. Run DBSCAN and select the cluster containing the most core seed points.
4. Remove statistical outliers (SOR).
5. Run DBSCAN again and retain the largest remaining cluster.

The selected pixels are written to the sibling ``DEPTH-SEG`` directory.  The
output keeps the source depth image's shape, integer dtype, and raw depth units;
all rejected/background pixels are exactly zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:  # The current environment can use sklearn/scipy instead.
    o3d = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
except ImportError:
    SklearnDBSCAN = None


DEFAULT_INPUT_ROOT = Path("datasets/pig_rgbd")
DEFAULT_CAMERA_JSON = Path("datasets/camera_calibration.json")
DEFAULT_MANIFEST = Path("runs/preprocessing/depth_seg_dbscan_manifest.csv")
MASK_SUFFIXES = {".png", ".tif", ".tiff", ".bmp"}
DEPTH_SUFFIXES = {".png", ".tif", ".tiff"}
MANIFEST_FIELDS = (
    "source_mask",
    "source_depth",
    "output_depth_seg",
    "backend",
    "candidate_points",
    "core_seed_points",
    "first_clusters",
    "selected_seed_points",
    "after_seed_cluster",
    "after_sor",
    "second_clusters",
    "output_valid_pixels",
    "retained_candidate_fraction",
    "seconds",
    "status",
    "error",
)


@dataclass(frozen=True)
class CameraCalibration:
    depth_width: int
    depth_height: int
    depth_fx: float
    depth_fy: float
    depth_cx: float
    depth_cy: float
    color_width: int
    color_height: int
    color_fx: float
    color_fy: float
    color_cx: float
    color_cy: float
    depth_scale: float
    rotation_depth_to_color: np.ndarray
    translation_depth_to_color: np.ndarray


@dataclass(frozen=True)
class MaskDepthPair:
    session_dir: Path
    mask_dir: Path
    depth_dir: Path
    mask_path: Path
    depth_path: Path


@dataclass(frozen=True)
class ProjectedCandidates:
    points_xyz_m: np.ndarray
    depth_y: np.ndarray
    depth_x: np.ndarray
    seed_flags: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pig depth using MASK-core seeds, 3-D DBSCAN, statistical "
            "outlier removal, and a final largest-cluster pass."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--camera-json", type=Path, default=DEFAULT_CAMERA_JSON)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mask-dir-name", default="MASK")
    parser.add_argument("--depth-dir-name", default="DEPTH")
    parser.add_argument("--output-dir-name", default="DEPTH-SEG")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional alternate output root for testing; default saves beside RGB/MASK/DEPTH.",
    )
    parser.add_argument(
        "--core-distance-px",
        type=float,
        default=4.0,
        help="Minimum RGB-mask distance from its edge for a core seed pixel.",
    )
    parser.add_argument(
        "--candidate-dilate-px",
        type=int,
        default=4,
        help="RGB-mask dilation used only to collect candidate 3-D points.",
    )
    parser.add_argument("--dbscan-eps-m", type=float, default=0.06)
    parser.add_argument("--dbscan-min-points", type=int, default=200)
    parser.add_argument("--sor-neighbors", type=int, default=30)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    parser.add_argument(
        "--backend",
        choices=("auto", "open3d", "sklearn"),
        default="auto",
        help="auto prefers Open3D and otherwise uses the installed sklearn/scipy backend.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process only this many sorted pairs; 0 processes all pairs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute an image even when a same-shape/same-dtype output already exists.",
    )
    args = parser.parse_args()
    if args.core_distance_px < 0:
        parser.error("--core-distance-px cannot be negative")
    if args.candidate_dilate_px < 0:
        parser.error("--candidate-dilate-px cannot be negative")
    if args.dbscan_eps_m <= 0:
        parser.error("--dbscan-eps-m must be positive")
    if args.dbscan_min_points < 2:
        parser.error("--dbscan-min-points must be at least 2")
    if args.sor_neighbors < 1:
        parser.error("--sor-neighbors must be positive")
    if args.sor_std_ratio <= 0:
        parser.error("--sor-std-ratio must be positive")
    if args.max_images < 0:
        parser.error("--max-images cannot be negative")
    return args


def choose_backend(requested: str) -> str:
    if requested == "open3d":
        if o3d is None:
            raise ImportError("Open3D backend requested, but open3d is not installed")
        return "open3d"
    if requested == "sklearn":
        if SklearnDBSCAN is None or cKDTree is None:
            raise ImportError("sklearn backend requires scikit-learn and scipy")
        return "sklearn"
    if o3d is not None:
        return "open3d"
    if SklearnDBSCAN is not None and cKDTree is not None:
        return "sklearn"
    raise ImportError(
        "No supported 3-D backend is available. Install open3d, or install "
        "scikit-learn together with scipy."
    )


def load_camera_calibration(path: Path) -> CameraCalibration:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    streams = config.get("chosen_streams", {})
    depth = streams.get("depth_intrinsics") or streams.get("depth_profile", {}).get("intrinsics")
    color = streams.get("color_intrinsics") or streams.get("color_profile", {}).get("intrinsics")
    extrinsic = config.get("extrinsics", {}).get("depth_to_color")
    if depth is None or color is None or extrinsic is None:
        raise KeyError("Camera JSON needs depth/color intrinsics and depth_to_color extrinsics")
    rotation = np.asarray(extrinsic["rotation_row_major_3x3"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(extrinsic["translation_m"], dtype=np.float64).reshape(3)
    depth_scale = float(config.get("depth_scale_m_per_unit", 0.001))
    if depth_scale <= 0:
        raise ValueError(f"Invalid depth scale: {depth_scale}")
    return CameraCalibration(
        depth_width=int(depth["width"]),
        depth_height=int(depth["height"]),
        depth_fx=float(depth["fx"]),
        depth_fy=float(depth["fy"]),
        depth_cx=float(depth["ppx"]),
        depth_cy=float(depth["ppy"]),
        color_width=int(color["width"]),
        color_height=int(color["height"]),
        color_fx=float(color["fx"]),
        color_fy=float(color["fy"]),
        color_cx=float(color["ppx"]),
        color_cy=float(color["ppy"]),
        depth_scale=depth_scale,
        rotation_depth_to_color=rotation,
        translation_depth_to_color=translation,
    )


def named_child(parent: Path, name: str) -> Path | None:
    wanted = name.casefold()
    return next(
        (child for child in parent.iterdir() if child.is_dir() and child.name.casefold() == wanted),
        None,
    )


def frame_key(path: Path) -> str:
    stem = path.stem.casefold()
    for suffix in ("_color_mask", "_mask", "_color"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_pairs(root: Path, mask_name: str, depth_name: str) -> tuple[list[MaskDepthPair], list[Path]]:
    mask_dirs = sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.casefold() == mask_name.casefold()
    )
    pairs: list[MaskDepthPair] = []
    unmatched: list[Path] = []
    for mask_dir in mask_dirs:
        session_dir = mask_dir.parent
        depth_dir = named_child(session_dir, depth_name)
        masks = sorted(
            path
            for path in mask_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in MASK_SUFFIXES
        )
        if depth_dir is None:
            unmatched.extend(masks)
            continue
        depth_index: dict[str, list[Path]] = {}
        for depth_path in sorted(depth_dir.rglob("*")):
            if depth_path.is_file() and depth_path.suffix.casefold() in DEPTH_SUFFIXES:
                depth_index.setdefault(frame_key(depth_path), []).append(depth_path)
        for mask_path in masks:
            matches = depth_index.get(frame_key(mask_path), [])
            if len(matches) == 1:
                pairs.append(
                    MaskDepthPair(session_dir, mask_dir, depth_dir, mask_path, matches[0])
                )
            else:
                unmatched.append(mask_path)
    return pairs, unmatched


def read_image(path: Path, flags: int) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def read_binary_mask(path: Path) -> np.ndarray:
    mask = read_image(path, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2-D image: {path}, shape={mask.shape}")
    binary = (mask > 127).astype(np.uint8)
    if not np.any(binary):
        raise ValueError(f"Mask contains no foreground: {path}")
    return binary


def read_raw_depth(path: Path) -> np.ndarray:
    depth = read_image(path, cv2.IMREAD_UNCHANGED)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise ValueError(f"Depth must be a 2-D integer image: {path}, {depth.shape}/{depth.dtype}")
    return depth


def mask_core(mask: np.ndarray, distance_px: float) -> np.ndarray:
    if distance_px <= 0:
        return mask.astype(bool)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    core = distance >= distance_px
    return core if np.any(core) else mask.astype(bool)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def make_projected_candidates(
    depth: np.ndarray,
    color_mask: np.ndarray,
    calibration: CameraCalibration,
    core_distance_px: float,
    candidate_dilate_px: int,
) -> ProjectedCandidates:
    """Back-project depth pixels and select those landing in the RGB candidate mask."""
    mask_h, mask_w = color_mask.shape
    depth_h, depth_w = depth.shape
    valid_y, valid_x = np.nonzero(depth > 0)
    if valid_x.size == 0:
        raise RuntimeError("Depth image contains no valid pixels")

    scale_x = depth_w / calibration.depth_width
    scale_y = depth_h / calibration.depth_height
    fx_d = calibration.depth_fx * scale_x
    fy_d = calibration.depth_fy * scale_y
    cx_d = (calibration.depth_cx + 0.5) * scale_x - 0.5
    cy_d = (calibration.depth_cy + 0.5) * scale_y - 0.5

    z_d = depth[valid_y, valid_x].astype(np.float64) * calibration.depth_scale
    x_d = (valid_x.astype(np.float64) - cx_d) * z_d / fx_d
    y_d = (valid_y.astype(np.float64) - cy_d) * z_d / fy_d
    points_depth = np.column_stack((x_d, y_d, z_d))

    points_color = (
        points_depth @ calibration.rotation_depth_to_color.T
        + calibration.translation_depth_to_color
    )
    z_c = points_color[:, 2]
    front = z_c > 1e-8
    if not np.any(front):
        raise RuntimeError("No valid depth points are in front of the color camera")

    points_depth = points_depth[front]
    points_color = points_color[front]
    valid_y = valid_y[front]
    valid_x = valid_x[front]
    z_c = points_color[:, 2]
    color_u = calibration.color_fx * points_color[:, 0] / z_c + calibration.color_cx
    color_v = calibration.color_fy * points_color[:, 1] / z_c + calibration.color_cy
    mask_u = np.rint((color_u + 0.5) * mask_w / calibration.color_width - 0.5).astype(np.int32)
    mask_v = np.rint((color_v + 0.5) * mask_h / calibration.color_height - 0.5).astype(np.int32)
    inside = (mask_u >= 0) & (mask_u < mask_w) & (mask_v >= 0) & (mask_v < mask_h)
    if not np.any(inside):
        raise RuntimeError("No depth points project inside the MASK image")

    points_depth = points_depth[inside]
    valid_y = valid_y[inside]
    valid_x = valid_x[inside]
    mask_u = mask_u[inside]
    mask_v = mask_v[inside]

    core = mask_core(color_mask, core_distance_px)
    candidate_mask = dilate_mask(color_mask, candidate_dilate_px)
    candidate_flags = candidate_mask[mask_v, mask_u]
    if not np.any(candidate_flags):
        raise RuntimeError("MASK projection produced zero candidate 3-D points")

    seed_flags = core[mask_v[candidate_flags], mask_u[candidate_flags]]
    if not np.any(seed_flags):
        # A very small or poorly aligned mask still gets seeds from its original area.
        seed_flags = color_mask[mask_v[candidate_flags], mask_u[candidate_flags]].astype(bool)
    if not np.any(seed_flags):
        raise RuntimeError("MASK projection produced zero core seed points")
    return ProjectedCandidates(
        points_xyz_m=np.ascontiguousarray(points_depth[candidate_flags], dtype=np.float64),
        depth_y=valid_y[candidate_flags].astype(np.int32, copy=False),
        depth_x=valid_x[candidate_flags].astype(np.int32, copy=False),
        seed_flags=np.asarray(seed_flags, dtype=bool),
    )


def dbscan_labels(points: np.ndarray, eps_m: float, min_points: int, backend: str) -> np.ndarray:
    if len(points) < min_points:
        raise RuntimeError(f"Only {len(points)} points are available; DBSCAN needs {min_points}")
    if backend == "open3d":
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return np.asarray(
            cloud.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False),
            dtype=np.int32,
        )
    model = SklearnDBSCAN(
        eps=eps_m,
        min_samples=min_points,
        metric="euclidean",
        algorithm="kd_tree",
        leaf_size=40,
        n_jobs=-1,
    )
    return model.fit_predict(points).astype(np.int32, copy=False)


def count_clusters(labels: np.ndarray) -> int:
    valid = labels[labels >= 0]
    return int(np.unique(valid).size) if valid.size else 0


def select_seed_cluster(labels: np.ndarray, seed_flags: np.ndarray) -> tuple[np.ndarray, int, int]:
    valid_labels = np.unique(labels[labels >= 0])
    if valid_labels.size == 0:
        raise RuntimeError("First DBSCAN found no valid cluster")
    best_label = -1
    best_seed_count = -1
    best_size = -1
    for label in valid_labels:
        members = labels == label
        seed_count = int(np.count_nonzero(seed_flags & members))
        size = int(np.count_nonzero(members))
        if (seed_count, size) > (best_seed_count, best_size):
            best_label = int(label)
            best_seed_count = seed_count
            best_size = size
    if best_seed_count == 0:
        # No non-noise cluster contains a core seed: the largest valid cluster is safest.
        sizes = [(int(np.count_nonzero(labels == label)), int(label)) for label in valid_labels]
        best_size, best_label = max(sizes)
    return labels == best_label, max(0, best_seed_count), best_label


def statistical_outlier_mask(
    points: np.ndarray,
    neighbors: int,
    std_ratio: float,
    backend: str,
) -> np.ndarray:
    if len(points) <= 2:
        return np.ones(len(points), dtype=bool)
    k = min(neighbors, len(points) - 1)
    if backend == "open3d":
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        _, inliers = cloud.remove_statistical_outlier(
            nb_neighbors=k,
            std_ratio=std_ratio,
        )
        keep = np.zeros(len(points), dtype=bool)
        keep[np.asarray(inliers, dtype=np.int64)] = True
        return keep

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1, workers=-1)
    mean_neighbor_distance = distances[:, 1:].mean(axis=1)
    threshold = (
        float(mean_neighbor_distance.mean())
        + std_ratio * float(mean_neighbor_distance.std(ddof=0))
    )
    return mean_neighbor_distance <= threshold


def largest_cluster_mask(labels: np.ndarray) -> np.ndarray:
    valid = labels[labels >= 0]
    if valid.size == 0:
        raise RuntimeError("Second DBSCAN found no valid cluster")
    cluster_ids, counts = np.unique(valid, return_counts=True)
    largest_label = int(cluster_ids[int(np.argmax(counts))])
    return labels == largest_label


def refine_depth(
    depth: np.ndarray,
    color_mask: np.ndarray,
    calibration: CameraCalibration,
    core_distance_px: float,
    candidate_dilate_px: int,
    dbscan_eps_m: float,
    dbscan_min_points: int,
    sor_neighbors: int,
    sor_std_ratio: float,
    backend: str,
) -> tuple[np.ndarray, dict[str, int | float]]:
    candidates = make_projected_candidates(
        depth,
        color_mask,
        calibration,
        core_distance_px,
        candidate_dilate_px,
    )
    labels1 = dbscan_labels(
        candidates.points_xyz_m,
        dbscan_eps_m,
        dbscan_min_points,
        backend,
    )
    chosen1, chosen_seed_count, _ = select_seed_cluster(labels1, candidates.seed_flags)
    points1 = candidates.points_xyz_m[chosen1]
    y1 = candidates.depth_y[chosen1]
    x1 = candidates.depth_x[chosen1]

    keep_sor = statistical_outlier_mask(points1, sor_neighbors, sor_std_ratio, backend)
    points2 = points1[keep_sor]
    y2 = y1[keep_sor]
    x2 = x1[keep_sor]
    if len(points2) < 2:
        raise RuntimeError("Statistical outlier removal left too few points")

    second_min_points = min(
        len(points2),
        max(20, dbscan_min_points // 2),
    )
    labels2 = dbscan_labels(points2, dbscan_eps_m, second_min_points, backend)
    chosen2 = largest_cluster_mask(labels2)
    final_y = y2[chosen2]
    final_x = x2[chosen2]
    if final_x.size == 0:
        raise RuntimeError("Final largest-cluster selection retained zero points")

    output = np.zeros_like(depth)
    output[final_y, final_x] = depth[final_y, final_x]
    stats: dict[str, int | float] = {
        "candidate_points": len(candidates.points_xyz_m),
        "core_seed_points": int(np.count_nonzero(candidates.seed_flags)),
        "first_clusters": count_clusters(labels1),
        "selected_seed_points": chosen_seed_count,
        "after_seed_cluster": len(points1),
        "after_sor": len(points2),
        "second_clusters": count_clusters(labels2),
        "output_valid_pixels": len(final_x),
        "retained_candidate_fraction": round(len(final_x) / len(candidates.points_xyz_m), 6),
    }
    return output, stats


def output_path_for_pair(
    pair: MaskDepthPair,
    input_root: Path,
    output_dir_name: str,
    output_root: Path | None,
) -> Path:
    relative_depth = pair.depth_path.relative_to(pair.depth_dir).with_suffix(".png")
    if output_root is None:
        base = pair.session_dir / output_dir_name
    else:
        base = output_root / pair.session_dir.relative_to(input_root) / output_dir_name
    return base / relative_depth


def atomic_write_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        ok, encoded = cv2.imencode(".png", depth)
        if not ok:
            raise RuntimeError(f"Cannot encode depth PNG: {path}")
        encoded.tofile(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def existing_output_is_valid(path: Path, source: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        output = read_raw_depth(path)
    except Exception:
        return False
    return output.shape == source.shape and output.dtype == source.dtype


def open_manifest(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
    if needs_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def manifest_row(pair: MaskDepthPair, output: Path, backend: str) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "source_mask": str(pair.mask_path),
            "source_depth": str(pair.depth_path),
            "output_depth_seg": str(output),
            "backend": backend,
        }
    )
    return row


def validate_unique_outputs(
    pairs: list[MaskDepthPair],
    input_root: Path,
    output_dir_name: str,
    output_root: Path | None,
) -> None:
    seen: set[Path] = set()
    for pair in pairs:
        output = output_path_for_pair(pair, input_root, output_dir_name, output_root)
        if output in seen:
            raise RuntimeError(f"Output path collision: {output}")
        seen.add(output)


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.camera_json = args.camera_json.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.expanduser().resolve()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if not args.camera_json.is_file():
        raise FileNotFoundError(f"Camera JSON does not exist: {args.camera_json}")

    backend = choose_backend(args.backend)
    calibration = load_camera_calibration(args.camera_json)
    pairs, unmatched = find_pairs(args.input_root, args.mask_dir_name, args.depth_dir_name)
    if args.max_images:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise RuntimeError(f"No matching MASK/DEPTH pairs found below {args.input_root}")
    validate_unique_outputs(pairs, args.input_root, args.output_dir_name, args.output_root)

    manifest_handle, writer = open_manifest(args.manifest)
    ok_count = skipped_count = error_count = 0
    print(f"[INFO] input root       : {args.input_root}")
    print(f"[INFO] camera JSON     : {args.camera_json}")
    print(f"[INFO] paired images   : {len(pairs)}")
    print(f"[INFO] unmatched masks : {len(unmatched)}")
    print(f"[INFO] 3-D backend     : {backend}")
    print(
        f"[INFO] output           : {args.output_root} (mirrored test/export root)"
        if args.output_root is not None
        else f"[INFO] output           : sibling {args.output_dir_name} directories"
    )
    print(f"[INFO] manifest         : {args.manifest}")
    try:
        for index, pair in enumerate(pairs, 1):
            started = time.perf_counter()
            output_path = output_path_for_pair(
                pair,
                args.input_root,
                args.output_dir_name,
                args.output_root,
            )
            row = manifest_row(pair, output_path, backend)
            try:
                depth = read_raw_depth(pair.depth_path)
                if not args.overwrite and existing_output_is_valid(output_path, depth):
                    skipped_count += 1
                    row["output_valid_pixels"] = int(np.count_nonzero(read_raw_depth(output_path)))
                    row["status"] = "skipped_existing_valid"
                else:
                    color_mask = read_binary_mask(pair.mask_path)
                    output, stats = refine_depth(
                        depth=depth,
                        color_mask=color_mask,
                        calibration=calibration,
                        core_distance_px=args.core_distance_px,
                        candidate_dilate_px=args.candidate_dilate_px,
                        dbscan_eps_m=args.dbscan_eps_m,
                        dbscan_min_points=args.dbscan_min_points,
                        sor_neighbors=args.sor_neighbors,
                        sor_std_ratio=args.sor_std_ratio,
                        backend=backend,
                    )
                    if not np.any(output):
                        raise RuntimeError("Refinement produced an empty depth image")
                    atomic_write_depth(output_path, output)
                    if not existing_output_is_valid(output_path, depth):
                        raise RuntimeError("Saved DEPTH-SEG failed shape/dtype validation")
                    # Every non-zero output value must be an unchanged source raw-depth value.
                    selected = output > 0
                    if not np.array_equal(output[selected], depth[selected]):
                        raise RuntimeError("Saved output changed raw depth values")
                    row.update(stats)
                    row["status"] = "ok"
                    ok_count += 1
            except Exception as exc:
                error_count += 1
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = round(time.perf_counter() - started, 4)
            writer.writerow(row)
            manifest_handle.flush()
            print(
                f"[{index:>6}/{len(pairs)}] {row['status']} | "
                f"candidate={row['candidate_points'] or '-'} "
                f"seed={row['core_seed_points'] or '-'} "
                f"final={row['output_valid_pixels'] or '-'} | "
                f"{pair.depth_path} -> {output_path}"
            )
    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted; completed images and manifest rows are preserved.")
    finally:
        manifest_handle.close()

    print(f"[DONE] ok={ok_count}, skipped={skipped_count}, errors={error_count}")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
