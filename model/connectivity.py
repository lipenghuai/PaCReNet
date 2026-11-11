"""Connected components and optional decoding on the row-major patch grid."""

from functools import lru_cache
from numbers import Integral

import numpy as np
import torch


def validate_grid_size(grid_size: tuple[int, int]) -> tuple[int, int]:
    if len(grid_size) != 2 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in grid_size
    ):
        raise ValueError("grid_size must contain two positive integers")
    return int(grid_size[0]), int(grid_size[1])


def validate_connectivity(connectivity: int) -> None:
    if isinstance(connectivity, bool) or not isinstance(connectivity, Integral) or connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")


@lru_cache(maxsize=16, typed=True)
def grid_neighbors(
    grid_size: tuple[int, int], connectivity: int = 8
) -> tuple[tuple[int, ...], ...]:
    """Cache bounded, non-wrapping neighbors for component and topology routines."""
    height, width = validate_grid_size(grid_size)
    validate_connectivity(connectivity)
    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return tuple(
        tuple(
            (row + dr) * width + col + dc
            for dr, dc in offsets
            if 0 <= row + dr < height and 0 <= col + dc < width
        )
        for row in range(height)
        for col in range(width)
    )


def component_indices(mask: np.ndarray, connectivity: int = 8) -> list[list[int]]:
    """Return components as flat indices, seeded in deterministic row-major order."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("component mask must be a two-dimensional grid")
    neighbors = grid_neighbors(tuple(mask.shape), connectivity)
    active = mask.reshape(-1)
    visited = np.zeros(active.size, dtype=bool)
    components = []
    for seed in np.flatnonzero(active):
        if visited[seed]:
            continue
        visited[seed] = True
        pending = [int(seed)]
        component = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if active[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
        components.append(component)
    return components


def component_counts(
    classes: torch.Tensor, grid_size: tuple[int, int], connectivity: int = 8
) -> torch.Tensor:
    """Return CPU counts [B, 2] for background (0) and full pig (1)."""
    height, width = validate_grid_size(grid_size)
    validate_connectivity(connectivity)
    if classes.ndim != 2 or classes.shape[1] != height * width:
        raise ValueError("classes must have shape [B, grid_height * grid_width]")
    grids = classes.detach().cpu().numpy().reshape(-1, height, width)
    counts = [
        [len(component_indices(grid == class_id, connectivity)) for class_id in (0, 1)]
        for grid in grids
    ]
    return torch.tensor(counts, dtype=torch.int64).reshape(-1, 2)


def decode_patch_classes(
    logits: torch.Tensor,
    grid_size: tuple[int, int],
    *,
    enforce_connectivity: bool = False,
    connectivity: int = 8,
) -> torch.Tensor:
    """Optionally keep one body component and one background component per frame.

    Body fragments become background; isolated background regions become edge.
    If background was absent, discarded body fragments become edge instead.
    Empty classes remain empty. Edge connectivity is deliberately unconstrained.
    """
    height, width = validate_grid_size(grid_size)
    validate_connectivity(connectivity)
    if logits.ndim != 3 or logits.shape[1:] != (height * width, 3):
        raise ValueError("logits must have shape [B, grid_height * grid_width, 3]")
    if not isinstance(enforce_connectivity, bool):
        raise ValueError("enforce_connectivity must be a boolean")
    classes = logits.argmax(dim=-1)
    if not enforce_connectivity:
        return classes
    grids = classes.detach().cpu().numpy().copy().reshape(-1, height, width)
    for grid in grids:
        body_replacement = 0 if np.any(grid == 0) else 2
        for class_id, replacement in ((1, body_replacement), (0, 2)):
            components = component_indices(grid == class_id, connectivity)
            if len(components) <= 1:
                continue
            largest = max(range(len(components)), key=lambda index: len(components[index]))
            flat = grid.reshape(-1)
            for index, component in enumerate(components):
                if index != largest:
                    flat[component] = replacement
    return torch.from_numpy(grids.reshape(classes.shape)).to(device=logits.device)
