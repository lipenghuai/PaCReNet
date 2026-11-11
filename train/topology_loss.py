"""A differentiable zero-dimensional persistent-homology connectivity prior."""

import numpy as np
import torch

from model.connectivity import component_counts, grid_neighbors, validate_grid_size


def _persistence_pairs(
    values: np.ndarray, neighbors: tuple[tuple[int, ...], ...]
) -> tuple[int, list[tuple[int, int]]]:
    """Pair extra component births with merge vertices in a superlevel filtration.

    Union-find and sorting select critical indices using detached values. The
    loss gathers the original torch values at those indices, preserving gradients
    to both a fragment peak and its connecting saddle. Ties use row-major order.
    """
    order = np.argsort(-values, kind="stable")
    parent = [-1] * len(values)
    birth = list(range(len(values)))
    pairs = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for raw_index in order:
        index = int(raw_index)
        roots = {find(neighbor) for neighbor in neighbors[index] if parent[neighbor] >= 0}
        if not roots:
            parent[index] = index
            continue
        elder = min(roots, key=lambda root: (-values[birth[root]], birth[root]))
        parent[index] = elder
        for root in roots:
            if root == elder:
                continue
            if values[birth[root]] > values[index]:
                pairs.append((birth[root], index))
            parent[root] = elder
    return birth[find(int(order[0]))], pairs


def topology_connectivity_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grid_size: tuple[int, int],
    connectivity: int = 8,
) -> torch.Tensor:
    """Encourage one component for each class whose target has one component.

    For each eligible frame/class, strengthen the essential component and shrink
    the squared persistence of extra components. Classes with zero or multiple
    target components contribute no prior; eligibility is independent per class.
    A class-versus-best-alternative sigmoid has its 0.5 boundary at argmax, unlike
    a class softmax threshold in three-class segmentation. This is a soft prior,
    not a guarantee for discrete predictions.
    """
    height, width = validate_grid_size(grid_size)
    neighbors = grid_neighbors((height, width), connectivity)
    if logits.ndim != 3 or logits.shape[1:] != (height * width, 3):
        raise ValueError("topology logits must have shape [B, grid_height * grid_width, 3]")
    if tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("topology labels must have shape [B, grid_height * grid_width]")
    target_counts = component_counts(labels, (height, width), connectivity)
    scores = logits.float()
    fields = []
    for class_id in (0, 1):
        alternatives = [index for index in range(3) if index != class_id]
        margin = scores[..., class_id] - scores[..., alternatives].amax(dim=-1)
        fields.append(torch.sigmoid(margin))
    confidence = torch.stack(fields, dim=1)
    detached = confidence.detach().cpu().numpy()
    terms = []
    for frame_index in range(logits.shape[0]):
        for class_id in (0, 1):
            if target_counts[frame_index, class_id] != 1:
                continue
            essential, pairs = _persistence_pairs(detached[frame_index, class_id], neighbors)
            field = confidence[frame_index, class_id]
            term = (1.0 - field[essential]).square()
            if pairs:
                births, deaths = zip(*pairs)
                term = term + (field[list(births)] - field[list(deaths)]).square().sum()
            terms.append(term)
    return torch.stack(terms).mean() if terms else scores.sum() * 0.0
