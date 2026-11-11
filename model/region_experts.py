"""Fixed semantic routing through independent body and boundary token experts."""

import torch
from torch import nn


class BodyEdgeExpertRouter(nn.Module):
    """Apply a residual MLP only to tokens assigned to its semantic region.

    Class 0 is background and passes through unchanged. Classes 1 and 2 use
    separate trainable experts; no learned gate or extra labels are required.
    The output retains the original patch ordering and embedding dimension.
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("expert embedding and hidden dimensions must be positive")
        self.embed_dim = embed_dim
        self.body_expert = self._make_expert(embed_dim, hidden_dim)
        self.edge_expert = self._make_expert(embed_dim, hidden_dim)

    @staticmethod
    def _make_expert(embed_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self, tokens: torch.Tensor, patch_classes: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError("tokens must have shape [B, N, embed_dim]")
        if patch_classes.ndim != 2 or patch_classes.shape != tokens.shape[:2]:
            raise ValueError("patch_classes must have shape [B, N] matching tokens")
        if patch_classes.dtype is not torch.long:
            raise ValueError("patch_classes must have torch.long dtype")
        if patch_classes.device != tokens.device:
            raise ValueError("patch_classes and tokens must be on the same device")
        if torch.any((patch_classes < 0) | (patch_classes > 2)):
            raise ValueError("patch_classes must contain only background=0, body=1, edge=2")

        routed = tokens.clone()
        for region, expert in ((1, self.body_expert), (2, self.edge_expert)):
            mask = patch_classes == region
            if torch.any(mask):
                selected = tokens[mask]
                routed[mask] = selected + expert(selected)
        return routed
