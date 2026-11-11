"""Vision Transformer regression over selected or zero-filled patch grids."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .profiles import get_profile
from .region_experts import BodyEdgeExpertRouter


@dataclass(slots=True)
class RegressorOutput:
    """Weight prediction and retained-patch counts for a batch."""

    prediction: torch.Tensor
    selected_counts: torch.Tensor


def pack_kept_tokens(
    tokens: torch.Tensor, keep_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack row-major selected tokens and mark batch padding positions."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, N, D]")
    if keep_mask.ndim != 2 or tuple(keep_mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError("keep_mask must have shape [B, N] matching tokens")
    if keep_mask.dtype is not torch.bool:
        raise ValueError("keep_mask must have boolean dtype")
    if keep_mask.device != tokens.device:
        raise ValueError("keep_mask and tokens must be on the same device")

    counts = keep_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("keep_mask must retain at least one patch per sample")
    max_count = int(counts.max().item())
    packed = tokens.new_zeros(tokens.shape[0], max_count, tokens.shape[2])
    padding_mask = torch.ones(
        tokens.shape[0], max_count, dtype=torch.bool, device=tokens.device
    )
    for row in range(tokens.shape[0]):
        selected = tokens[row, keep_mask[row]]
        packed[row, : selected.shape[0]] = selected
        padding_mask[row, : selected.shape[0]] = False
    return packed, padding_mask, counts


class PigWeightViT(nn.Module):
    """Regress weight with sparse selection or a full zero-filled patch grid."""

    def __init__(
        self,
        input_channels: int,
        image_size: tuple[int, int],
        patch_size: int,
        profile: str,
        overrides: Mapping[str, int | float] | None = None,
        *,
        expert_routing: bool = False,
        background_mode: str = "drop",
    ) -> None:
        super().__init__()
        image_height, image_width = image_size
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_size dimensions must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if not isinstance(expert_routing, bool):
            raise ValueError("expert_routing must be a boolean")
        if not isinstance(background_mode, str) or background_mode not in ("drop", "zero"):
            raise ValueError("background_mode must be 'drop' or 'zero'")

        config = get_profile("regressor", profile, overrides)
        self.input_channels = input_channels
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.profile_name = profile
        self.overrides = dict(overrides or {})
        self.expert_routing = expert_routing
        self.background_mode = background_mode
        self.grid_size = (image_height // patch_size, image_width // patch_size)
        patch_count = self.grid_size[0] * self.grid_size[1]

        self.patch_embed = nn.Conv2d(
            input_channels,
            config.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.expert_router = (
            BodyEdgeExpertRouter(config.embed_dim, int(config.embed_dim * config.mlp_ratio))
            if expert_routing else None
        )
        self.position_tokens = nn.Parameter(
            torch.empty(1, patch_count, config.embed_dim)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_tokens, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, inputs: torch.Tensor, keep_mask: torch.Tensor,
        *, patch_classes: torch.Tensor | None = None,
    ) -> RegressorOutput:
        """Predict from foreground patches, preserving zero-filled slots if configured."""
        expected_shape = (self.input_channels, *self.image_size)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected_shape:
            raise ValueError(
                "expected input shape [B, C, H, W] with "
                f"[C, H, W]={expected_shape}, got {tuple(inputs.shape)}"
            )

        expected_mask_shape = (inputs.shape[0], self.position_tokens.shape[1])
        if keep_mask.shape != expected_mask_shape or keep_mask.dtype is not torch.bool:
            raise ValueError("keep_mask must be boolean with shape [B, N] matching tokens")
        if keep_mask.device != inputs.device:
            raise ValueError("keep_mask and inputs must be on the same device")
        if self.background_mode == "zero":
            # Mask normalized inputs out of place: the contributor may share
            # this tensor and must continue to see the original full image.
            pixel_mask = keep_mask.reshape(inputs.shape[0], 1, *self.grid_size)
            pixel_mask = pixel_mask.repeat_interleave(self.patch_size, dim=2)
            pixel_mask = pixel_mask.repeat_interleave(self.patch_size, dim=3)
            inputs = inputs.masked_fill(~pixel_mask, 0)

        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        if self.expert_router is not None:
            if patch_classes is None:
                raise ValueError("patch_classes are required when expert_routing is enabled")
            if patch_classes.shape != tokens.shape[:2]:
                raise ValueError("patch_classes must have shape [B, N] matching tokens")
            if patch_classes.device != tokens.device:
                raise ValueError("patch_classes and tokens must be on the same device")
            if torch.any(keep_mask & (patch_classes == 0)):
                raise ValueError("retained patches must be routed to body or edge")
            tokens = self.expert_router(tokens, patch_classes.masked_fill(~keep_mask, 0))
        tokens = tokens + self.position_tokens
        if self.background_mode == "zero":
            # Biases and position embeddings must not revive background slots.
            # Keep every grid position in the sequence, including empty grids.
            packed = tokens.masked_fill(~keep_mask.unsqueeze(-1), 0)
            padding_mask = torch.zeros_like(keep_mask)
            counts = keep_mask.sum(dim=1)
        else:
            packed, padding_mask, counts = pack_kept_tokens(tokens, keep_mask)

        cls_tokens = self.cls_token.expand(inputs.shape[0], -1, -1)
        sequence = torch.cat((cls_tokens, packed), dim=1)
        cls_padding = torch.zeros(
            inputs.shape[0], 1, dtype=torch.bool, device=inputs.device
        )
        padding_mask = torch.cat((cls_padding, padding_mask), dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        prediction = self.regression_head(self.norm(encoded[:, 0])).squeeze(-1)
        return RegressorOutput(prediction=prediction, selected_counts=counts)
