"""Transformer classifier that predicts a contribution class for each image patch."""

from collections.abc import Mapping

import torch
from torch import nn

from .profiles import get_profile
from .connectivity import decode_patch_classes


class PatchContributionNet(nn.Module):
    """Classify every non-overlapping input patch as background, pig, or edge."""

    def __init__(
        self,
        input_channels: int,
        image_size: tuple[int, int],
        patch_size: int,
        profile: str,
        overrides: Mapping[str, int | float] | None = None,
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

        config = get_profile("contributor", profile, overrides)
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.profile_name = profile
        self.overrides = dict(overrides or {})
        self.grid_size = (image_height // patch_size, image_width // patch_size)
        patch_count = self.grid_size[0] * self.grid_size[1]

        self.patch_embed = nn.Conv2d(
            input_channels,
            config.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_tokens = nn.Parameter(torch.empty(1, patch_count, config.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(config.embed_dim, 3)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_tokens, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return per-patch logits in row-major patch-grid order."""
        expected_shape = (self.input_channels, *self.image_size)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected_shape:
            raise ValueError(
                "expected input shape [B, C, H, W] with "
                f"[C, H, W]={expected_shape}, got {tuple(inputs.shape)}"
            )

        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_tokens
        tokens = self.encoder(tokens)
        return self.classifier(self.norm(tokens))

    @torch.no_grad()
    def predict_classes(
        self, inputs: torch.Tensor, *, enforce_connectivity: bool = True, connectivity: int = 8
    ) -> torch.Tensor:
        """Decode [B, N] classes; use eval() for deterministic inference."""
        return decode_patch_classes(
            self(inputs), self.grid_size,
            enforce_connectivity=enforce_connectivity, connectivity=connectivity,
        )
