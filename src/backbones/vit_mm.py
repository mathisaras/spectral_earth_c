import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial

from .spectral_adapter import SpectralAdapter


class ViTMM(nn.Module):
    """
    Vision Transformer backbone with sensor-specific stems + shared transformer trunk.

    Each sensor has:
      - A small stem (SpectralAdapter for EnMAP; direct PatchEmbed for others)
      - N sensor-specific transformer blocks
    After sensor-specific processing, features are fused and passed through
    M shared transformer blocks for joint representation.

    Outputs a per-patch [CLS] token representation of dimension `embed_dim`.
    """

    def __init__(
        self,
        sensors: list[str],
        in_channels: dict[str, int],
        sensor_patch_sizes: dict[str, int],
        sensor_img_sizes: dict[str, int],
        reduced_channels: int = 128,
        embed_dim: int = 768,
        sensor_depth: int = 2,
        shared_depth: int = 10,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dynamic_img_size: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.sensor_img_sizes = sensor_img_sizes
        self.sensors = sensors

        # ===== sensor-specific stems + patch embedding =====
        self.stems = nn.ModuleDict()
        for s in sensors:
            if s == "enmap":
                # SpectralAdapter reduces EnMAP bands to `reduced_channels`
                stem = nn.Sequential(
                    SpectralAdapter(),
                    PatchEmbed(
                        img_size=self.sensor_img_sizes[s],
                        patch_size=sensor_patch_sizes[s],
                        in_chans=reduced_channels,
                        embed_dim=embed_dim,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        flatten=True,
                    ),
                )
            else:
                # Directly embed raw bands for other sensors
                stem = nn.Sequential(
                    PatchEmbed(
                        img_size=self.sensor_img_sizes[s],
                        patch_size=sensor_patch_sizes[s],
                        in_chans=in_channels[s],
                        embed_dim=embed_dim,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        flatten=True,
                    ),
                )
            self.stems[s] = stem

        # ===== sensor-specific transformer blocks =====
        self.sensor_blocks = nn.ModuleDict()
        for s in sensors:
            blocks = []
            for _ in range(sensor_depth):
                blk = Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                blocks.append(blk)
            self.sensor_blocks[s] = nn.Sequential(*blocks)

        # ===== shared CLS token + positional embedding =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=0.0)

        # ===== shared transformer trunk =====
        shared_blocks = []
        for _ in range(shared_depth):
            blk = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            shared_blocks.append(blk)
        self.shared_blocks = nn.Sequential(*shared_blocks)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Dict mapping sensor -> tensor of shape [B, C, H, W]
        Returns:
            cls_feats: Tensor [B, embed_dim]
        """
        batch_size = next(iter(x.values())).size(0)
        sensor_tokens = []

        # process each sensor separately
        for s in self.sensors:
            xi = x[s]  # [B, C, H, W]
            out = self.stems[s](xi)           # -> [B, N, embed_dim]
            out = self.sensor_blocks[s](out)  # -> [B, N, embed_dim]
            sensor_tokens.append(out)

        # concatenate tokens from all sensors
        tokens = torch.cat(sensor_tokens, dim=1)  # [B, total_patches, embed_dim]

        # prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B,1,embed_dim]
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # adjust positional embedding if needed
        if self.pos_embed.size(1) != tokens.size(1):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, tokens.size(1), tokens.size(2)), requires_grad=True
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # shared transformer trunk
        out = self.shared_blocks(tokens)
        out = self.norm(out)

        # return CLS token embedding
        return out[:, 0]


class ViTMMBase(ViTMM):
    """
    Base multimodal ViT with default settings for EnMAP, Sentinel-2, and Landsat-8.
    """
    def __init__(
        self,
        patch_size: int = 128,
        dynamic_img_size: bool = True,
        **kwargs,
    ):
        super().__init__(
            sensors=["enmap", "s2", "l8"],
            in_channels={"enmap": 202, "s2": 12, "l8": 7},
            sensor_patch_sizes={"enmap": 4, "s2": 12, "l8": 4},
            sensor_img_sizes={
                "enmap": 128,
                "s2": 384,
                "l8": 128
            },
            reduced_channels=128,
            embed_dim=768,
            sensor_depth=2,
            shared_depth=10,
            num_heads=12,
            mlp_ratio=4.0,
            dynamic_img_size=dynamic_img_size,
            **kwargs,
        )
