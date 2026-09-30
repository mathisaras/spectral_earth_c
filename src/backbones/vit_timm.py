"""
ViT base class using VisionTransformer from timm.models.vision_transformer.
"""

from __future__ import annotations

from functools import partial

import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer


class VitBaseTimm(VisionTransformer):
    """
    ViT-Base: inherits timm's VisionTransformer with standard ViT-Base params
    (12 layers, 12 heads, 768 dim). Configurable img_size, patch_size, in_chans.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=None,
        global_pool: str = "token",
        num_classes: int = 0,
        name: str = "vit_b_timm",
        **kwargs,
    ) -> None:
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            global_pool=global_pool,
            num_classes=num_classes,
            **kwargs,
        )
        self.name = name


def vit_b_timm(
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    name: str = "vit_b_timm",
    **kwargs,
) -> VitBaseTimm:
    """Factory: ViT-Base from timm (12 layers, 12 heads, 768 dim)."""
    return VitBaseTimm(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        name=name,
        **kwargs,
    )
