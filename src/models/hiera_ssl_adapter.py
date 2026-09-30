import math
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def is_hiera_style_backbone(backbone: nn.Module) -> bool:
    """Best-effort check for Hiera-style backbones used in this repo."""
    required = (
        "mask_spatial_shape",
        "q_pool",
        "q_stride",
        "mu_size",
        "norm",
    )
    has_block_stack = hasattr(backbone, "blocks")
    return has_block_stack and all(hasattr(backbone, attr) for attr in required)


class HieraMaskedBackboneAdapter(nn.Module):
    """
    Adapter exposing a MaskedVisionTransformer-like token API for Hiera backbones.

    The adapter keeps Hiera's dense-token path and exposes a ViT-like token API.

    Important detail for masked-token SSL:
    Hiera's native `mask` argument is MAE-style keep/drop masking. It drops masked
    units from the sequence, which is not what iBOT expects. For token-masked
    training we therefore mask the image/grid input first and then run the dense
    encoder so the masked outputs remain contextualized by the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        include_cls_token: bool = True,
        global_pool: str = "",
    ) -> None:
        super().__init__()
        if not is_hiera_style_backbone(backbone):
            raise TypeError(
                f"HieraMaskedBackboneAdapter expected a Hiera-style backbone, got {type(backbone).__name__}."
            )

        self.backbone = backbone
        self.include_cls_token = include_cls_token
        self.global_pool = global_pool
        self.attn_pool = None
        self.embed_dim = self._infer_embed_dim(backbone)
        self.num_patches = int(math.prod(backbone.mask_spatial_shape))
        self.num_prefix_tokens = 1 if include_cls_token else 0
        self.mask_spatial_shape = tuple(int(v) for v in backbone.mask_spatial_shape)
        if hasattr(backbone, "tokens_spatial_shape"):
            self.tokens_spatial_shape = tuple(int(v) for v in backbone.tokens_spatial_shape)
        self.patch_embed = getattr(backbone, "patch_embed", None)

        grid_size = self.mask_spatial_shape
        if len(grid_size) != 2:
            raise ValueError(
                "HieraMaskedBackboneAdapter currently supports 2D backbones only."
            )
        self.grid_size: Tuple[int, int] = grid_size

        if hasattr(backbone, "img_size") and self.grid_size[0] > 0:
            self.effective_patch_size = int(backbone.img_size // self.grid_size[0])
        else:
            self.effective_patch_size = None

        # Provide vit.patch_embed.grid_size/num_patches for existing training code.
        self.vit = SimpleNamespace(
            patch_embed=SimpleNamespace(
                grid_size=self.grid_size,
                num_patches=self.num_patches,
            )
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _infer_embed_dim(backbone: nn.Module) -> int:
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            block = backbone.blocks[-1]
            if hasattr(block, "dim_out"):
                return int(block.dim_out)
        if hasattr(backbone, "embed_dim"):
            return int(backbone.embed_dim)
        if hasattr(backbone, "num_features"):
            return int(backbone.num_features)
        raise ValueError(
            f"Could not infer embedding dimension for backbone type {type(backbone).__name__}."
        )

    @property
    def sequence_length(self) -> int:
        return self.num_patches + self.num_prefix_tokens

    def _encode_dense_tokens(
        self,
        images: Tensor,
        spectral_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = images.shape[0]
        keep_all = torch.ones(
            batch_size,
            self.num_patches,
            dtype=torch.bool,
            device=images.device,
        )

        tokens = self.backbone(images, mask=keep_all, spectral_mask=spectral_mask)
        tokens = self.backbone.norm(tokens)

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected token tensor with shape [B, N, D], got {tuple(tokens.shape)}."
            )

        if tokens.shape[1] == self.num_patches + 1:
            cls_token = tokens[:, :1]
            patch_tokens = tokens[:, 1:]
        elif tokens.shape[1] == self.num_patches:
            patch_tokens = tokens
            cls_token = patch_tokens.mean(dim=1, keepdim=True)
        else:
            raise RuntimeError(
                "Hiera output token count is incompatible with adapter expectations. "
                f"Expected {self.num_patches} (or {self.num_patches + 1} with CLS), got {tokens.shape[1]}. "
                "Ensure q_pool/mask_unit_size are configured so final encoder tokens map to mask units."
            )

        if self.include_cls_token:
            return torch.cat([cls_token, patch_tokens], dim=1)
        return patch_tokens

    def _token_mask_to_patch_mask(
        self,
        *,
        tokens_shape: Tuple[int, int],
        idx_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set.")

        batch_size, seq_len = tokens_shape

        if idx_mask is not None:
            idx_mask = idx_mask.long()
            token_mask = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=idx_mask.device,
            )
            token_mask.scatter_(1, idx_mask, True)
        else:
            token_mask = mask

        if token_mask is None:
            return None

        token_mask = token_mask.to(dtype=torch.bool)
        if token_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"Mask shape {tuple(token_mask.shape)} does not match token shape {(batch_size, seq_len)}."
            )

        if self.num_prefix_tokens > 0:
            patch_mask = token_mask[:, self.num_prefix_tokens :]
        else:
            patch_mask = token_mask

        if patch_mask.shape[1] != self.num_patches:
            raise ValueError(
                "Patch-token mask shape is incompatible with the adapter grid. "
                f"Expected {self.num_patches} patch entries, got {patch_mask.shape[1]}."
            )
        return patch_mask

    def _mask_images_for_context(
        self,
        images: Tensor,
        patch_mask: Optional[Tensor],
    ) -> Tensor:
        if patch_mask is None:
            return images

        keep_mask = (~patch_mask).view(images.shape[0], 1, *self.mask_spatial_shape).float()
        keep_mask = F.interpolate(
            keep_mask,
            size=images.shape[2:],
            mode="nearest",
        ).to(dtype=images.dtype, device=images.device)
        return images * keep_mask

    def _mask_tokens(
        self,
        tokens: Tensor,
        idx_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set.")

        if idx_mask is not None:
            idx_mask = idx_mask.long()
            mask = torch.zeros(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
            mask.scatter_(1, idx_mask, True)

        if mask is None:
            return tokens

        mask = mask.to(dtype=torch.bool, device=tokens.device)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match token shape {tuple(tokens.shape[:2])}."
            )

        mask_token = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
        return torch.where(mask.unsqueeze(-1), mask_token, tokens)

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        spectral_mask: Optional[Tensor] = None,
    ) -> Tensor:
        patch_mask = self._token_mask_to_patch_mask(
            tokens_shape=(images.shape[0], self.sequence_length),
            idx_mask=idx_mask,
            mask=mask,
        )
        if patch_mask is not None:
            images = self._mask_images_for_context(images, patch_mask)

        tokens = self._encode_dense_tokens(images, spectral_mask=spectral_mask)

        if idx_keep is not None:
            idx_keep = idx_keep.long()
            gather_idx = idx_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            tokens = torch.gather(tokens, dim=1, index=gather_idx)

        return tokens

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        spectral_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tokens = self.encode(
            images,
            idx_mask=idx_mask,
            idx_keep=idx_keep,
            mask=mask,
            spectral_mask=spectral_mask,
        )
        if self.attn_pool is not None:
            return self.attn_pool(tokens)
        if self.global_pool == "avg":
            return tokens[:, self.num_prefix_tokens :].mean(dim=1)
        if self.global_pool:
            return tokens[:, 0]
        return tokens

    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
        spectral_mask: Optional[Tensor] = None,
    ):
        out = self.encode(
            images,
            idx_mask=idx_mask,
            idx_keep=idx_keep,
            mask=mask,
            spectral_mask=spectral_mask,
        )
        return out, []
