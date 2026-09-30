import contextlib
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .hiera import do_masked_conv, get_1d_sincos_pos_embed_from_grid

try:
    from timm.models.vision_transformer import Block as ViTBlock
except Exception as exc:  # pragma: no cover
    raise ImportError("Please install timm to use the spectral encoder.") from exc


def _build_1d_sincos_pos_embed(
    embed_dim: int,
    length: int,
) -> torch.Tensor:
    grid = np.arange(length, dtype=np.float32)
    pos = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return torch.from_numpy(pos).unsqueeze(0)


class _SpectralStageEncoder(nn.Module):
    """Grouped spectral tokenizer with projected-attention aggregation."""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_stride: Tuple[int, int],
        patch_kernel: Tuple[int, int],
        patch_padding: Tuple[int, int],
        channel_groups: List[List[int]],
        embed_dim: int,
        spec_depth: int,
        spec_num_heads: int,
        mlp_ratio: float,
        spectral_pos_embed_type: str = "learnable",
        pooling: str = "projected_attention",
        spectral_token_dim: Optional[int] = None,
        spectral_fusion_dim: Optional[int] = None,
        spectral_fusion_heads: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer: nn.LayerNorm = nn.LayerNorm,
        sdp_policy: str = "auto",
        flash_max_seqlen: int = 4096,
        sdp_max_seqlen_dim: int = 65535,
    ) -> None:
        super().__init__()
        pooling_aliases = {
            "proj_attn": "projected_attention",
            "projected_attention": "projected_attention",
        }
        pooling = pooling_aliases.get(str(pooling).lower(), str(pooling).lower())
        if pooling != "projected_attention":
            raise ValueError(
                "SpectralEarth-FM uses projected_attention spectral pooling. "
                f"Got '{pooling}'."
            )

        self.output_dim = int(embed_dim)
        self.D = int(spectral_token_dim or embed_dim)
        self.fusion_dim = int(spectral_fusion_dim or embed_dim)
        self.fusion_heads = int(spectral_fusion_heads or spec_num_heads)
        if self.D % int(spec_num_heads) != 0:
            raise ValueError(
                f"spectral_token_dim ({self.D}) must be divisible by "
                f"spec_num_heads ({spec_num_heads})."
            )
        if self.fusion_dim % self.fusion_heads != 0:
            raise ValueError(
                f"spectral_fusion_dim ({self.fusion_dim}) must be divisible by "
                f"spectral_fusion_heads ({self.fusion_heads})."
            )

        if sdp_policy not in {"auto", "math", "flash"}:
            raise ValueError(f"Unknown sdp_policy: {sdp_policy}")
        self.sdp_policy = sdp_policy
        self.flash_max_seqlen = int(flash_max_seqlen)
        self.sdp_max_seqlen_dim = int(sdp_max_seqlen_dim)

        self.img_size = tuple(int(v) for v in img_size)
        self.patch_stride = tuple(int(v) for v in patch_stride)
        self.patch_kernel = tuple(int(v) for v in patch_kernel)
        self.patch_padding = tuple(int(v) for v in patch_padding)
        self.channel_groups = channel_groups
        self.G = len(channel_groups)
        self.spec_num_heads = int(spec_num_heads)
        self.pooling = pooling

        self.patch_embed_list = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=len(group),
                    out_channels=self.D,
                    kernel_size=self.patch_kernel,
                    stride=self.patch_stride,
                    padding=self.patch_padding,
                    bias=True,
                )
                for group in self.channel_groups
            ]
        )

        self.spectral_mask_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.trunc_normal_(self.spectral_mask_token, std=0.02)

        if spectral_pos_embed_type == "learnable":
            self.spec_pos_embed = nn.Parameter(torch.zeros(1, self.G, self.D))
            nn.init.trunc_normal_(self.spec_pos_embed, std=0.02)
        elif spectral_pos_embed_type == "sincos":
            pos = _build_1d_sincos_pos_embed(self.D, self.G).float()
            self.register_buffer("spec_pos_embed", pos, persistent=True)
        else:
            raise ValueError(f"Unknown spectral_pos_embed_type: {spectral_pos_embed_type}")

        self.spec_blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=self.D,
                    num_heads=self.spec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for _ in range(int(spec_depth))
            ]
        )
        self.spec_norm = norm_layer(self.D)

        self.fusion_input_proj = nn.Linear(self.D, self.fusion_dim)
        self.fusion_query_token = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.fusion_heads,
            batch_first=True,
        )
        self.fusion_norm = norm_layer(self.fusion_dim)
        if spectral_pos_embed_type == "learnable":
            self.fusion_pos_embed = nn.Parameter(torch.zeros(1, self.G, self.fusion_dim))
            nn.init.trunc_normal_(self.fusion_pos_embed, std=0.02)
        else:
            fusion_pos = _build_1d_sincos_pos_embed(self.fusion_dim, self.G).float()
            self.register_buffer("fusion_pos_embed", fusion_pos, persistent=True)
        nn.init.trunc_normal_(self.fusion_query_token, std=0.02)

        self._fusion_attn_chunk_size = 16384
        self.output_proj = (
            nn.Identity()
            if self.fusion_dim == self.output_dim
            else nn.Linear(self.fusion_dim, self.output_dim)
        )

        self.apply(self._init_weights)

    def _sdp_context(
        self,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if device.type != "cuda":
            return contextlib.nullcontext()
        if self.sdp_policy == "math":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        if self.sdp_policy == "flash":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            )

        head_dim = self.D // self.spec_num_heads
        dtype_ok = dtype in (torch.float16, torch.bfloat16)
        batch_heads_ok = (batch_size * self.spec_num_heads) <= self.sdp_max_seqlen_dim
        flash_ok = (
            dtype_ok
            and head_dim <= 128
            and seq_len <= self.flash_max_seqlen
            and seq_len <= self.sdp_max_seqlen_dim
            and batch_heads_ok
        )
        return torch.backends.cuda.sdp_kernel(
            enable_flash=flash_ok,
            enable_math=not flash_ok,
            enable_mem_efficient=False,
        )

    def _run_fusion_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        n = q.shape[0]
        if n <= self._fusion_attn_chunk_size:
            if q.device.type == "cuda":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    out, _ = self.fusion_attn(q, k, v, need_weights=False)
                    return out
            out, _ = self.fusion_attn(q, k, v, need_weights=False)
            return out

        out_chunks: List[torch.Tensor] = []
        for start in range(0, n, self._fusion_attn_chunk_size):
            end = min(start + self._fusion_attn_chunk_size, n)
            q_i = q[start:end]
            k_i = k[start:end]
            v_i = v[start:end]
            if q.device.type == "cuda":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    out_i, _ = self.fusion_attn(q_i, k_i, v_i, need_weights=False)
            else:
                out_i, _ = self.fusion_attn(q_i, k_i, v_i, need_weights=False)
            out_chunks.append(out_i)
        return torch.cat(out_chunks, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        spectral_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        group_tokens = []
        for idx, group in enumerate(self.channel_groups):
            gx = do_masked_conv(x[:, group], self.patch_embed_list[idx], mask)
            group_tokens.append(gx.flatten(2).transpose(1, 2))

        x_grp = torch.stack(group_tokens, dim=1)
        batch_size, groups, num_tokens, dim = x_grp.shape
        if num_tokens == 0:
            raise ValueError(f"No spatial tokens produced for input {height}x{width}.")

        x_spec = x_grp.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, groups, dim)
        if spectral_mask is not None:
            expected = (batch_size, num_tokens, groups)
            if tuple(spectral_mask.shape) != expected:
                raise ValueError(
                    f"spectral_mask must have shape {expected}, got {tuple(spectral_mask.shape)}"
                )
            keep = spectral_mask.to(dtype=torch.bool, device=x_spec.device).reshape(
                batch_size * num_tokens,
                groups,
                1,
            )
            mask_token = self.spectral_mask_token.expand(batch_size * num_tokens, groups, -1)
            x_spec = torch.where(keep, x_spec, mask_token.to(dtype=x_spec.dtype))

        x_spec = x_spec + self.spec_pos_embed
        with self._sdp_context(x_spec.shape[1], batch_size * num_tokens, x_spec.dtype, x_spec.device):
            for block in self.spec_blocks:
                x_spec = block(x_spec)
        x_spec = self.spec_norm(x_spec)

        x_fused = self.fusion_input_proj(x_spec)
        x_fused = x_fused + self.fusion_pos_embed
        query = self.fusion_query_token.expand(batch_size * num_tokens, -1, -1)
        fused = self._run_fusion_attention(query, x_fused, x_fused)
        out_feats = self.fusion_norm(fused.squeeze(1))
        out_feats = self.output_proj(out_feats)
        return out_feats.view(batch_size, num_tokens, self.output_dim)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            weight = module.weight
            nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
