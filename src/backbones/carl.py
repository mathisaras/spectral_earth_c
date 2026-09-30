# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.
# https://github.com/IMSY-DKFZ/CARL/tree/master
# Download weights: wget -O carl_ssl_checkpoint.ckpt https://zenodo.org/records/18671944/files/ssl_checkpoint_carl.ckpt

import logging
import math
from functools import partial
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

import numpy as np
import timm
from timm.models import Eva, VisionTransformer
from timm.models.eva import apply_keep_indices_nlc
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch import Tensor

from src.utils.sensor_registry import SensorRegistry
from src.utils.spectral_metadata import load_spectral_metadata


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, theta=10000):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid, theta=theta)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, theta=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / theta**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, sigma):
        super().__init__()
        self.d_hid = d_hid
        self.sigma = sigma
   
        rng_state = torch.get_rng_state()
        temp_gen = torch.Generator()
        seed = 0
        temp_gen.manual_seed(seed)  # Fixed seed for reproducible sampling    
        B_gauss = 2 * torch.tensor(math.pi) * (torch.randn((d_hid // 2), generator=temp_gen) * sigma + 0)
        torch.set_rng_state(rng_state)
        B_gauss = torch.stack([B_gauss[i//2] for i in range(d_hid)], dim=0)
        self.register_buffer("dims", B_gauss, persistent=True)

    def get_position_angle_vec(self, position):
        position = position.unsqueeze(-1)
        dims = self.dims.unsqueeze(0).unsqueeze(0)
        out = position * dims
        return out

    def forward(self, w) -> torch.Tensor:
        if self.sigma == 0:
            b, c = w.shape
            d = self.d_hid
            return torch.zeros(b, c, d, device=w.device, dtype=w.dtype)
        wlens_encoded = self.get_position_angle_vec(w)  # No need to clone here initially

        sin_encoded = torch.sin(wlens_encoded[..., 0::2])  # Compute sine out-of-place
        cos_encoded = torch.cos(wlens_encoded[..., 1::2])  # Compute cosine out-of-place

        # Construct the output tensor by combining sine and cosine results
        wlens_encoded_out = torch.empty_like(wlens_encoded)  # Create a new tensor with the same shape
        wlens_encoded_out[..., 0::2] = sin_encoded
        wlens_encoded_out[..., 1::2] = cos_encoded

        return wlens_encoded_out


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_bias, dropout_p=self.attn_drop.p, is_causal=False
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dim_k: int = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        if dim_k is None:
            dim_k = dim

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.v = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_bias=None) -> Tensor:

        B, N, C = q.shape
        B, N_k, _ = k.shape
        q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(k).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(v).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_bias, dropout_p=self.attn_drop.p, is_causal=False
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    if len(masks) == 1:
        # Optimize single mask case
        m = masks[0]
        if x.ndim == 4:
            mask_keep = m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-2), x.size(-1))
        elif x.ndim == 3:
            mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return torch.gather(x, dim=1, index=mask_keep)
    
    all_x = []
    for m in masks:
        if x.ndim == 4:
            mask_keep = m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-2), x.size(-1))
        elif x.ndim == 3:
            mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        proj_drop=0
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.is_cross_attn = isinstance(self.attn, CrossAttention)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x, keys=None, values=None) -> Tensor:
        if self.is_cross_attn:
            return self.forward_crossattention(x, keys, values)

        def attn_residual_func(x: Tensor) -> Tensor:
            v = self.attn(self.norm1(x))
            return self.ls1(v)

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                x,
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                x,
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            v = attn_residual_func(x)
            x = x + self.drop_path1(v)
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            v = attn_residual_func(x)
            x = x + v
            x = x + ffn_residual_func(x)
        return x

    def forward_crossattention(self, queries, keys=None, values=None) -> Tensor:
        def attn_residual_func(queries, keys, values) -> Tensor:
            key_norm = self.norm1(keys)
            v_norm = self.norm1(values)
            x = self.attn(q=self.norm1(queries), k=key_norm, v=v_norm)
            return self.ls1(x)

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.2:
            # the overhead is compensated only for a drop path rate larger than 0.1
            queries = drop_add_residual_stochastic_depth(
                queries,
                keys,
                values,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            queries = drop_add_residual_stochastic_depth(
                queries,
                keys,
                values,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            out = attn_residual_func(queries, keys, values)
            queries = queries + self.drop_path1(out)
            queries = queries + self.drop_path2(ffn_residual_func(queries))
        else:
            out = attn_residual_func(queries, keys, values)
            queries = queries + out
            queries = queries + ffn_residual_func(queries)
        return queries
    

def drop_add_residual_stochastic_depth(
    x: Tensor,
    keys,
    values,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    keys_subset = keys[brange]
    values_subset = values[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset, keys_subset, values_subset)
    if isinstance(residual, tuple):
        residual = residual[0]
    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class SpectralEncoder(nn.Module):
    """Encoder that ingests spectral tokens and produces query embeddings.

    The encoder accepts a sequence of spectral tokens ``x`` and a corresponding
    wavelength tensor ``w``. It builds wavelength positional encodings (via
    ``PositionalEncoding``), maintains a small set of learned ``queries``
    (spectral representations), and alternates self- and cross-attention blocks
    so queries can attend to the spectral tokens and produce a compact spectral
    representation for downstream tasks.

    Args:
        embed_dim: Embedding dimension for tokens and queries.
        n_queries: Number of learned spectral representations.
        pos_enc_sigma: Scaling parameter for wavelength positional encoding.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for MLP hidden dimension.
        qkv_bias: Whether to include bias in query, key, value projections.
        ffn_bias: Whether to include bias in feed-forward network.
        proj_bias: Whether to include bias in projection layers.
        drop_path_rate: Stochastic depth rate.
        drop_path_uniform: Whether to use uniform stochastic depth schedule.
        act_layer: Activation function layer.
        proj_drop: Dropout rate for projections.
        drop: General dropout rate.
        attn_drop: Dropout rate for attention weights.
        layer_scale: Layer scaling factor for initialization.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_queries: int = 4,
        pos_enc_sigma: float = 3,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        act_layer=nn.GELU,
        proj_drop: float = 0.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layer_scale: float = 1e-5,
    ):
        super().__init__()

        # Store configuration
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_queries = n_queries

        # Initialize learned query vectors and their positional encodings
        self.queries = nn.Parameter(
            torch.zeros(1, n_queries, embed_dim), requires_grad=True
        )
        self.queries_pos_enc = nn.Parameter(
            torch.zeros(1, n_queries, embed_dim), requires_grad=False
        )

        # Build stochastic depth schedule
        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # Build alternating attention blocks (self-attention, cross-attention, ...)
        blocks_list = []
        for i in range(depth):
            attn_class = SelfAttention if i % 2 == 0 else CrossAttention
            blocks_list.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=Mlp,
                    init_values=layer_scale,
                    attn_class=attn_class,
                    proj_drop=proj_drop,
                    drop=drop,
                    attn_drop=attn_drop,
                )
            )

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.positional_embedding = PositionalEncoding(embed_dim, sigma=pos_enc_sigma)

        # Set up random generator for weight initialization
        self.rng = torch.Generator()
        self.rng.manual_seed(0)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize queries
        torch.nn.init.trunc_normal_(self.queries, std=0.02, generator=self.rng)
        # Initialize positional encodings for queries
        queries_pos_enc = get_1d_sincos_pos_embed(
            self.embed_dim, self.n_queries, cls_token=False, theta=10000
        )
        self.queries_pos_enc.data.copy_(torch.from_numpy(queries_pos_enc).float().unsqueeze(0))

        if self.embed_dim == 384:
            model_init = "vit_small_patch14_dinov2.lvd142m"
        elif self.embed_dim == 768:
            model_init = "vit_base_patch14_dinov2.lvd142m"
        else:
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    trunc_normal_(module.weight, std=.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif hasattr(module, 'init_weights'):
                    module.init_weights()
            self.apply(_init_weights)
            return

        # Initialize spectral encoder blocks from DINOv2 weights
        dino = timm.create_model(model_init, pretrained=True)
        block_weights = dino.state_dict()
        block_weights = {k: v for k, v in block_weights.items() if 'blocks' in k}
        new_block_weights = {}
        for k, v in block_weights.items():
            block_id = int(k.split('.')[1])
            if block_id >= self.n_blocks:
                continue
            if "attn.qkv.weight" in k:
                w_q = v.data[:self.embed_dim]
                w_k = v.data[self.embed_dim:(2*self.embed_dim)]
                w_v = v.data[-self.embed_dim:]
                k_q = k.replace("attn.qkv.weight", "attn.q.weight")
                k_k = k.replace("attn.qkv.weight", "attn.k.weight")
                k_v = k.replace("attn.qkv.weight", "attn.v.weight")
                new_block_weights[k_q] = w_q
                new_block_weights[k_k] = w_k
                new_block_weights[k_v] = w_v
            elif "attn.qkv.bias" in k:
                b_q = v.data[:self.embed_dim]
                b_k = v.data[self.embed_dim:(2*self.embed_dim)]
                b_v = v.data[-self.embed_dim:]
                k_q = k.replace("attn.qkv.bias", "attn.q.bias")
                k_k = k.replace("attn.qkv.bias", "attn.k.bias")
                k_v = k.replace("attn.qkv.bias", "attn.v.bias")
                new_block_weights[k_q] = b_q
                new_block_weights[k_k] = b_k
                new_block_weights[k_v] = b_v
            else:
                new_block_weights[k] = v
        missing, unexpected = self.load_state_dict(new_block_weights, strict=False)
        logging.info(f"Spectral Encoder weight initialization from {model_init}: {missing} missing, {unexpected} unexpected")
        del dino

    def prepare_tokens(
        self, x: torch.Tensor, w: torch.Tensor, masks=Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encodings to spectral tokens and apply optional masks.

        Args:
            x: Spectral token embeddings of shape (batch_size, seq_len, embed_dim).
            w: Wavelength values of shape (batch_size, seq_len).
            masks: Optional mask tensor to apply to tokens.

        Returns:
            Tuple of (encoded_tokens, masked_wavelengths):
                - encoded_tokens: Tokens with positional encodings added.
                - masked_wavelengths: Wavelength tensor with masks applied.
        """
        pos_enc = self.positional_embedding.forward(w)

        # Align batch dimensions if positional encoding is shared
        repeats = x.shape[0] // pos_enc.shape[0]
        assert pos_enc.shape[0] * repeats == x.shape[0], (
            "Batch size of x must be a multiple of positional encoding batch size."
            f" Got {x.shape[0]} and {pos_enc.shape[0]}."
        )
        if repeats > 1:
            pos_enc = pos_enc.repeat_interleave(repeats, dim=0)

        x = x + pos_enc
        # Apply masks if provided
        if masks is not None:
            x = apply_masks(x, masks)
 
        return x

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, masks: Optional[List[torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        """Encode spectral tokens and produce query embeddings.

        The network alternates between refining the spectral tokens with
        self-attention blocks and allowing the learned queries to attend to
        the tokens with cross-attention blocks.

        Args:
            x: Spectral token embeddings of shape (batch_size, seq_len, embed_dim).
            w: Wavelength values of shape (batch_size, seq_len).
            masks: Optional channel indices to keep, used in CARL-SSL for masked tokens.

        Returns:
            Dictionary:
                - queries: Learned query embeddings of shape (batch_size, n_queries, embed_dim).
                - spectral_tokens: Refined spectral tokens of shape (batch_size, seq_len, embed_dim).
        """

        # Prepare tokens with positional encodings and apply masks
        spectral_tokens = self.prepare_tokens(x, w, masks)

        # Initialize and encode query vectors
        queries = self.queries.expand(spectral_tokens.shape[0], -1, -1)
        queries_pos_enc = self.queries_pos_enc.expand_as(queries)
        queries = queries + queries_pos_enc

        # Apply alternating self- and cross-attention blocks
        for blk in self.blocks:
            if isinstance(blk.attn, CrossAttention):
                # Cross-attention: queries attend to spectral tokens
                queries = blk(queries, spectral_tokens, spectral_tokens)
            else:
                # Self-attention: refine spectral tokens
                spectral_tokens = blk(spectral_tokens)

        # Normalize query embeddings
        queries = self.norm(queries)

        return {"queries": queries, "spectral_tokens": spectral_tokens}
    

class TimmWrapper(nn.Module):
    """Wrapper for TIMM Vision Transformers.
    
    Provides a clean interface to the pre-trained TIMM models.
    
    Attributes:
        net: The underlying EVA2 Vision Transformer model.
        embed_dim: Embedding dimension of the model.
        num_prefix_tokens: Number of prefix tokens (e.g., cls token).
    """
    
    def __init__(
        self,
        model_name: str = "timm/eva02_base_patch14_224.mim_in22k",
        depth: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the spatial encoder.
        
        Args:
            model_name: Name of the TIMM model to load.
            depth: Number of transformer blocks to use (uses first `depth` blocks).
            grad_checkpointing: Whether to use gradient checkpointing for memory efficiency.
            model_kwargs: Additional keyword arguments to pass to the model.
            
        Raises:
            RuntimeError: If the model cannot be loaded from TIMM.
        """
        super().__init__()
        
        model_kwargs = dict(model_kwargs or {})
        dynamic_model_kwargs = {**model_kwargs, "dynamic_img_size": True}
        
        # Load pre-trained model from TIMM
        try:
            timm_module: Union[Eva, VisionTransformer] = timm.create_model(
                model_name,
                pretrained=True,
                **dynamic_model_kwargs
            )
        except Exception as e:
            if "dynamic_img_size" not in str(e):
                raise RuntimeError(f"Failed to load model '{model_name}' from TIMM: {e}")
            timm_module = timm.create_model(
                model_name,
                pretrained=True,
                **model_kwargs,
            )
        
        # Truncate to specified depth
        if depth is not None:
            timm_module.blocks = timm_module.blocks[:depth]
        self.net = timm_module

        # Store configuration
        self.num_prefix_tokens = self.net.num_prefix_tokens
        self.embed_dim = self.net.embed_dim

    def _resize_abs_pos_embed(self, pos_embed: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """Resize absolute position embeddings when the token grid changes."""
        num_prefix = int(self.num_prefix_tokens)
        if pos_embed.shape[1] == target_tokens:
            return pos_embed

        prefix = pos_embed[:, :num_prefix]
        pos_tokens = pos_embed[:, num_prefix:]
        target_grid_tokens = target_tokens - num_prefix
        old_size = int(math.sqrt(pos_tokens.shape[1]))
        new_size = int(math.sqrt(target_grid_tokens))
        if old_size * old_size != pos_tokens.shape[1] or new_size * new_size != target_grid_tokens:
            raise ValueError(
                "Cannot resize CARL spatial position embeddings for a non-square token grid: "
                f"{pos_embed.shape[1]} -> {target_tokens} tokens."
            )

        pos_tokens = pos_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        return torch.cat([prefix, pos_tokens], dim=1)

    def _pos_embed_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply TIMM spatial encoder positional embeddings to precomputed tokens.

        Older TIMM versions exposed ``_pos_embed`` for this.  The version in the
        current environment does not, so we reproduce the relevant EVA/VIT path
        while keeping compatibility with versions that still provide the method.
        """
        if hasattr(self.net, "_pos_embed"):
            out_pos_embed = self.net._pos_embed(x)
            if isinstance(out_pos_embed, tuple):
                return out_pos_embed
            return out_pos_embed, None

        if x.ndim == 4:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        elif x.ndim != 3:
            raise ValueError(f"Expected spatial tokens with shape BxHxWxD or BxNxD, got {tuple(x.shape)}")

        cls_token = getattr(self.net, "cls_token", None)
        if cls_token is not None:
            x = torch.cat((cls_token.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1), x), dim=1)

        pos_embed = getattr(self.net, "pos_embed", None)
        if pos_embed is not None:
            pos_embed = self._resize_abs_pos_embed(pos_embed, x.shape[1])
            x = x + pos_embed.to(dtype=x.dtype, device=x.device)

        pos_drop = getattr(self.net, "pos_drop", None)
        if pos_drop is not None:
            x = pos_drop(x)

        rot_pos_embed = self.net.rope.get_embed() if getattr(self.net, "rope", None) is not None else None
        patch_drop = getattr(self.net, "patch_drop", None)
        if patch_drop is not None:
            x, keep_indices = patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        return x, rot_pos_embed

    def forward(self, x: torch.Tensor, masks: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through the spatial encoder.
        
        Args:
            x: Input tensor of shape (B, H, W, D) or (B, N, D) where:
               - B is batch size
               - H, W are spatial dimensions (for spatial input)
               - N is sequence length (for sequence input)
               - D is embedding dimension
            masks: Optional list of long tensors indicating kept token indices. Used in CARL-SSL.
               
        Returns:
            Encoded features of shape (B, N, D) where N includes prefix tokens.
        """
        # Apply positional embeddings.  Supports both older TIMM ``_pos_embed``
        # and newer EVA/VIT modules that expose ``pos_embed`` directly.
        x, rot_pos_embed = self._pos_embed_tokens(x)
        hidden_states = x

        # Create attention mask if masks are provided
        attn_mask = self.create_attn_mask(x, masks)
        
        # Prepare kwargs for transformer blocks
        block_kwargs = {}
        if rot_pos_embed is not None:
            block_kwargs["rope"] = rot_pos_embed
        if attn_mask is not None:
            block_kwargs["attn_mask"] = attn_mask
        # Apply transformer blocks
        for block in self.net.blocks:
            hidden_states = block(hidden_states, **block_kwargs)
        # Remove prefix tokens (e.g., cls token) if present
        if self.num_prefix_tokens > 0:
            hidden_states = hidden_states[:, self.num_prefix_tokens:, :]
        
        if masks is not None:
            hidden_states = apply_masks(hidden_states, masks)

        return hidden_states
    
    def create_attn_mask(self, 
            x: torch.Tensor, 
            masks: Optional[List[torch.Tensor]] = None
        
        ) -> Optional[torch.Tensor]:
        """Create attention mask for input tensor.
        
        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Attention mask tensor of shape (B, 1, N, N).
        """
        if masks is None:
            return None
        reg_tokens = self.num_prefix_tokens
        b, n, d = x.shape
        masks = [m.cpu() for m in masks]
        zeros = torch.zeros((b, reg_tokens), dtype=torch.bool)
        zeros = [zeros for _ in range(len(masks))]

        adjusted_mask = [m + reg_tokens for m in masks]
        adjusted_mask = [
            torch.cat([z, m], dim=1) for z, m in zip(zeros, adjusted_mask)
        ][0]

        attn_mask = torch.zeros((b, n, n), dtype=torch.bool)
        batch_indices = torch.arange(b).unsqueeze(1)  # Shape (B, 1)
        attn_mask[batch_indices, adjusted_mask] = torch.ones(
            (b, adjusted_mask.shape[1], n),
            dtype=torch.bool,
        )
        attn_mask = attn_mask.transpose(1, 2).unsqueeze(1).to(x.device)

        return attn_mask


class CARLModel(nn.Module):
    """Collaborative Attentive Representation Learning model.
    
    Combines spectral and spatial encoders for multispectral image analysis.
    The model:
    1. Embeds input images into patch representations
    2. Encodes spectral information across wavelengths
    3. Connects spectral embeddings to spatial domain
    4. Applies spatial transformer to capture spatial relationships
    
    Attributes:
        spectral_tf: Spectral encoder for wavelength-aware feature extraction
        spatial_encoder: Vision transformer for spatial feature learning
        embedder: Convolutional patch embedder
        linear_connector: Linear projection between spectral and spatial spaces
    """
    
    def __init__(
        self,
        spec_encoder_kwargs: Dict[str, Any] = {},
        spat_encoder_kwargs: Dict[str, Any] = {},
        patch_size: int = 8,
        ssl_ckpt_path: Optional[Union[str, Path]] = None,
        wave_list: Optional[Dict[str, List[float]]] = None,
        sensor: str = "enmap",
        sensor_config_name: Optional[str] = None,
        img_size: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize CARL model.
        
        Args:
            spec_encoder_kwargs: Configuration for spectral encoder.
            spat_encoder_kwargs: Configuration for spatial encoder.
            patch_size: Size of patches for embedding (e.g., 8 means 8x8 patches).
            ssl_ckpt_path: Optional path to a CARL-SSL checkpoint to load weights from.
            wave_list: Dictionary mapping sensor name to list of wavelength values.
            sensor: Key into wave_list to select the wavelength list.
            sensor_config_name: Optional key into configs/sensor when wave_list is
                not provided. Wavelengths are loaded from the processed band view
                and converted from nm to micrometers, matching CARL pre-training.
            **kwargs: Additional keyword arguments (e.g., n_classes, not used here).
        """
        super().__init__()

        # Store wavelength configuration
        self.wave_list = wave_list
        self.sensor = str(sensor_config_name or sensor).lower()
        if self.wave_list is not None:
            self.wv_list = self.wave_list[self.sensor]
        else:
            sensor_cfg = SensorRegistry.get(self.sensor)
            metadata = load_spectral_metadata(sensor_cfg, view="processed")
            self.wv_list = (metadata.band_centers_nm * 0.001).astype("float32").tolist()

        if self.wv_list is None:
            raise ValueError(
                "CARLModel needs wavelengths. Provide wave_list or sensor_config_name/sensor."
            )

        # Initialize spectral and spatial encoders
        self.spectral_tf = SpectralEncoder(**spec_encoder_kwargs)

        # Patch embedding: convert images to patch embeddings
        self.patch_size = patch_size
        self.img_size = int(img_size) if img_size is not None else None
        self.in_chans = len(self.wv_list)
        self.embedder = nn.Conv2d(
            in_channels=1,
            out_channels=self.spectral_tf.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if spat_encoder_kwargs != {}:
            self.spatial_encoder = TimmWrapper(**spat_encoder_kwargs)

            # Projection layer to align spectral and spatial embedding dimensions
            self.linear_connector = nn.Linear(
                self.spectral_tf.embed_dim,
                self.spatial_encoder.embed_dim,
            )

        # Expose output feature dimension for downstream heads
        self.num_features = self.spatial_encoder.embed_dim if hasattr(self, 'spatial_encoder') else self.spectral_tf.embed_dim

        # Load SSL checkpoint if provided
        self.ssl_ckpt_path = ssl_ckpt_path
        if ssl_ckpt_path is not None:
            self.load_ssl_checkpoint(ssl_ckpt_path)

    @staticmethod
    def _standardize(img: torch.Tensor) -> torch.Tensor:
        """Per-image zero-mean unit-variance standardization (matching CARL pre-training).

        Args:
            img: Tensor of shape (B, C, H, W).

        Returns:
            Standardized tensor of the same shape.
        """
        # Compute mean and std per sample over (C, H, W)
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        std = img.std(dim=(1, 2, 3), keepdim=True)
        return (img - mean) / (std + 1e-6)

    def _encode_spectral(
        self, img: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """Run patch embedding and spectral encoding, returning spatial-ready tokens.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple of (spec_spatial, patch_height, patch_width) where spec_spatial
            has shape (B, H', W', D_spatial).
        """
        # Per-image standardization (matching CARL pre-training)
        # img shape: (B, C, H, W) -> compute mean/std over (C, H, W) per sample
        img = self._standardize(img)

        batch_size, num_channels, height, width = img.shape

        # Build wavelength tensor from stored list
        wavelengths = torch.tensor(self.wv_list, device=img.device).float()
        wavelengths = wavelengths.unsqueeze(0).expand(batch_size, -1)  # (B, C)

        # (B, C, H, W) -> (B*C, 1, H, W)
        img_reshaped = rearrange(img, "b c h w -> (b c) 1 h w")

        # Embed patches: (B*C, 1, H, W) -> (B*C, D, H', W')
        patch_embeddings = self.embedder(img_reshaped)
        patch_height, patch_width = patch_embeddings.shape[-2], patch_embeddings.shape[-1]
        embed_dim = patch_embeddings.shape[1]

        # (B*C, D, H', W') -> (B*H'*W', C, D)
        patch_seq = rearrange(
            patch_embeddings,
            "(b c) d nh nw -> (b nh nw) c d",
            b=batch_size, c=num_channels, nh=patch_height, nw=patch_width, d=embed_dim,
        )

        # Spectral encoding
        spectral_output = self.spectral_tf(patch_seq, wavelengths)
        spec_representations = spectral_output["queries"]

        # Aggregate and normalize
        spec_representations = spec_representations.sum(dim=1)
        spec_representations = F.layer_norm(spec_representations, spec_representations.shape[-1:])

        # Project to spatial encoder dimension
        spec_representations = self.linear_connector(spec_representations)

        # Reshape back to spatial structure for spatial encoder
        # (B*H'*W', D_s) -> (B, H', W', D_s)
        spec_representations = rearrange(
            spec_representations, "(b nh nw) d -> b nh nw d",
            b=batch_size, nh=patch_height, nw=patch_width,
        )
        return spec_representations, patch_height, patch_width

    def forward(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through CARL model.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            spatial_representations of shape (B, D, H', W').
        """
        spec_representations, patch_height, patch_width = self._encode_spectral(img)
        batch_size = spec_representations.shape[0]

        # Spatial encoding: (B, H', W', D_s) -> (B, H'*W', D_s)
        spatial_representations = self.spatial_encoder(spec_representations)

        # (B, H'*W', D_s) -> (B, D_s, H', W')
        spatial_representations = rearrange(
            spatial_representations, "b (h w) d -> b d h w",
            b=batch_size, d=spatial_representations.shape[-1], h=patch_height, w=patch_width,
        )
        return spatial_representations

    def _intermediate_layers(
        self,
        img: torch.Tensor,
        n: Union[int, List[int]] = 1,
    ) -> List[torch.Tensor]:
        """Collect intermediate spatial encoder block outputs.

        Args:
            img: Input image tensor of shape (B, C, H, W).
            n: If int, take last *n* blocks. If list, take blocks at those indices.

        Returns:
            List of tensors of shape (B, N, D) from the selected blocks.
        """
        spec_representations, _, _ = self._encode_spectral(img)

        # Run through the spatial encoder's pos_embed + blocks manually
        net = self.spatial_encoder.net
        x, rot_pos_embed = self.spatial_encoder._pos_embed_tokens(spec_representations)

        block_kwargs = {}
        if rot_pos_embed is not None:
            block_kwargs["rope"] = rot_pos_embed

        num_blocks = len(net.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        outputs = []
        for i, blk in enumerate(net.blocks):
            x = blk(x, **block_kwargs)
            if i in take_indices:
                outputs.append(x)
        return outputs

    def get_intermediate_layers(
        self,
        img: torch.Tensor,
        n: Union[int, List[int]] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.

        Args:
            img: Input image tensor of shape (B, C, H, W).
            n: Which blocks to return (int = last n, list = specific indices).
            reshape: If True reshape tokens to spatial grid (B, D, H', W').
            return_prefix_tokens: If True return prefix (cls) tokens alongside.
            norm: If True apply the spatial encoder's final norm.

        Returns:
            Tuple of output tensors from selected blocks.
        """
        net = self.spatial_encoder.net
        num_prefix = self.spatial_encoder.num_prefix_tokens

        outputs = self._intermediate_layers(img, n)

        if norm:
            outputs = [net.norm(out) for out in outputs]

        prefix_tokens = [out[:, :num_prefix] for out in outputs]
        outputs = [out[:, num_prefix:] for out in outputs]

        if reshape:
            _, _, h, w = img.shape
            gh = h // self.patch_size
            gw = w // self.patch_size
            outputs = [
                out.reshape(img.shape[0], gh, gw, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def load_ssl_checkpoint(self, ckpt_path: Union[str, Path]) -> None:
        """Load CARL-SSL checkpoint.

        Handles both raw state_dicts and Lightning-style checkpoints with a
        ``state_dict`` key.  Strips common prefixes like ``model.`` and
        ``module.`` that training wrappers often add.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading CARL SSL checkpoint from {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            return

        state_dict = ckpt.get("state_dict", ckpt)

        new_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            new_state[new_k] = v

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        logger.info(f"CARL SSL checkpoint loaded. Missing keys: {missing}")
        logger.info(f"CARL SSL checkpoint loaded. Unexpected keys: {unexpected}")
