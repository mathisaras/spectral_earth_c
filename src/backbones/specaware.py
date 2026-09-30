# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_

# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import numpy as np

import torch
from torch import Tensor

from src.utils.sensor_registry import SensorRegistry
from src.utils.spectral_metadata import load_spectral_metadata



# Scale 2D position encoding
# https://github.com/microsoft/torchgeo/blob/main/torchgeo/models/scale_mae.py


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int, grid_size: int, res: Tensor, cls_token: bool = False
) -> Tensor:
    """Generate spatial resolution specific 2D positional embeddings.

    Args:
        embed_dim: Dimension of the positional embeddings.
        grid_size: Height (ph) and width (pw) of the image patches.
        res: Spatial resolution tensor of shape (N,) of the image.
        cls_token: Increase positional embedding size by 1 for class token.

    Returns:
        pos_embed: Spatial resolution aware positional embeddings (Ph * Pw, D).
    """
    device, dtype = res.device, res.dtype
    # res = torch.tensor(res, dtype=torch.float32, device=device)
    grid_h = torch.arange(grid_size, dtype=dtype, device=device)
    grid_w = torch.arange(grid_size, dtype=dtype, device=device)
    grid: Tensor = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    grid = torch.einsum('chw,n->cnhw', grid, res)
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, embed_dim], dtype=dtype, device=device), pos_embed],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: Tensor) -> Tensor:
    """Generate 2D sin-cos positional embedding from grid.

    Args:
        embed_dim: Dimension of the positional embeddings.
        grid: Tensor representing the image patch grid (C, N, Ph, Pw)

    Returns:
        emb: 2D sin-cos positional embeddings (Ph * Pw, D).
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb



# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import torch

__all__ = ["area", "compute_patch_areas", "radius_earth"]


radius_earth = 6378137 / 1000
"""float: Radius of the earth in kilometers."""


def area(polygon: torch.Tensor) -> torch.Tensor:
    """Compute the area of a polygon specified by latitudes and longitudes in degrees.

    This function is a PyTorch port of the PyPI package `area`. In particular, it is heavily
    inspired by the following file:

        https://github.com/scisco/area/blob/9d9549d6ebffcbe4bffe11b71efa2d406d1c9fe9/area/__init__.py

    Args:
        polygon (:class:`torch.Tensor`): Polygon of the shape `(*b, n, 2)` where `b` is an optional
            multidimensional batch size, `n` is the number of points of the polygon, and 2
            concatenates first latitudes and then longitudes. The polygon does not have be closed.

    Returns:
        :class:`torch.Tensor`: Area in square kilometers.
    """
    # Be sure to close the loop.
    polygon = torch.cat((polygon, polygon[..., -1:, :]), axis=-2)

    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]  # Number of points of the polygon

    rad = torch.deg2rad  # Convert degrees to radians.

    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n

            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]

            area = area + (rad(lon_upper) - rad(lon_lower)) * torch.sin(rad(lat_middle))

    area = area * radius_earth * radius_earth / 2

    return torch.abs(area)


def expand_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Expand matrix by adding one row and one column to each side, using
    linear interpolation.

    Args:
        matrix (:class:`torch.Tensor`): Matrix to expand.

    Returns:
        :class:`torch.Tensor`: `matrix`, but with two extra rows and two extra columns.
    """
    # Add top and bottom rows.
    matrix = torch.cat(
        (
            2 * matrix[0:1] - matrix[1:2],
            matrix,
            2 * matrix[-1:] - matrix[-2:-1],
        ),
        dim=0,
    )

    # Add left and right columns.
    matrix = torch.cat(
        (
            2 * matrix[:, 0:1] - matrix[:, 1:2],
            matrix,
            2 * matrix[:, -1:] - matrix[:, -2:-1],
        ),
        dim=1,
    )

    return matrix


def compute_patch_areas(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """A pair of latitude and longitude matrices defines a number non-intersecting patches on the
    Earth. For a global grid, these patches span the entire surface of the Earth. For a local grid,
    the patches might span only a country or a continent. This function computes the area of every
    specified patch.

    To divide the Earth into patches, the idea is to let a grid point be the _center_ of the
    corresponding patch. The vertices of this patch will then sit exactly inbetween the grid
    point and the grid points immediately diagonally and non-diagonally above, below, left, and
    right. For a grid point at the very top of the grid, for example, there is no immediately above
    grid point. In that case, we enlarge the grid by a row at the top by linearly interpolating the
    latitudinal progression.

    Summary of algorithm:
    1. Enlarge the latitude and longitude matrices by adding one row and one column to each side.
    2. Calculate the patch vertices by averaging every 2x2 square in the enlarged grid. We also
        call these points the midpoints.
    3. By using the vertices of the patches, i.e. the midpoints, compute the areas of the patches.

    Args:
        lat (:class:`torch.Tensor`): Latitude matrix. Must be decreasing along rows.
        lon (:class:`torch.Tensor`): Longitude matrix. Must be increasing along columns.

    Returns:
        :class:`torch.Tensor`: Areas in square kilometer.
    """
    if not (lat.dim() == lon.dim() == 2):
        raise ValueError("`lat` and `lon` must both be matrices.")
    if lat.shape != lat.shape:
        raise ValueError("`lat` and `lon` must have the same shape.")

    # Check that the latitude matrix is decreasing in the appropriate way.
    if not torch.all(lat[1:] - lat[:-1] <= 0):
        raise ValueError("`lat` must be decreasing along rows.")

    # Check that the longitude matrix is increasing in the appropriate way.
    if not torch.all(lon[:, 1:] - lon[:, :-1] >= 0):
        raise ValueError("`lon` must be increasing along columns.")

    # Enlarge the latitude and longitude matrices for the midpoint computation.
    lat = expand_matrix(lat)
    lon = expand_matrix(lon)

    # Latitudes cannot expand beyond the poles.
    lat = torch.clamp(lat, -90, 90)

    # Calculate midpoints between entries in lat/lon. This is very important for symmetry of the
    # resulting areas.
    lat_midpoints = (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:]) / 4
    lon_midpoints = (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:]) / 4

    # Determine squares and return the area of those squares.
    top_left = torch.stack((lat_midpoints[1:, :-1], lon_midpoints[1:, :-1]), dim=-1)
    top_right = torch.stack((lat_midpoints[1:, 1:], lon_midpoints[1:, 1:]), dim=-1)
    bottom_left = torch.stack((lat_midpoints[:-1, :-1], lon_midpoints[:-1, :-1]), dim=-1)
    bottom_right = torch.stack((lat_midpoints[:-1, 1:], lon_midpoints[:-1, 1:]), dim=-1)
    polygon = torch.stack((top_left, top_right, bottom_right, bottom_left), dim=-2)

    return area(polygon)

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "FourierExpansion",
    "pos_expansion",
    "scale_expansion",
    "lead_time_expansion",
    "levels_expansion",
    "absolute_time_expansion",
]


class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower (float): Lower wavelength.
            upper (float): Upper wavelength.
            assert_range (bool, optional): Assert that the encoded tensor is within the specified
                wavelength range. Defaults to `True`.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`torch.Tensor`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            torch.Tensor: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(self.lower <= x.abs(), torch.all(x.abs() <= self.upper))
        in_range_or_zero = torch.all(
            torch.logical_or(in_range, x == 0)
        )  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range"
                f" `[{self.lower}, {self.upper}]`."
            )

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.


# Determine a reasonable smallest value for the scale embedding by assuming a smallest delta in
# latitudes and longitudes.
_delta = 0.01  # Reasonable smallest delta in latitude and longitude
_min_patch_area: float = area(
    torch.tensor(
        [
            # The smallest patches will be at the poles. Just use the north pole.
            [90, 0],
            [90, _delta],
            [90 - _delta, _delta],
            [90 - _delta, 0],
        ],
        dtype=torch.float64,
    )
).item()
_area_earth = 4 * np.pi * radius_earth * radius_earth

pos_expansion = FourierExpansion(_delta, 720)
""":class:`.FourierExpansion`: Fourier expansion for the encoding of latitudes and longitudes in
degrees."""

scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
""":class:`.FourierExpansion`: Fourier expansion for the encoding of patch areas in squared
kilometers."""

lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)
""":class:`.FourierExpansion`: Fourier expansion for the lead time encoding in hours."""

levels_expansion = FourierExpansion(0.01, 1e5)
""":class:`.FourierExpansion`: Fourier expansion for the pressure level encoding in hPa."""

absolute_time_expansion = FourierExpansion(1, 24 * 365.25, assert_range=False)
""":class:`.FourierExpansion`: Fourier expansion for the absolute time encoding in hours."""

### new for SSL4EO-S ###
# min wavelength: ultraviolet light (100 nm)
# max wavelength: radio waves (1 m)
spectrum_central_expansion = FourierExpansion(1e-7, 1)
""":class:`.FourierExpansion`: Fourier expansion for the spectrum central wavelength encoding."""

# min bandwidth: 10nm
# max bandwidth: 1m
spectrum_width_expansion = FourierExpansion(1e-7, 1)
""":class:`.FourierExpansion`: Fourier expansion for the spectrum bandwidth encoding."""


### new for HyperSpectral Embedding ###
### NASA AVIRIS wavelength from 350-2550
hyperspectral_expansion = FourierExpansion(350, 2550)
""":class:`.FourierExpansion`: Fourier expansion for the hyperspectral encoding."""


class SentenceEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(SentenceEncoder, self).__init__()
        self.embedding_dim = self.get_embedding_dim(model_name)
        if SentenceTransformer is None:
            self.model = None
            print(
                "Warning: sentence_transformers is not installed. "
                "SpecAware attribute embeddings will use deterministic fallback "
                "vectors for lightweight checks; install sentence-transformers for "
                "faithful SpecAware runs."
            )
        else:
            self.model = SentenceTransformer(model_name)

            for param in self.model.parameters():
                param.requires_grad = False

    def get_embedding_dim(self, model_name):
        if model_name == "sentence-transformers/all-MiniLM-L6-v2":
            return 384
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, sentences):
        if self.model is None:
            text = str(sentences)
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn(self.embedding_dim, generator=generator) * 0.02
        with torch.no_grad():
            return self.model.encode(sentences, convert_to_tensor=True)


class AttributeEncoder(nn.Module):
    def __init__(self, embedding_dim=32, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(AttributeEncoder, self).__init__()
        self.sentence_encoder = SentenceEncoder(model_name)

        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(self.sentence_encoder.embedding_dim, embedding_dim)
        self.fc.apply(self.weight_init)

        self.processed_cache = {}
        self._preprocess_sentences()
        
        del self.sentence_encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _preprocess_sentences(self, sentences=None):
        if isinstance(sentences, str):
            sentences = [sentences]
        elif sentences is None:
            sentences = {"avc": "AVIRIS-Classic",
                         "avng": "AVIRIS-NG",
                         "av3": "AVIRIS-3",
                         "L1": "L1 Calibrated Radiance",
                         "L2": "L2 Surface Reflectance",
                         "others": "other HSI sensors",
                         }
                         
        for key, sentence in sentences.items():
            with torch.no_grad():
                embedding = self.sentence_encoder(sentence)
                self.processed_cache[key] = embedding.detach()

    def forward(self, sentence):
        if sentence in self.processed_cache:
            sentence_embedding = self.processed_cache[sentence].detach()
        else:
            raise ValueError(f"Sentence {sentence} not found in cache")

        sentence_embedding = sentence_embedding.to(self.fc.weight.device)
        sentence_embedding = self.fc(sentence_embedding)

        return sentence_embedding


class MLPResLayer(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, hidden_dim=None):
        super(MLPResLayer, self).__init__()
        hidden_dim = in_dim * 2 if hidden_dim is None else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
            self.proj.apply(self.weight_init)
        else:
            self.proj = nn.Identity()

        self.mlp.apply(self.weight_init)
        
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.mlp(x) + self.proj(x)
        


class FWHMEncoder(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, min_fwhm=5.0, max_fwhm=15.0):
        super(FWHMEncoder, self).__init__()
        self.min_fwhm = min_fwhm
        self.max_fwhm = max_fwhm
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.mlp.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, fwhm):
        # 1. min-max normalize fwhm to [0, 1]
        normed = (fwhm - self.min_fwhm) / (self.max_fwhm - self.min_fwhm)
        # 2. encode fwhm to embedding
        normed = normed.unsqueeze(-1)  # [B, C] -> [B, C, 1]
        fwhm_embedding = self.mlp(normed)
        return fwhm_embedding


class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # projection
        self.modal1_proj = nn.Linear(dim, dim)
        self.modal2_proj = nn.Linear(dim, dim)
        
        # fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.use_gating = False
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Sigmoid()
            )
    
    def forward(self, feat1, feat2):
        """
        feat1, feat2: [B, C, dim]
        """
        f1 = self.modal1_proj(feat1)
        f2 = self.modal2_proj(feat2)
        
        combined = torch.cat([f1, f2], dim=-1)
        
        fused = self.fusion(combined)
        
        if self.use_gating:
            gate_weight = self.gate(combined)
            output = gate_weight * fused + (1 - gate_weight) * f1
        else:
            output = fused + f1
        
        return output


class SpectralEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SpectralEmbedding, self).__init__()
        # in this experiment, we just use the VNIR band and SWIR band from 350-2550 nm
        self.wavelength_encoder = FourierExpansion(350.0, 2550.0)
        self.fwhm_encoder = FWHMEncoder(embedding_dim)

        self.attribute_encoder = AttributeEncoder(embedding_dim//2)
        self.embedding_dim = embedding_dim
        
        self.fusion_weights = nn.Parameter(torch.tensor([0.8, 0.2]))
        
        self.spectral_enhancement_mlp = MLPResLayer(embedding_dim, embedding_dim)
        
        self.cross_modal_fusion = CrossModalFusion(embedding_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, wavelengths, fwhm, sensor_name, data_level):
        if isinstance(sensor_name, list):
            sensor_name = str(list(set(sensor_name))[0])
        if isinstance(data_level, list):
            data_level = str(list(set(data_level))[0])
            
        # spectral feature encoding
        wavelengths_embedding = self.wavelength_encoder(wavelengths, self.embedding_dim)
        fwhm_embedding = self.fwhm_encoder(fwhm)
        
        weights = F.softmax(self.fusion_weights, dim=0)
        spectral_embedding = (weights[0] * wavelengths_embedding + 
                            weights[1] * fwhm_embedding)  # [B, C, 128]
        spectral_embedding = self.spectral_enhancement_mlp(spectral_embedding)
        
        # attribute feature encoding
        sensor_name_embedding = self.attribute_encoder(sensor_name).repeat(
            wavelengths_embedding.shape[0], wavelengths_embedding.shape[1], 1)
        data_level_embedding = self.attribute_encoder(data_level).repeat(
            wavelengths_embedding.shape[0], wavelengths_embedding.shape[1], 1)
        sensor_embedding = torch.cat([sensor_name_embedding, data_level_embedding], dim=-1)
        
        # fusion
        final_embedding = self.cross_modal_fusion(spectral_embedding, sensor_embedding)
        final_embedding = final_embedding + spectral_embedding
        
        return final_embedding


class SpectralAwareTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=1024, num_heads=4, num_layers=1):
        super().__init__()
        self.input_dim = input_dim

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            dim_feedforward=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=True,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, spectral_features):
        ### input: [B, C, wv_planes]
        output = self.transformer_encoder(spectral_features) + spectral_features
        
        return output


# Adaptive HyperEmbedding for Multi-sensor HSI via HyperNet
# Some codes from DOFA and Copernicus-FM, many thanks!

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class MatrixGenerator(nn.Module):
    # Generate U, V matrices
    def __init__(self, wv_planes=128, embed_dim=1024, kernel_size=8, rank=64):
        super().__init__()
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.patch_dim = kernel_size * kernel_size
        self.rank = rank

        self.u_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(wv_planes, wv_planes//2),
            nn.GELU(),
            nn.Linear(wv_planes//2, embed_dim * rank)  # [E*r]
        )

        self.v_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(wv_planes, wv_planes // 2),
            nn.GELU(),
            nn.Linear(wv_planes // 2, rank * self.patch_dim)  # [r*k²]
        )

        # bias
        self.bias_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes // 2),
            nn.GELU(),
            nn.Linear(wv_planes // 2, embed_dim)  # E
        )


    def forward(self, spectral_features):
        # input: [B, C, wv_planes]
        B, C, _ = spectral_features.shape

        u_params = self.u_generator(spectral_features)
        v_params = self.v_generator(spectral_features)

        U_matrices = u_params.view(B, C, self.embed_dim, self.rank)
        V_matrices = v_params.view(B, C, self.rank, self.patch_dim)

        sample_features = spectral_features.mean(dim=1)  # [B, wv_planes]
        bias = self.bias_generator(sample_features)  # [B, embed_dim]

        return U_matrices, V_matrices, bias


class ContentFeatureExtractor(nn.Module):
    # Extract content features from HSI patch simply
    def __init__(self, img_size=224, kernel_size=16, wv_planes=128):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes

        # patch pooling
        self.patch_pool_avg = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
        self.patch_pool_max = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
        expected_patch_num = (img_size // kernel_size) ** 2  # 224//8=28

        self.avg_feature_encoder = nn.Sequential(
            nn.Linear(expected_patch_num, expected_patch_num // 4),
            nn.GELU(),
            nn.Linear(expected_patch_num // 4, wv_planes)
        )

        self.max_feature_encoder = nn.Sequential(
            nn.Linear(expected_patch_num, expected_patch_num // 4),
            nn.GELU(),
            nn.Linear(expected_patch_num // 4, wv_planes)
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(wv_planes * 2, wv_planes * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(wv_planes * 2, wv_planes),
        )

        self.layer_norm = nn.LayerNorm(wv_planes)

    def forward(self, x):
        B, C, H, W = x.shape

        x_pool_avg = self.patch_pool_avg(x) 
        x_pool_max = self.patch_pool_max(x) 

        x_avg_flat = x_pool_avg.view(B, C, -1)
        x_max_flat = x_pool_max.view(B, C, -1)

        avg_features = self.avg_feature_encoder(x_avg_flat)
        max_features = self.max_feature_encoder(x_max_flat)

        combined_features = torch.cat([avg_features, max_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.layer_norm(fused_features)

        return fused_features




class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # projection
        self.modal1_proj = nn.Linear(dim, dim)
        self.modal2_proj = nn.Linear(dim, dim)
        
        # fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.use_gating = False
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Sigmoid()
            )
    
    def forward(self, feat1, feat2):
        """
        feat1, feat2: [B, C, dim]
        """
        f1 = self.modal1_proj(feat1)
        f2 = self.modal2_proj(feat2)
        
        combined = torch.cat([f1, f2], dim=-1)
        
        fused = self.fusion(combined)
        
        if self.use_gating:
            gate_weight = self.gate(combined)
            output = gate_weight * fused + (1 - gate_weight) * f1
        else:
            output = fused + f1
        
        return output


class HyperEmbedding(nn.Module):
    def __init__(self, img_size=224, wv_planes=128, kernel_size=16, num_heads=4,
                 num_layers=1, embed_dim=768, rank=64):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.rank = rank

        # spectral embedding
        self.spectral_embedding = SpectralEmbedding(wv_planes)

        self.spectral_aware_transformer = SpectralAwareTransformer(
            input_dim=wv_planes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.content_feature_extractor = ContentFeatureExtractor(
            img_size=img_size,
            kernel_size=kernel_size,
            wv_planes=wv_planes
        )

        self.matrix_generator = MatrixGenerator(
            wv_planes=wv_planes,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            rank=rank
        )

        self.feature_fusion = CrossModalFusion(wv_planes)
        self.scaler = nn.Parameter(torch.tensor(0.02))
        
        self._init_weights()


    def _init_weights(self):
        bias_gen_output_head = self.matrix_generator.bias_generator[-1]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is bias_gen_output_head:
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


    def matrix_computation(self, patches, U_matrices, V_matrices):
        """
        patches: [B, N, C×k²]
        U_matrices: [B, C, embed_dim, rank]  
        V_matrices: [B, C, rank, k²]
        return: [B, N, embed_dim]
        """
        B, N, _ = patches.shape
        C = U_matrices.shape[1]
        patch_dim = self.kernel_size * self.kernel_size
        
        patches_reshaped = patches.view(B, N, C, patch_dim)
    
        intermediate = torch.einsum('bncp,bcrp->bncr', 
                                    patches_reshaped, 
                                    V_matrices)
        
        channel_outputs = torch.einsum('bncr,bcer->bnce', 
                                        intermediate, 
                                        U_matrices)
        
        output = channel_outputs.sum(dim=2)  # [B, N, embed_dim]
    
        return output

    def forward(self, x, wavelengths, fwhm, sensor_name, data_level):
        B, C, H, W = x.shape

        if isinstance(wavelengths, np.ndarray):
            wavelengths = torch.from_numpy(wavelengths).float()

        # Spectral Embedding
        meta_embedding = self.spectral_embedding(wavelengths, fwhm, sensor_name, data_level)
        enhanced_meta_embedding = self.spectral_aware_transformer(meta_embedding)

        # Content Feature Extraction
        content_features = self.content_feature_extractor(x)

        # Fusion
        enhanced_features = self.feature_fusion(
            enhanced_meta_embedding, content_features) + enhanced_meta_embedding

        # Unfold
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.kernel_size)
        patches = patches.permute(0, 2, 1)

        # Matrix Computation
        U_matrices, V_matrices, bias = self.matrix_generator(enhanced_features)
        output = self.matrix_computation(patches, U_matrices, V_matrices)  # [B, N, embed_dim]

        output = output + bias.unsqueeze(1)
        output = output * self.scaler
        
        return output, enhanced_meta_embedding


class MaskedHSIAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        cls_embed=True,
        drop_path_rate=0.,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.cls_embed = cls_embed
        self.embed_dim = embed_dim
    
    
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = HyperEmbedding(
            img_size=img_size,
            wv_planes=128,
            kernel_size=patch_size,
            embed_dim=embed_dim
        )

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim,
                  num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  drop_path=dpr[i],
                  norm_layer=norm_layer) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        # HyperMAE encoder END--------------------------------------------------------------------------



    def forward(
        self,
        imgs,
        GSD=None,
        wavelength=None,
        fwhm=None,
        sensor_name=None,
        data_level=None,
        feature_idx=None,
    ):
        B, C, H, W = imgs.shape

        if GSD is None:
            GSD = torch.tensor([10], device=imgs.device) * torch.ones(imgs.shape[0], device=imgs.device)
        if len(GSD.shape) == 2:
            GSD = GSD.squeeze(1)
        if len(wavelength.shape) == 1:
            wavelength = wavelength.unsqueeze(0).repeat(B, 1)
        if len(fwhm.shape) == 1:
            fwhm = fwhm.unsqueeze(0).repeat(B, 1)

        x, _ = self.patch_embed(imgs, wavelength, fwhm, sensor_name, data_level)

        # get 2d pos embed
        assert len(GSD.shape) == 1, f"GSD shape: {GSD.shape}"
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            self.embed_dim,
            H // self.patch_size,
            GSD.float(),
            cls_token=True)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :].type_as(x).to(
            x.device).clone().detach()

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token + pos_embed[:, :1, :].type_as(
                x).to(x.device).clone().detach()
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        features = []
        feature_idx = [2, 5, 8, 11] if feature_idx is None else [int(i) for i in feature_idx]
        # feature_idx = [5, 11, 17, 23]
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in feature_idx:
                if self.cls_embed:
                    out = x[:, 1:, :] 
                else:
                    out = x
                B, L, D = out.shape
                H = W = int(L ** 0.5)
                out = out.permute(0, 2, 1).reshape(B, D, H, W)
                features.append(out)

        return features, x



def mae_vit_base_patch8_hsi(**kwargs):
    model = MaskedHSIAutoencoderViT(embed_dim=768,
                                    depth=12,
                                    num_heads=12,
                                    mlp_ratio=4,
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    **kwargs)
    return model


mae_vit_base_patch8 = mae_vit_base_patch8_hsi


class SpecAwareBackbone(nn.Module):
    """SpecAware encoder wrapper for downstream segmentation.

    The copied SpecAware implementation expects sensor metadata at every
    forward call.  This wrapper binds one processed sensor view from the current
    sensor configs and exposes the same `get_intermediate_layers` style used by
    the other ViT-like backbones in this repository.
    """

    def __init__(
        self,
        sensor: str = "enmap",
        sensor_config_name: Optional[str] = None,
        img_size: int = 224,
        patch_size: int = 8,
        pretrained: Optional[Union[str, Path]] = None,
        sensor_name: str = "others",
        data_level: str = "L2",
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        self.sensor = str(sensor_config_name or sensor).lower()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.sensor_name = str(sensor_name)
        self.data_level = str(data_level)

        sensor_cfg = SensorRegistry.get(self.sensor)
        metadata = load_spectral_metadata(sensor_cfg, view="processed")
        self.in_chans = metadata.num_bands
        self.num_features = int(encoder_kwargs.get("embed_dim", 1024))

        self.register_buffer(
            "wavelength_nm",
            torch.tensor(metadata.band_centers_nm, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "fwhm_nm",
            torch.tensor(metadata.band_widths_nm, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "gsd",
            torch.tensor(float(sensor_cfg.get("resolution", 1.0)), dtype=torch.float32),
            persistent=False,
        )

        self.encoder = MaskedHSIAutoencoderViT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            **encoder_kwargs,
        )

        if pretrained:
            self.load_pretrained(pretrained)

    def _metadata_kwargs(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        return {
            "GSD": self.gsd.to(device=device).repeat(batch_size),
            "wavelength": self.wavelength_nm.to(device=device),
            "fwhm": self.fwhm_nm.to(device=device),
            "sensor_name": [self.sensor_name] * batch_size,
            "data_level": [self.data_level] * batch_size,
        }

    def _forward_features(
        self,
        imgs: torch.Tensor,
        feature_idx: Optional[Sequence[int]] = None,
    ) -> list[torch.Tensor]:
        features, _ = self.encoder(
            imgs,
            **self._metadata_kwargs(batch_size=imgs.shape[0], device=imgs.device),
            feature_idx=feature_idx,
        )
        return list(features)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self._forward_features(imgs)[-1]

    def get_intermediate_layers(
        self,
        imgs: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ):
        if n is None:
            requested = None
        elif isinstance(n, int):
            depth = len(self.encoder.blocks)
            requested = list(range(depth - n, depth)) if n > 0 else None
        else:
            requested = [int(idx) for idx in n]

        selected = self._forward_features(imgs, feature_idx=requested)

        if not reshape:
            selected = [
                feature.flatten(2).transpose(1, 2).contiguous()
                for feature in selected
            ]

        if return_prefix_tokens:
            return tuple((feature, None) for feature in selected)
        return tuple(selected)

    def load_pretrained(self, ckpt_path: Union[str, Path]) -> None:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "model.", "encoder."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value
        msg = self.encoder.load_state_dict(cleaned, strict=False)
        print(f"SpecAware checkpoint loaded from {ckpt_path}: {msg}")
