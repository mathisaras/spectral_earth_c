"""Decoder modules for various tasks."""

from .conv_head import ConvHead
from .fcn_head import FCNHead
from .linear_head import LinearHead
from .lightweight_multitap_seg_head import LightweightMultiTapSegHead
from .mlp_head import MLPHead

__all__ = [
    'ConvHead',
    'FCNHead',
    'LightweightMultiTapSegHead',
    'LinearHead',
    'MLPHead',
] 
