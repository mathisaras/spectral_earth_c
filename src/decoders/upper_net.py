# Adapted from https://github.com/VMarsocci/pangaea-bench/blob/main/pangaea/decoders/upernet.py
# Above implementation is based on https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py

from typing import Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



class UperNet(nn.Module):
    """
    UperNet decoder for semantic segmentation tasks.

    Args:
        channels: The number of output channels for the PSP Module (hparam).
        output_layers: The intermediate layers from which to extract the output.
        pool_scales: average pooling scales used in PSP module (hparam).
        feature_multiplier: scales the output dims of the encoder's intermediate layers (hparam)
        in_channels: input channel dimensions of each intermediate layer from the encoder.
                     if not provided: calculated based on encoder's output dims and feature multiplier
        pyramid_output = if True, encoder outputs a pyramid of features at different scales
        upscale_method: either "pixel_shuffle" or "bilinear_interpolation"
        num_classes: number of classes for segmentation
    """

    def __init__(
        self,
        channels: int = 512,
        output_layers: Union[int, List[int]] = [3, 5, 7, 11],
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
        in_channels: Optional[List[int]] = None,
        pyramid_output: bool = False,
        upscale_method: str = 'bilinear_interpolation',
        num_classes: int = 1,
    ):
        super().__init__()

        # Convert OmegaConf ListConfig to Python list if needed
        if hasattr(output_layers, '_content'):
            self.input_layers = list(output_layers)
        else:
            self.input_layers = output_layers if isinstance(output_layers, list) else [output_layers]
        self.input_layers_num = len(self.input_layers)

        # TODO: Get output_dim from encoder, when None is passed
        if in_channels is None:
            # This will be set by the model when it has access to the encoder
            self.in_channels = None
        else:
            self.in_channels = [dim * feature_multiplier for dim in in_channels]

        if pyramid_output:
            rescales = [1 for _ in range(self.input_layers_num)]
        else:
            scales = [4, 2, 1, 0.5]
            rescales = [
                scales[int(i / self.input_layers_num * 4)]
                for i in range(self.input_layers_num)
            ]

        # Initialize neck later when we know the embed_dim
        self.rescales = rescales
        self.neck = None

        self.align_corners = False

        self.channels = channels
        self.num_classes = num_classes

        # PSP Module - will be initialized later
        # Convert OmegaConf ListConfig to Python list if needed
        if hasattr(pool_scales, '_content'):
            self.pool_scales = list(pool_scales)
        else:
            self.pool_scales = pool_scales
        self.psp_modules = None

        # Bottleneck - will be initialized later
        self.bottleneck = None

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # These will be initialized when we know the in_channels
        self._lateral_convs_initialized = False
        self._fpn_convs_initialized = False

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.input_layers) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

        self.upscale_method = upscale_method

    def _initialize_lateral_convs(self, in_channels_list):
        """Initialize lateral convolutions when we know the input channels."""
        if self._lateral_convs_initialized:
            return
            
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    padding=0,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
        self._lateral_convs_initialized = True

    def _initialize_psp_and_bottleneck(self, in_channels_list):
        """Initialize PSP modules and bottleneck when we know the input channels."""
        if self.in_channels is not None:
            return
            
        self.in_channels = in_channels_list
        
        # Initialize neck
        self.neck = Feature2Pyramid(embed_dim=self.in_channels, rescales=self.rescales)
        
        # Initialize PSP modules with correct in_channels
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        
        # Initialize bottleneck with correct in_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(self.pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )
        


    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, features, output_shape=None):
        """
        Forward pass for UperNet decoder.
        
        Args:
            features: List of feature tensors from different layers of the backbone
            output_shape: Optional output shape for upsampling
            
        Returns:
            Segmentation logits
        """
        # Store device and dtype from input features for proper initialization
        if len(features) > 0:
            self._device = features[0].device
            self._dtype = features[0].dtype
        
        # Initialize components if not done yet
        if self.in_channels is None:
            # Assume all inputs have the same number of channels
            in_channels_list = [features[i].shape[1] for i in range(len(features))]
            self._initialize_psp_and_bottleneck(in_channels_list)
            self._initialize_lateral_convs(in_channels_list)
            
            # Move the entire module to the correct device and dtype
            if hasattr(self, '_device'):
                self.to(device=self._device, dtype=self._dtype)
        
        # Pass the concatenated features through the neck
        feat = self.neck(features)

        # Pass the concatenated features through the UperNet
        feat = self._forward_feature(features)
        feat = self.dropout(feat)

        # Use pixel shuffle or bilinear interpolation for upsampling
        if self.upscale_method == "bilinear_interpolation":
            seg_logit = self.conv_seg(feat)
            if output_shape is not None:
                seg_logit = F.interpolate(seg_logit, size=output_shape, mode="bilinear", align_corners=self.align_corners)
        else:
            raise ValueError(f"Invalid upscale method: {self.upscale_method}")

        return seg_logit


class RegUPerNet(nn.Module):
    """
    UperNet customized for regression tasks.

    Args:
        channels: The number of output channels for the PSP Module (hparam).
        output_layers: The intermediate layers from which to extract the output.
        pool_scales: average pooling scales used in PSP module (hparam).
        feature_multiplier: scales the output dims of the encoder's intermediate layers (hparam)
        in_channels: input channel dimensions of each intermediate layer from the encoder.
                     if not provided: calculated based on encoder's output dims and feature_multiplier
        pyramid_output = if True, encoder outputs a pyramid of features at different scales
        upscale_method: either "pixel_shuffle" or "bilinear_interpolation"
    """

    def __init__(
        self,
        channels: int = 512,
        output_layers: Union[int, List[int]] = [3, 5, 7, 11],
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
        in_channels: Optional[List[int]] = None,
        pyramid_output: bool = False,
        upscale_method: str = 'bilinear_interpolation',
    ):
        super().__init__()

        # Convert OmegaConf ListConfig to Python list if needed
        if hasattr(output_layers, '_content'):
            self.input_layers = list(output_layers)
        else:
            self.input_layers = output_layers if isinstance(output_layers, list) else [output_layers]
        self.input_layers_num = len(self.input_layers)

        # TODO: Get output_dim from encoder, when None is passed
        if in_channels is None:
            self.in_channels = [
                dim * feature_multiplier for dim in self.encoder.output_dim
            ]
        else:
            self.in_channels = [dim * feature_multiplier for dim in in_channels]

        if pyramid_output:
            rescales = [1 for _ in range(self.input_layers_num)]
        else:
            scales = [4, 2, 1, 0.5]
            rescales = [
                scales[int(i / self.input_layers_num * 4)]
                for i in range(self.input_layers_num)
            ]

        self.neck = Feature2Pyramid(embed_dim=self.in_channels, rescales=rescales)

        self.align_corners = False

        self.channels = channels
        self.num_classes = 1  # regression

        # PSP Module
        # Convert OmegaConf ListConfig to Python list if needed
        if hasattr(pool_scales, '_content'):
            pool_scales_list = list(pool_scales)
        else:
            pool_scales_list = pool_scales
            
        self.psp_modules = PPM(
            pool_scales_list,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    padding=0,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.in_channels) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv_reg = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

        self.upscale_method = upscale_method


    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, img, output_shape=None):
        # Pass the concatenated features through the neck
        feat = self.neck(img)

        # Pass the concatenated features through the RegUPerNet
        feat = self._forward_feature(img)
        feat = self.dropout(feat)

        # Use pixel shuffle or bilinear interpolation for upsampling
        if self.upscale_method == "bilinear_interpolation":
            reg_logit = self.conv_reg(feat)
            output = torch.relu(reg_logit)
            output = F.interpolate(output, size=output_shape, mode="bilinear")
        else:
            raise ValueError(f"Invalid upscale method: {self.upscale_method}")

        return output


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
    """

    def __init__(
        self,
        embed_dim,
        rescales=(4, 2, 1, 0.5),
    ):
        super().__init__()
        # Convert OmegaConf ListConfig to Python list if needed
        if hasattr(rescales, '_content'):
            self.rescales = list(rescales)
        else:
            self.rescales = rescales
            
        # Convert embed_dim to list if it's not already
        if hasattr(embed_dim, '_content'):
            self.embed_dim = list(embed_dim)
        else:
            self.embed_dim = embed_dim if isinstance(embed_dim, list) else [embed_dim]
            
        self.upsample_4x = None
        self.ops = nn.ModuleList()

        for i, k in enumerate(self.rescales):
            # Get the embedding dimension for this layer
            if i < len(self.embed_dim):
                dim = self.embed_dim[i]
            else:
                # Fallback to the last dimension if we have more rescales than embed_dims
                dim = self.embed_dim[-1]
                
            if k == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            dim, dim, kernel_size=2, stride=2
                        ),
                        nn.SyncBatchNorm(dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            dim, dim, kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            dim, dim, kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif k == 0.25:
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return tuple(outputs)