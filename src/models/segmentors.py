"""Segmentor classes that combine backbones and decoders."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Mapping, List, Optional, Sequence, Union
import torch


ModelInput = Union[Tensor, Mapping[str, Tensor]]


def _as_hw(value) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Cannot infer spatial size from an empty sequence.")
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _input_spatial_size(
    x: ModelInput,
    output_size: Optional[Sequence[int]] = None,
) -> tuple[int, int]:
    if output_size is not None:
        return _as_hw(output_size)
    if isinstance(x, Tensor):
        return int(x.shape[-2]), int(x.shape[-1])
    for value in x.values():
        if isinstance(value, Tensor):
            return int(value.shape[-2]), int(value.shape[-1])
    raise ValueError("Cannot infer spatial size from an empty sensor input dict.")


def _match_input_spatial_size(
    output: Tensor,
    x: ModelInput,
    output_size: Optional[Sequence[int]] = None,
) -> Tensor:
    """Keep dense logits aligned with the input/target spatial grid."""
    input_size = _input_spatial_size(x, output_size)
    if output.shape[-2:] == input_size:
        return output
    return F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)


def _make_backbone_probe_input(backbone: nn.Module) -> ModelInput:
    canonical_sensors = list(getattr(backbone, "canonical_sensors", []) or [])
    sensor_in_chans = getattr(backbone, "sensor_in_chans", {}) or {}
    sensor_input_size = getattr(backbone, "sensor_input_size", {}) or {}

    if len(canonical_sensors) > 1:
        return {
            sensor: torch.randn(
                1,
                int(sensor_in_chans[sensor]),
                *_as_hw(sensor_input_size[sensor]),
            )
            for sensor in canonical_sensors
        }

    if hasattr(backbone, "img_size") and hasattr(backbone, "in_chans"):
        h, w = _as_hw(backbone.img_size)
        return torch.randn(1, int(backbone.in_chans), h, w)

    if canonical_sensors:
        sensor = canonical_sensors[0]
        return torch.randn(
            1,
            int(sensor_in_chans[sensor]),
            *_as_hw(sensor_input_size[sensor]),
        )

    if sensor_in_chans and sensor_input_size:
        sensor = next(iter(sensor_in_chans))
        return torch.randn(
            1,
            int(sensor_in_chans[sensor]),
            *_as_hw(sensor_input_size[sensor]),
        )

    raise AttributeError(
        "The backbone is missing enough input-shape metadata for eager "
        "segmentation decoder initialization."
    )


class ConvSegmentor(nn.Module):
    """ViT-based segmentor that combines ViT backbone with decoder head."""
    
    def __init__(self, backbone: nn.Module, decoder: nn.Module, patch_size: int = 4):
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder
        self.patch_size = patch_size
        self.embedding_size = backbone.num_features
        
    def forward(self, x: Tensor, output_size: Optional[Sequence[int]] = None) -> Tensor:
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)
        
        # Get features from backbone
        features = self.encoder.get_intermediate_layers(x, norm=True)[0]

        features = features.permute(0, 2, 1)
        features = features.reshape(batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))
        
        # Apply decoder
        output = self.decoder(features)
        return _match_input_spatial_size(output, x, output_size)



class MultiScaleConvSegmentor(nn.Module):
    """
    (FINAL, CORRECTED VERSION - SMART ADAPTER)
    Adapts a multi-output backbone for a single-input decoder. This class
    is responsible for all feature fusion logic.
    """
    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        """
        Args:
            backbone (nn.Module): The multi-output backbone.
            decoder (nn.Module): The single-input decoder head.
            fusion_channels (int): The number of channels for the feature map
                AFTER all backbone features have been fused and projected. This
                MUST match the `in_channels` of the decoder.
        """
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder
        self.fusion_channels = decoder.in_channels
        
        self.projection: nn.Module = None

    def _initialize_projection(self, features: List[Tensor]):
        """Creates the 1x1 projection conv layer."""
        concatenated_channels = sum(f.shape[1] for f in features)
        
        print(f"Segmentor initializing fusion projection: {concatenated_channels} channels -> {self.fusion_channels} channels.")
        
        self.projection = nn.Conv2d(
            in_channels=concatenated_channels,
            out_channels=self.fusion_channels,
            kernel_size=1
        )
        self.projection.to(features[0].device, features[0].dtype)

    def forward(self, x: ModelInput, output_size: Optional[Sequence[int]] = None) -> Tensor:
        # 1. Get the tuple of feature maps from the backbone
        features = list(self.encoder.get_intermediate_layers(x, reshape=True))
        
        # 2. Lazily initialize the projection layer
        if self.projection is None:
            self._initialize_projection(features)
        
        # --- ROBUST SPATIAL SIZE DETECTION ---
        # Find the maximum height and width from all feature maps
        shapes = [f.shape[2:] for f in features]
        target_h = max(s[0] for s in shapes)
        target_w = max(s[1] for s in shapes)
        target_size = (target_h, target_w)
        
        # 3. Upsample all features to the largest target size
        upsampled_features = []
        for feature_map in features:
            if feature_map.shape[2:] != target_size:
                upsampled = F.interpolate(feature_map, size=target_size, mode='bilinear', align_corners=False)
                upsampled_features.append(upsampled)
            else:
                upsampled_features.append(feature_map)
        
        # 4. Concatenate along the channel dimension to fuse features
        fused_features = torch.cat(upsampled_features, dim=1)
        
        # 5. Project the fused features to the unified channel dimension
        projected_features = self.projection(fused_features)
        
        # 6. Pass the final, single tensor to the simple decoder
        output = self.decoder(projected_features)

        return _match_input_spatial_size(output, x, output_size)

# We can keep your original LearnableUpsampleBlock, but a simple ConvTranspose2d
# is often sufficient for the U-Net style path. For clarity and to match the new logic,
# let's define a simple fusion block.
class ConvBlock(nn.Module):
    """A standard Conv-BN-ReLU block for feature fusion."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class LearnableMultiScaleConvSegmentor(nn.Module):
    """
    A multi-scale segmentor that fuses features using a U-Net-like decoder path.
    It progressively upsamples deep features by a factor of 2 and fuses them with
    higher-resolution feature maps from the backbone.

    All layers are initialized EAGERLY in the constructor by inferring the input
    shape from the backbone, ensuring all parameters are tracked by wandb.
    """
    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        """
        Args:
            backbone (nn.Module): The multi-output backbone. For single-sensor
                inputs it should expose `img_size` and `in_chans`; for
                multi-branch inputs it should expose canonical_sensors plus
                sensor_in_chans/sensor_input_size.
            decoder (nn.Module): The single-input decoder head.
        """
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder

        # --- Eager Initialization ---
        # 1. Infer input shape from the backbone's attributes
        print("Segmentor: Inferring input shape from backbone for eager initialization...")
        dummy_input = _make_backbone_probe_input(backbone)
        
        # 2. Get feature map shapes with a dummy forward pass
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(dummy_input, reshape=True)

        # 3. Reverse features to start fusion from the deepest (lowest res) layer
        features_rev = list(reversed(features))
        
        # --- Create Learnable Fusion Path ---
        self.upsamplers = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        # Start with the channel count of the deepest feature map
        current_channels = features_rev[0].shape[1]

        # Iterate from the deepest layer up to the second-to-last one
        for i in range(len(features_rev) - 1):
            skip_connection_channels = features_rev[i+1].shape[1]
            
            # Upsampler: Upsamples by 2x and typically halves the channels.
            upsampler = nn.ConvTranspose2d(
                current_channels, current_channels // 2, kernel_size=2, stride=2
            )
            self.upsamplers.append(upsampler)

            # Fusion Conv: Takes the concatenated (upsampled + skip) features
            # and processes them. The output channels become the new 'current_channels'.
            fusion_conv = ConvBlock(
                in_channels=current_channels // 2 + skip_connection_channels,
                out_channels=skip_connection_channels
            )
            self.fusion_convs.append(fusion_conv)
            
            # Update the channel count for the next iteration
            current_channels = skip_connection_channels

        # --- Final Projection ---
        # Project the output of the last fusion block to match the decoder's expected input.
        decoder_in_channels = self._infer_decoder_in_channels()
        print(f"Segmentor: Creating final projection {current_channels} -> {decoder_in_channels} channels.")
        self.final_projection = nn.Conv2d(current_channels, decoder_in_channels, kernel_size=1)


    def _infer_decoder_in_channels(self) -> int:
        """Inspects the decoder to find the in_channels of its first Conv2d layer."""
        try:
            first_conv = next(m for m in self.decoder.modules() if isinstance(m, nn.Conv2d))
            return first_conv.in_channels
        except StopIteration:
            raise ValueError("Could not automatically infer decoder's input channels. "
                             "Ensure the decoder has at least one nn.Conv2d layer.")

    def forward(self, x: ModelInput, output_size: Optional[Sequence[int]] = None) -> Tensor:
        # 1. Get feature maps and reverse them to start from the deepest
        features = self.encoder.get_intermediate_layers(x, reshape=True)
        features_rev = list(reversed(features))

        # 2. Start the progressive upsampling with the deepest feature map
        current_feature = features_rev[0]

        # 3. Loop through the upsamplers and fusion blocks
        for i in range(len(self.upsamplers)):
            upsampled = self.upsamplers[i](current_feature)
            skip_connection = features_rev[i+1]
            fused = torch.cat([upsampled, skip_connection], dim=1)
            current_feature = self.fusion_convs[i](fused)

        # 4. Apply the final 1x1 conv to match decoder's expected channels
        projected = self.final_projection(current_feature)
        
        # 5. Pass the single, fused tensor to the decoder head
        output = self.decoder(projected)

        return _match_input_spatial_size(output, x, output_size)


def _normalize_tap_indices(tap_indices: Optional[Sequence[int]]) -> Optional[List[int]]:
    if tap_indices is None:
        return None
    return [int(idx) for idx in tap_indices]


def _select_feature_taps(
    features: Sequence[Tensor],
    tap_indices: Optional[Sequence[int]],
) -> List[Tensor]:
    features_list = list(features)
    if tap_indices is None:
        return features_list
    selected: List[Tensor] = []
    for idx in tap_indices:
        selected.append(features_list[int(idx)])
    return selected


def _default_vit_taps_from_backbone(backbone: nn.Module) -> List[int]:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None and hasattr(backbone, "vit_core"):
        blocks = getattr(backbone.vit_core, "blocks", None)
    if blocks is None and hasattr(backbone, "encoder"):
        blocks = getattr(backbone.encoder, "blocks", None)
    if blocks is None and hasattr(backbone, "spatial_encoder"):
        spatial_encoder = getattr(backbone, "spatial_encoder")
        blocks = getattr(spatial_encoder, "blocks", None)
        if blocks is None and hasattr(spatial_encoder, "net"):
            blocks = getattr(spatial_encoder.net, "blocks", None)
    if blocks is None and hasattr(backbone, "model"):
        blocks = getattr(backbone.model, "blocks", None)

    depth = len(blocks) if blocks is not None else 12
    if depth <= 1:
        return [0]
    if depth == 12:
        return [3, 7, 11]

    taps = [
        max(0, int(torch.ceil(torch.tensor(depth / 3.0)).item()) - 1),
        max(0, int(torch.ceil(torch.tensor(2.0 * depth / 3.0)).item()) - 1),
        depth - 1,
    ]
    # Keep order but remove duplicates for shallow models.
    deduped: List[int] = []
    for idx in taps:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def _backbone_looks_hierarchical(backbone: nn.Module) -> bool:
    class_name = backbone.__class__.__name__.lower()
    if "hiera" in class_name:
        return True
    return hasattr(backbone, "stage_ends") and (
        hasattr(backbone, "reroll") or hasattr(backbone, "shared_blocks")
    )


class MultiTapSegmentor(nn.Module):
    """Shared lightweight multi-tap segmentor for ViTs and Hiera-like backbones.

    Hiera-like backbones keep their native multi-scale stages because hierarchy
    is part of the encoder.  Vanilla ViTs use several intermediate transformer
    block taps, so they are not artificially restricted to the final token map.
    The decoder is shared for both cases.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module,
        tap_indices: Optional[Sequence[int]] = None,
        is_vit: Optional[bool] = None,
        patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder
        self.tap_indices = _normalize_tap_indices(tap_indices)
        self.is_vit = (
            not _backbone_looks_hierarchical(backbone)
            if is_vit is None
            else bool(is_vit)
        )
        inferred_patch_size = patch_size if patch_size is not None else getattr(backbone, "patch_size", None)
        if isinstance(inferred_patch_size, (list, tuple)):
            inferred_patch_size = inferred_patch_size[0]
        self.patch_size = int(inferred_patch_size) if inferred_patch_size is not None else None

    def _tokens_to_feature_map(
        self,
        feature: Tensor,
        x: ModelInput,
        output_size: Optional[Sequence[int]] = None,
    ) -> Tensor:
        if isinstance(feature, tuple):
            feature = feature[0]

        if feature.ndim == 4:
            input_hw = _input_spatial_size(x, output_size)
            h_patch = input_hw[0] // int(self.patch_size or 1)
            w_patch = input_hw[1] // int(self.patch_size or 1)
            if feature.shape[-2:] == (h_patch, w_patch):
                return feature
            if feature.shape[1:3] == (h_patch, w_patch):
                return feature.permute(0, 3, 1, 2).contiguous()
            return feature

        if feature.ndim != 3:
            raise ValueError(
                "ViT multi-tap features must be token tensors [B, N, C] or "
                f"spatial maps [B, C, H, W], got {tuple(feature.shape)}."
            )
        if self.patch_size is None:
            raise ValueError(
                "MultiTapSegmentor needs patch_size to reshape ViT token outputs."
            )

        batch_size, num_tokens, channels = feature.shape
        input_hw = _input_spatial_size(x, output_size)
        h_patch = input_hw[0] // self.patch_size
        w_patch = input_hw[1] // self.patch_size
        if num_tokens != h_patch * w_patch:
            side = int(num_tokens ** 0.5)
            if side * side != num_tokens:
                raise ValueError(
                    f"Cannot reshape {num_tokens} tokens into a feature map. "
                    f"Expected {h_patch * w_patch} tokens from input/patch_size."
                )
            h_patch = w_patch = side

        return (
            feature.permute(0, 2, 1)
            .reshape(batch_size, channels, h_patch, w_patch)
            .contiguous()
        )

    def _vit_feature_maps(
        self,
        x: ModelInput,
        output_size: Optional[Sequence[int]] = None,
    ) -> List[Tensor]:
        tap_indices = self.tap_indices or _default_vit_taps_from_backbone(self.encoder)
        outputs = self.encoder.get_intermediate_layers(x, n=tap_indices, norm=True)
        return [
            self._tokens_to_feature_map(feature, x, output_size)
            for feature in outputs
        ]

    def _hierarchical_feature_maps(self, x: ModelInput) -> List[Tensor]:
        outputs = self.encoder.get_intermediate_layers(x, reshape=True)
        return _select_feature_taps(outputs, self.tap_indices)

    def forward(self, x: ModelInput, output_size: Optional[Sequence[int]] = None) -> Tensor:
        features = (
            self._vit_feature_maps(x, output_size)
            if self.is_vit
            else self._hierarchical_feature_maps(x)
        )
        dense_output_size = _input_spatial_size(x, output_size)
        output = self.decoder(features, output_size=dense_output_size)
        return _match_input_spatial_size(output, x, dense_output_size)


class UperNetSegmentor(nn.Module):
    """UperNet-based segmentor that combines backbone with UperNet decoder head."""
    
    def __init__(self, backbone: nn.Module, decoder: nn.Module, patch_size: int = 4):
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder
        self.patch_size = patch_size
        
    def forward(self, x: Tensor, output_size: Optional[Sequence[int]] = None) -> Tensor:
        batch_size = x.shape[0]
        input_shape = _input_spatial_size(x, output_size)
        
        # Get intermediate features from backbone
        if hasattr(self.encoder, 'get_intermediate_layers'):
            # For ViT-style backbones
            # Get the output_layers from the decoder configuration
            output_layers = getattr(self.decoder, 'input_layers', [3, 5, 7, 11])
            intermediate_features = self.encoder.get_intermediate_layers(x, n=output_layers, norm=True)
            
            # Convert tuple to list if needed
            if isinstance(intermediate_features, tuple):
                intermediate_features = list(intermediate_features)
            
            # Convert features to spatial format
            features_list = []
            for i, feat in enumerate(intermediate_features):
                # Reshape from (B, L, C) to (B, C, H, W)
                feat = feat.permute(0, 2, 1)
                
                # Calculate spatial dimensions based on the actual sequence length
                # For ViT, sequence length = H * W / (patch_size^2)
                seq_len = feat.shape[2]  # This is the sequence length after permute
                spatial_size = int(seq_len ** 0.5)  # Assuming square feature maps
                
                # Reshape to spatial format
                feat = feat.reshape(batch_size, feat.shape[1], spatial_size, spatial_size)
                features_list.append(feat)
        else:
            # For CNN-style backbones that return multiple features
            features_list = self.encoder(x)
            if not isinstance(features_list, (list, tuple)):
                features_list = [features_list]
        
        # Apply UperNet decoder
        output = self.decoder(features_list, output_shape=input_shape)
        return _match_input_spatial_size(output, x, output_size)


class FCNSegmentor(nn.Module):
    """FCN-based segmentor that combines CNN backbone with FCN decoder."""
    
    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder

    def forward(self, x: Tensor, output_size: Optional[Sequence[int]] = None) -> Tensor:
        input_shape = _input_spatial_size(x, output_size)
        
        # Get features from backbone
        features = self.encoder(x)
        
        # Apply decoder
        output = self.decoder(features)
        
        # Upsample to input size
        return F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)


class ClassificationModel(nn.Module):
    """Classification model that combines backbone with classification head."""
    
    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder
        
    def forward(self, x: Tensor) -> Tensor:
        # Get features from backbone
        features = self.encoder(x)
        
        # Apply decoder
        output = self.decoder(features)
        return output 
