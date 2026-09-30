"""Utility functions for model instantiation with Hydra."""

import os
from typing import Dict, Any, Optional, List, Sequence, Union
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn as nn

from .segmentors import (
    ClassificationModel,
    ConvSegmentor,
    FCNSegmentor,
    LearnableMultiScaleConvSegmentor,
    MultiScaleConvSegmentor,
    MultiTapSegmentor,
    UperNetSegmentor,
    _backbone_looks_hierarchical,
    _default_vit_taps_from_backbone,
)


def infer_spatial_decoder_patch_size(
    backbone: nn.Module,
    fallback: Optional[int] = None,
) -> Optional[int]:
    """Infer the spatial upsampling factor needed by conv segmentation heads.

    Multi-sensor Hiera exposes the active/canonical sensor stride through
    ``patch_stride``. That is the value decoder heads need for dense prediction:
    EMIT uses stride 1, while ENMAP/DESIS typically use stride 2.
    """
    patch_stride = getattr(backbone, "patch_stride", None)
    if patch_stride is not None:
        if isinstance(patch_stride, (list, tuple)):
            if len(patch_stride) >= 2 and int(patch_stride[0]) == int(patch_stride[1]):
                return int(patch_stride[0])
        else:
            return int(patch_stride)

    patch_size = getattr(backbone, "patch_size", fallback)
    if patch_size is None:
        return fallback
    if isinstance(patch_size, (list, tuple)):
        if len(patch_size) >= 1:
            return int(patch_size[0])
        return fallback
    return int(patch_size)


def _as_int_pair(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return fallback
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _infer_backbone_probe_shape(
    backbone: nn.Module,
    probe_in_chans: Optional[int] = None,
    probe_img_size: Any = None,
) -> tuple[int, int, int]:
    """Infer a dummy input shape for eager decoder probing."""

    in_chans = probe_in_chans
    if in_chans is None:
        in_chans = getattr(backbone, "in_chans", None)
    if in_chans is None and getattr(backbone, "canonical_sensor", None) is not None:
        sensor = getattr(backbone, "canonical_sensor")
        sensor_in_chans = getattr(backbone, "sensor_in_chans", {})
        in_chans = sensor_in_chans.get(sensor)
    if in_chans is None and getattr(backbone, "sensor_in_chans", None):
        first_sensor = next(iter(backbone.sensor_in_chans))
        in_chans = backbone.sensor_in_chans[first_sensor]
    if isinstance(in_chans, (list, tuple)):
        in_chans = in_chans[0]
    if in_chans is None:
        in_chans = 3

    img_size = probe_img_size
    if img_size is None:
        img_size = getattr(backbone, "img_size", None)
    if img_size is None:
        img_size = getattr(backbone, "input_size", None)
    if img_size is None and getattr(backbone, "canonical_sensor", None) is not None:
        sensor = getattr(backbone, "canonical_sensor")
        sensor_input_size = getattr(backbone, "sensor_input_size", {})
        img_size = sensor_input_size.get(sensor)
    if img_size is None and getattr(backbone, "sensor_input_size", None):
        first_sensor = next(iter(backbone.sensor_input_size))
        img_size = backbone.sensor_input_size[first_sensor]

    height, width = _as_int_pair(img_size, fallback=(224, 224))
    return int(in_chans), height, width


def _infer_backbone_probe_input(
    backbone: nn.Module,
    probe_in_chans: Optional[int] = None,
    probe_img_size: Any = None,
) -> tuple[Union[torch.Tensor, dict[str, torch.Tensor]], Any, tuple[int, int]]:
    """Build a tensor or sensor-dict dummy input for decoder probing."""

    canonical_sensors = list(getattr(backbone, "canonical_sensors", []) or [])
    sensor_in_chans = getattr(backbone, "sensor_in_chans", {}) or {}
    sensor_input_size = getattr(backbone, "sensor_input_size", {}) or {}

    if len(canonical_sensors) > 1:
        dummy: dict[str, torch.Tensor] = {}
        probe_shapes: dict[str, tuple[int, int, int]] = {}
        for sensor in canonical_sensors:
            if sensor not in sensor_in_chans or sensor not in sensor_input_size:
                raise ValueError(
                    f"Cannot build probe input for canonical sensor '{sensor}'. "
                    "Backbone is missing sensor_in_chans or sensor_input_size metadata."
                )
            height, width = _as_int_pair(sensor_input_size[sensor], fallback=(224, 224))
            channels = int(sensor_in_chans[sensor])
            dummy[sensor] = torch.zeros(1, channels, height, width)
            probe_shapes[sensor] = (channels, height, width)
        first_shape = next(iter(probe_shapes.values()))
        return dummy, probe_shapes, (first_shape[1], first_shape[2])

    in_chans, height, width = _infer_backbone_probe_shape(
        backbone,
        probe_in_chans=probe_in_chans,
        probe_img_size=probe_img_size,
    )
    return torch.zeros(1, in_chans, height, width), (in_chans, height, width), (height, width)


def _feature_channels(
    feature: torch.Tensor,
    *,
    is_vit: bool,
    patch_size: Optional[int],
    input_hw: tuple[int, int],
) -> int:
    if isinstance(feature, tuple):
        feature = feature[0]
    if feature.ndim == 3:
        return int(feature.shape[-1])
    if feature.ndim != 4:
        raise ValueError(f"Expected 3D/4D feature tensor, got {tuple(feature.shape)}.")

    if is_vit and patch_size is not None:
        h_patch = input_hw[0] // int(patch_size)
        w_patch = input_hw[1] // int(patch_size)
        if tuple(feature.shape[1:3]) == (h_patch, w_patch):
            return int(feature.shape[-1])
    return int(feature.shape[1])


def infer_multitap_decoder_info(
    backbone: nn.Module,
    tap_indices: Optional[Sequence[int]] = None,
    is_vit: Optional[bool] = None,
    patch_size: Optional[int] = None,
    probe_in_chans: Optional[int] = None,
    probe_img_size: Any = None,
) -> dict[str, Any]:
    """Probe a backbone once so multi-tap decoder parameters are eager.

    The probe resolves the architecture mode, selected taps, and per-tap channel
    counts before the decoder is instantiated.  This keeps all decoder
    parameters registered in ``__init__`` instead of creating projections lazily
    during the first training forward.
    """

    if patch_size is None:
        patch_size = infer_spatial_decoder_patch_size(backbone)
    if patch_size is not None:
        patch_size = int(patch_size)

    dummy, probe_shape, input_hw = _infer_backbone_probe_input(
        backbone,
        probe_in_chans=probe_in_chans,
        probe_img_size=probe_img_size,
    )

    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            if is_vit is None:
                is_vit = not _backbone_looks_hierarchical(backbone)
                if not is_vit:
                    # Confirm the backbone actually exposes native spatial taps.
                    try:
                        native = backbone.get_intermediate_layers(dummy, reshape=True)
                        is_vit = not (
                            len(native) > 1
                            and all(getattr(f, "ndim", 0) == 4 for f in native)
                        )
                    except Exception:
                        is_vit = True

            if is_vit:
                resolved_taps = (
                    [int(idx) for idx in tap_indices]
                    if tap_indices is not None
                    else _default_vit_taps_from_backbone(backbone)
                )
                outputs = backbone.get_intermediate_layers(
                    dummy,
                    n=resolved_taps,
                    norm=True,
                )
            else:
                outputs_all = list(backbone.get_intermediate_layers(dummy, reshape=True))
                if tap_indices is None:
                    resolved_taps = None
                    outputs = outputs_all
                else:
                    resolved_taps = [int(idx) for idx in tap_indices]
                    outputs = [outputs_all[idx] for idx in resolved_taps]

            outputs = list(outputs)
            if not outputs:
                raise ValueError(
                    "Backbone probe returned no multi-tap features. "
                    "Set explicit tap_indices or check get_intermediate_layers."
                )

            channels = [
                _feature_channels(
                    feature,
                    is_vit=bool(is_vit),
                    patch_size=patch_size,
                    input_hw=input_hw,
                )
                for feature in outputs
            ]
    finally:
        if was_training:
            backbone.train()

    return {
        "in_channels_list": channels,
        "tap_indices": resolved_taps,
        "is_vit": bool(is_vit),
        "patch_size": patch_size,
        "probe_input_shape": probe_shape,
        "num_taps": len(channels),
    }


def instantiate_component(
    config: DictConfig,
    recursive: bool = False,
    **override_params: Any,
) -> nn.Module:
    """Instantiate a component using Hydra, merging the base config with override_params.

    Args:
        config: Hydra config for the component.
        **override_params: Dynamic parameters to override/set in the config.
                           Keys must match parameters in the component's config.
                           Example: num_input_features, num_classes for a decoder.

    Returns:
        Instantiated nn.Module.
    """
    # Filter out None values from override_params to prevent them from overriding
    # actual default values in the config with None.
    # Only non-None override_params will be merged.
    active_override_params = {k: v for k, v in override_params.items() if v is not None}
    
    merged_config = OmegaConf.merge(OmegaConf.create(config), OmegaConf.create(active_override_params))
    return hydra.utils.instantiate(merged_config, _recursive_=recursive)


def create_model(
    backbone: nn.Module,
    decoder: nn.Module,
    model_type: str = "conv_seg",
    patch_size: int = 4, # Relevant for ViTSegmentor
    tap_indices: Optional[Sequence[int]] = None,
    multi_tap_is_vit: Optional[bool] = None,
) -> nn.Module:
    """Create a complete model by combining backbone and decoder.
    
    Args:
        backbone: Instantiated backbone module.
        decoder: Instantiated decoder module.
        model_type: Type of model to create ("vit_seg", "conv_seg", "fcn_seg", "upernet_seg", "classification").
        patch_size: Patch size for ViT segmentation.
        
    Returns:
        Complete model.
    """
    if model_type == "vit_seg" or model_type == "conv_seg":
        return ConvSegmentor(backbone, decoder, patch_size=patch_size) 
    elif model_type == "multiscale_conv_seg":
        return MultiScaleConvSegmentor(backbone, decoder)
    elif model_type == "learnable_multiscale_conv_seg":
        return LearnableMultiScaleConvSegmentor(backbone, decoder)
    elif model_type in {"multi_tap_seg", "lightweight_multitap_seg"}:
        return MultiTapSegmentor(
            backbone,
            decoder,
            tap_indices=tap_indices,
            is_vit=multi_tap_is_vit,
            patch_size=patch_size,
        )
    elif model_type == "fcn_seg":
        return FCNSegmentor(backbone, decoder)
    elif model_type == "upernet_seg":
        return UperNetSegmentor(backbone, decoder, patch_size=patch_size)
    elif model_type == "classification":
        return ClassificationModel(backbone, decoder)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_weights(
    model: nn.Module, # Can be the full model or just the backbone
    pretrained_weights_path: Optional[str],
    strict: bool = False,
) -> None:
    """Load pretrained weights into a model from a given path.
    
    Args:
        model: Model to load weights into.
        pretrained_weights_path: Path to pretrained weights.
        strict: Whether to use strict loading.
    """
    def _remap_legacy_spec_vit_keys(
        state_dict: Dict[str, torch.Tensor],
        model_keys: set[str],
    ) -> Dict[str, torch.Tensor]:
        """Adapt old Spectral Earth SpecViT checkpoints to the wrapped patch embed.

        Older checkpoints store the ViT patch projection under
        ``vit_core.patch_embed.proj.*``.  The current SpecViT wraps patch_embed so
        the same layer lives at ``vit_core.patch_embed.original_patch_embed.proj.*``.
        Without this remap, strict=False silently leaves the patch projection
        randomly initialized, which is especially damaging for frozen linear probes.
        """

        remapped = dict(state_dict)

        old_patch_prefix = "vit_core.patch_embed.proj."
        new_patch_prefix = "vit_core.patch_embed.original_patch_embed.proj."
        if any(key.startswith(new_patch_prefix) for key in model_keys):
            for key in list(remapped.keys()):
                if not key.startswith(old_patch_prefix):
                    continue
                new_key = new_patch_prefix + key[len(old_patch_prefix):]
                if new_key in model_keys and new_key not in remapped:
                    remapped[new_key] = remapped[key]
                if key not in model_keys:
                    remapped.pop(key, None)

        adapter_prefix = "spectral_adapter."
        wrapped_adapter_prefix = "vit_core.patch_embed.spectral_adapter."
        if any(key.startswith(wrapped_adapter_prefix) for key in model_keys):
            for key, value in list(remapped.items()):
                if not key.startswith(adapter_prefix):
                    continue
                new_key = wrapped_adapter_prefix + key[len(adapter_prefix):]
                if new_key in model_keys and new_key not in remapped:
                    remapped[new_key] = value

        return remapped

    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        try:
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # If loading into a backbone submodule, state_dict may still have "backbone." prefix
            # (e.g. from a full Lightning .ckpt). Strip it so keys match the backbone's state_dict.
            model_keys = set(model.state_dict().keys())
            if state_dict and not any(k.startswith("backbone.") for k in model_keys):
                if all(k.startswith("backbone.") for k in state_dict.keys()):
                    state_dict = {k[len("backbone."):]: v for k, v in state_dict.items()}
                elif all(k.startswith("model.backbone.") for k in state_dict.keys()):
                    state_dict = {k[len("model.backbone."):]: v for k, v in state_dict.items()}

            state_dict = _remap_legacy_spec_vit_keys(state_dict, model_keys)

            msg = model.load_state_dict(state_dict, strict=strict)
            print(f"Successfully loaded pretrained weights from {pretrained_weights_path}. Load message: {msg}")
            if hasattr(model, 'pretrained_weights_loaded_externally'):
                 model.pretrained_weights_loaded_externally = True

        except Exception as e:
            raise e

    elif pretrained_weights_path:
        print(f"Warning: Pretrained weights path not found: {pretrained_weights_path}")
