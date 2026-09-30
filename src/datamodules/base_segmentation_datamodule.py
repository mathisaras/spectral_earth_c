from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import random
import warnings

import kornia.augmentation as K
import torch
import torch.nn.functional as F
from kornia.constants import DataKey, Resample
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from torch import Tensor
from omegaconf import OmegaConf
from hydra.utils import instantiate

from ..datasets.aligned_multi_sensor_segmentation_dataset import (
    AlignedMultiSensorSegmentationDataset,
)
from ..transforms.normalize_mm import SingleSensorNormalizer
from ..transforms.standardize_mm import SingleSensorStandardizer
from ..transforms.spectral_resample import SpectralResampler
from ..transforms.sensor_projection import SensorBranchProjector
from ..utils.sensor_registry import SensorRegistry
from ..utils.spectral_metadata import build_band_trim_transform


def _to_plain_dict(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _to_plain_container(cfg: Any) -> Any:
    if cfg is None:
        return None
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _normalize_sensor_name(sensor_name: str) -> str:
    return str(sensor_name).strip().upper()


def _as_sensor_name_list(values: Any) -> list[str]:
    values = _to_plain_container(values)
    if values is None:
        return []
    if isinstance(values, str):
        return [
            _normalize_sensor_name(value)
            for value in values.split(",")
            if value.strip()
        ]
    return [_normalize_sensor_name(value) for value in values]


class BaseSegmentationDataModule(NonGeoDataModule):
    """Base LightningDataModule for all segmentation datasets."""

    def __init__(
        self,
        dataset_class,
        sensor_config: Dict[str, Any],
        target_sensor_config: Optional[Dict[str, Any]] = None,
        target_sensor_configs: Optional[Any] = None,
        target_sensor_names: Optional[Any] = None,
        input_sensor_configs: Optional[Any] = None,
        input_sensor_names: Optional[Any] = None,
        joint_output_sensor: Optional[str] = None,
        input_standardize_sensors: Optional[Any] = None,
        projection_output_sensor: Optional[str] = None,
        projection_standardize_sensors: Optional[Any] = None,
        spectral_mapping_method: str = "auto",
        spectral_max_gap_nm: float = 80.0,
        spectral_min_coverage: float = 1.0e-6,
        spectral_uncovered_policy: str = "target_mean",
        batch_size: int = 64,
        img_size: Optional[Union[int, tuple[int, int]]] = 128,
        num_workers: int = 0,
        train_random_resized_crop_scale: tuple[float, float] = (0.4, 1.0),
        apply_input_normalization: bool = True,
        standardize: bool = False,
        standardization_mode: str = "bandwise",
        standardization_stats_path: Optional[str] = None,
        standardization_eps: float = 1.0e-6,
        post_projection_correction: Optional[Any] = None,
        **dataset_kwargs: Any,
    ) -> None:
        """
        Args:
            dataset_class: The dataset class to instantiate
            sensor_config: Source sensor configuration from hydra
            target_sensor_config: Optional target/model sensor configuration. If
                provided and distinct from sensor_config, inputs are spectrally
                resampled from source processed bands to target processed bands
                after source normalization.
            target_sensor_configs: Optional multi-target configs for projecting
                one source image into several backbone branches.
            target_sensor_names: Optional list/CSV of sensor names loaded from
                the SensorRegistry for multi-target projection.
            input_sensor_configs: Optional real multi-sensor input configs. When
                provided, the datamodule instantiates one aligned dataset per
                sensor and returns batch["image_by_sensor"] directly instead of
                spectrally projecting one source image.
            input_sensor_names: Optional list/CSV of real input sensor names
                loaded from the SensorRegistry.
            joint_output_sensor: Sensor used for batch["image"], mask sizing,
                and visualization in real multi-sensor mode.
            input_standardize_sensors: Optional list/CSV of real input sensors
                standardized after physical normalization. If omitted and
                standardize=True, every real input sensor is standardized.
            projection_output_sensor: Sensor key from the multi-target outputs
                to keep as batch["image"] for logging and reference sizing.
            projection_standardize_sensors: Optional list/CSV of projected
                target sensors that should receive target stats standardization.
                If null and standardize=True, every projected target is
                standardized.
            spectral_mapping_method: "auto", "bandpass", "linear",
                "linear_no_extrapolate", or "linear_gap_fill".
            spectral_max_gap_nm: Maximum source support gap filled by
                gap-aware interpolation methods.
            spectral_min_coverage: Target-band support fraction below which the
                projected band is treated as uncovered.
            spectral_uncovered_policy: Fill policy for uncovered target bands.
            batch_size: Size of each mini-batch
            img_size: Output spatial size. If None, defaults to the target
                sensor img_size when remapping is enabled, otherwise the source
                sensor img_size.
            num_workers: Number of workers for data loading
            train_random_resized_crop_scale: RandomResizedCrop scale range used
                for training augmentation.
            apply_input_normalization: If True, apply sensor-aware physical normalization.
            standardize: If True, apply mean/std standardization on top of normalization.
            standardization_mode: One of ["bandwise", "global"] for stats lookup.
            standardization_stats_path: Directory, index yaml, or sensor stats yaml.
            standardization_eps: Numerical stability epsilon for std clamping.
            post_projection_correction: Optional Hydra-instantiated transform
                applied after single-target spectral remapping and before
                optional target-sensor standardization.
            **dataset_kwargs: Additional keyword arguments passed to the dataset class
        """
        sensor_config = _to_plain_dict(sensor_config)
        target_sensor_config = _to_plain_dict(target_sensor_config)
        target_sensor_configs = _to_plain_container(target_sensor_configs)
        target_sensor_names = _to_plain_container(target_sensor_names)
        input_sensor_configs = _to_plain_container(input_sensor_configs)
        input_sensor_names = _to_plain_container(input_sensor_names)
        input_standardize_sensors = _to_plain_container(input_standardize_sensors)
        projection_standardize_sensors = _to_plain_container(projection_standardize_sensors)
        post_projection_correction = _to_plain_container(post_projection_correction)

        # Initialize dataset args with sensor config
        dataset_init_args = {
            "sensor_config": sensor_config,  # Pass full config to dataset
            **dataset_kwargs
        }
        
        super().__init__(dataset_class, batch_size, num_workers, **dataset_init_args)
        self.sensor_config = sensor_config
        self.target_sensor_config = target_sensor_config
        self.target_sensor_configs = self._resolve_target_sensor_configs(
            target_sensor_configs=target_sensor_configs,
            target_sensor_names=target_sensor_names,
        )
        self.input_sensor_configs = self._resolve_target_sensor_configs(
            target_sensor_configs=input_sensor_configs,
            target_sensor_names=input_sensor_names,
        )
        if self.input_sensor_configs and self.target_sensor_configs:
            raise ValueError(
                "Use either real aligned `input_sensor_names/input_sensor_configs` "
                "or synthetic `target_sensor_names/target_sensor_configs`, not both."
            )
        self.projection_output_sensor = (
            _normalize_sensor_name(projection_output_sensor)
            if projection_output_sensor is not None
            else None
        )
        if self.target_sensor_configs and self.projection_output_sensor is None:
            self.projection_output_sensor = next(iter(self.target_sensor_configs.keys()))
        if (
            self.projection_output_sensor is not None
            and self.target_sensor_configs
            and self.projection_output_sensor not in self.target_sensor_configs
        ):
            raise ValueError(
                f"projection_output_sensor={self.projection_output_sensor} is not "
                f"one of target_sensor_configs={list(self.target_sensor_configs.keys())}."
        )
        self.joint_output_sensor = (
            _normalize_sensor_name(joint_output_sensor)
            if joint_output_sensor is not None
            else None
        )
        if self.input_sensor_configs:
            if self.joint_output_sensor is None:
                source_sensor = _normalize_sensor_name(self.sensor_config.get("name", ""))
                self.joint_output_sensor = (
                    source_sensor
                    if source_sensor in self.input_sensor_configs
                    else next(iter(self.input_sensor_configs.keys()))
                )
            if self.joint_output_sensor not in self.input_sensor_configs:
                raise ValueError(
                    f"joint_output_sensor={self.joint_output_sensor} is not one "
                    f"of input_sensor_configs={list(self.input_sensor_configs.keys())}."
                )
        self.img_size = self._resolve_img_size(img_size)
        self.train_random_resized_crop_scale = (
            float(train_random_resized_crop_scale[0]),
            float(train_random_resized_crop_scale[1]),
        )
        self.apply_input_normalization = bool(apply_input_normalization)
        self.standardize = bool(standardize)
        self.standardization_mode = str(standardization_mode).lower()
        self.standardization_stats_path = standardization_stats_path
        self.standardization_eps = float(standardization_eps)
        self.band_trim = build_band_trim_transform(
            sensor_config=self.sensor_config,
            sensor_key="image",
        )

        self.sensor_projector: Optional[SensorBranchProjector] = None
        if self.target_sensor_configs:
            self.sensor_projector = SensorBranchProjector(
                source_sensor_config=self.sensor_config,
                target_sensor_configs=self.target_sensor_configs,
                apply_input_normalization=self.apply_input_normalization,
                standardize=self.standardize,
                standardize_sensors=_as_sensor_name_list(projection_standardize_sensors)
                if projection_standardize_sensors is not None
                else None,
                standardization_stats_path=self.standardization_stats_path,
                standardization_mode=self.standardization_mode,
                standardization_eps=self.standardization_eps,
                spectral_mapping_method=spectral_mapping_method,
                spectral_max_gap_nm=spectral_max_gap_nm,
                min_coverage=spectral_min_coverage,
                uncovered_policy=spectral_uncovered_policy,
            )

        self.input_sensor_sizes: dict[str, tuple[int, int]] = {}
        self.input_band_trims: dict[str, Optional[object]] = {}
        self.input_normalizers: dict[str, SingleSensorNormalizer] = {}
        self.input_standardizers: dict[str, SingleSensorStandardizer] = {}
        if self.input_sensor_configs:
            standardize_include = _as_sensor_name_list(input_standardize_sensors)
            if self.standardize and not self.standardization_stats_path:
                raise ValueError(
                    "standardize=True requires `standardization_stats_path` "
                    "for real multi-sensor inputs."
                )
            if self.standardize:
                standardize_set = (
                    set(standardize_include)
                    if input_standardize_sensors is not None
                    else set(self.input_sensor_configs.keys())
                )
            else:
                standardize_set = set()

            for sensor_name, cfg in self.input_sensor_configs.items():
                self.input_sensor_sizes[sensor_name] = _to_tuple(cfg.get("img_size"))
                self.input_band_trims[sensor_name] = build_band_trim_transform(
                    sensor_config=cfg,
                    sensor_key=sensor_name,
                )
                if self.apply_input_normalization:
                    self.input_normalizers[sensor_name] = SingleSensorNormalizer(
                        sensor_config=cfg
                    )
                if sensor_name in standardize_set:
                    self.input_standardizers[sensor_name] = (
                        SingleSensorStandardizer.from_stats_file(
                            stats_path=str(self.standardization_stats_path),
                            sensor_name=sensor_name,
                            mode=self.standardization_mode,
                            eps=self.standardization_eps,
                            sensor_config=cfg,
                        )
                    )

        self.normalizer = (
            SingleSensorNormalizer(sensor_config=self.sensor_config)
            if (
                self.apply_input_normalization
                and self.sensor_projector is None
                and not self.input_sensor_configs
            )
            else None
        )

        self.spectral_resampler: Optional[SpectralResampler] = None
        source_sensor_name = str(self.sensor_config.get("name", "")).upper()
        target_sensor_name = str(
            (self.target_sensor_config or {}).get("name", "")
        ).upper()
        if (
            self.sensor_projector is None
            and not self.input_sensor_configs
            and self.target_sensor_config is not None
            and target_sensor_name != source_sensor_name
        ):
            self.spectral_resampler = SpectralResampler.from_sensor_configs(
                source_sensor_config=self.sensor_config,
                target_sensor_config=self.target_sensor_config,
            )

        if post_projection_correction is not None and self.sensor_projector is not None:
            raise ValueError(
                "`post_projection_correction` currently supports the single-target "
                "spectral_resampler path only. Multi-target projection may standardize "
                "inside SensorBranchProjector, which would put raw-space corrections "
                "in the wrong space."
            )
        if post_projection_correction is not None and self.spectral_resampler is None:
            raise ValueError(
                "`post_projection_correction` requires a single-target sensor remap "
                "with `target_sensor_config` distinct from `sensor_config`."
            )
        if post_projection_correction is None:
            self.post_projection_correction = None
        elif isinstance(post_projection_correction, torch.nn.Module) or (
            callable(post_projection_correction)
            and not isinstance(post_projection_correction, (dict, list, tuple))
        ):
            self.post_projection_correction = post_projection_correction
        else:
            self.post_projection_correction = instantiate(post_projection_correction)

        self.standardizer: Optional[SingleSensorStandardizer] = None
        if self.standardize and self.sensor_projector is None and not self.input_sensor_configs:
            if not self.standardization_stats_path:
                raise ValueError(
                    "standardize=True requires `standardization_stats_path` "
                    "(directory, index yaml, or sensor stats yaml)."
                )
            if not self.apply_input_normalization:
                warnings.warn(
                    "standardize=True with apply_input_normalization=False. "
                    "Stats are usually expected over normalized inputs."
                )
            standardizer_sensor_cfg = (
                self.target_sensor_config
                if self.spectral_resampler is not None and self.target_sensor_config is not None
                else self.sensor_config
            )
            sensor_name = standardizer_sensor_cfg.get("name")
            if not sensor_name:
                raise ValueError(
                    "sensor_config.name is required when standardize=True."
                )
            self.standardizer = SingleSensorStandardizer.from_stats_file(
                stats_path=self.standardization_stats_path,
                sensor_name=str(sensor_name),
                mode=self.standardization_mode,
                eps=self.standardization_eps,
                sensor_config=standardizer_sensor_cfg,
            )

        self.train_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.img_size)),
            K.RandomResizedCrop(
                _to_tuple(self.img_size),
                scale=self.train_random_resized_crop_scale,
            ),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.img_size)),
            K.CenterCrop(self.img_size),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.img_size)),
            K.CenterCrop(self.img_size),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )

    def _resolve_target_sensor_configs(
        self,
        *,
        target_sensor_configs: Optional[Any],
        target_sensor_names: Optional[Any],
    ) -> dict[str, Dict[str, Any]]:
        """Resolve multi-target projection configs from names or config objects."""
        sensor_names = _as_sensor_name_list(target_sensor_names)
        if sensor_names:
            return {
                sensor_name: SensorRegistry.get(sensor_name)
                for sensor_name in sensor_names
            }

        if target_sensor_configs is None:
            return {}

        if isinstance(target_sensor_configs, dict):
            resolved = {}
            for key, cfg in target_sensor_configs.items():
                cfg_dict = _to_plain_dict(cfg)
                if cfg_dict is None:
                    continue
                sensor_name = _normalize_sensor_name(cfg_dict.get("name", key))
                resolved[sensor_name] = cfg_dict
            return resolved

        resolved = {}
        for cfg in target_sensor_configs:
            cfg_dict = _to_plain_dict(cfg)
            if cfg_dict is None:
                continue
            sensor_name = _normalize_sensor_name(cfg_dict.get("name", ""))
            if not sensor_name:
                raise ValueError("Every target sensor config must define `name`.")
            resolved[sensor_name] = cfg_dict
        return resolved

    def _resolve_img_size(
        self,
        img_size: Optional[Union[int, tuple[int, int]]],
    ) -> tuple[int, int]:
        """Resolve the effective spatial size used by the datamodule."""
        if img_size is not None:
            return _to_tuple(img_size)

        if self.target_sensor_configs and self.projection_output_sensor is not None:
            preferred_cfg = self.target_sensor_configs[self.projection_output_sensor]
        else:
            preferred_cfg = (
                self.target_sensor_config
                if self.target_sensor_config is not None
                else self.sensor_config
            )
        if preferred_cfg is None:
            raise ValueError(
                "img_size is None but neither sensor_config nor "
                "target_sensor_config is available."
            )

        resolved_img_size = preferred_cfg.get("img_size")
        if resolved_img_size is None:
            raise ValueError(
                "img_size is None and the effective sensor config does not define "
                "`img_size`."
            )
        return _to_tuple(resolved_img_size)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations after the batch is moved to the device."""
        if self.trainer:
            if self.input_sensor_configs:
                training = bool(self.trainer.training)
                return self._prepare_joint_batch(batch, training=training)

            if self.trainer.training:
                aug = self.train_aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing:
                aug = self.test_aug
            elif self.trainer.predicting:
                aug = self.test_aug
            else:
                raise NotImplementedError("Unknown trainer mode.")
            batch["image"] = batch["image"].float()
            if self.band_trim is not None:
                batch = self.band_trim(batch)
            batch = aug(batch)
            if self.normalizer is not None:
                batch["image"] = self.normalizer(batch["image"])
            if self.sensor_projector is not None:
                image_by_sensor = self.sensor_projector(batch["image"])
                batch["image_by_sensor"] = image_by_sensor
                output_sensor = self.projection_output_sensor or next(iter(image_by_sensor.keys()))
                batch["image"] = image_by_sensor[output_sensor]
            elif self.spectral_resampler is not None:
                batch["image"] = self.spectral_resampler(batch["image"])
            if self.post_projection_correction is not None:
                batch["image"] = self.post_projection_correction(batch["image"])
            if self.standardizer is not None:
                batch["image"] = self.standardizer(batch["image"])
            batch["image"] = batch["image"].to(batch["mask"].device)
        return batch 

    def setup(self, stage: str) -> None:
        """Set up datasets, with an aligned multi-sensor opt-in path."""
        if not self.input_sensor_configs:
            return super().setup(stage)

        if stage in ["fit"]:
            self.train_dataset = self._make_joint_dataset("train")
        if stage in ["fit", "validate"]:
            self.val_dataset = self._make_joint_dataset("val")
        if stage in ["test"]:
            self.test_dataset = self._make_joint_dataset("test")

    def _make_joint_dataset(self, split: str) -> AlignedMultiSensorSegmentationDataset:
        datasets = OrderedDict()
        for sensor_name, cfg in self.input_sensor_configs.items():
            kwargs = dict(self.kwargs)
            kwargs["sensor_config"] = cfg
            datasets[sensor_name] = self.dataset_class(split=split, **kwargs)

        return AlignedMultiSensorSegmentationDataset(
            datasets=datasets,
            output_sensor=self.joint_output_sensor,
            strict_alignment=True,
        )

    def _resize_image(self, image: Tensor, size: tuple[int, int]) -> Tensor:
        if tuple(int(v) for v in image.shape[-2:]) == size:
            return image.float()
        return F.interpolate(
            image.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _resize_mask(self, mask: Tensor, size: tuple[int, int]) -> Tensor:
        if mask.ndim == 3:
            mask_4d = mask.unsqueeze(1)
        elif mask.ndim == 4:
            mask_4d = mask
        else:
            raise ValueError(f"Expected mask shape [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")

        if tuple(int(v) for v in mask_4d.shape[-2:]) != size:
            mask_4d = F.interpolate(mask_4d.float(), size=size, mode="nearest")
        if int(mask_4d.shape[1]) != 1:
            raise ValueError(f"Expected a single-channel mask, got {tuple(mask_4d.shape)}")
        return mask_4d[:, 0].long()

    def _sample_joint_crop(
        self,
        size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, width = size
        area_scale = random.uniform(*self.train_random_resized_crop_scale)
        side_scale = area_scale ** 0.5
        crop_h = max(1, min(height, int(round(height * side_scale))))
        crop_w = max(1, min(width, int(round(width * side_scale))))
        top = 0 if crop_h >= height else random.randint(0, height - crop_h)
        left = 0 if crop_w >= width else random.randint(0, width - crop_w)
        return top, left, crop_h, crop_w

    def _crop_resize(
        self,
        tensor: Tensor,
        crop: tuple[int, int, int, int],
        ref_size: tuple[int, int],
        out_size: tuple[int, int],
        *,
        is_mask: bool = False,
    ) -> Tensor:
        top, left, crop_h, crop_w = crop
        height, width = tensor.shape[-2:]
        ref_h, ref_w = ref_size

        scaled_top = int(round(top * height / ref_h))
        scaled_left = int(round(left * width / ref_w))
        scaled_h = max(1, int(round(crop_h * height / ref_h)))
        scaled_w = max(1, int(round(crop_w * width / ref_w)))
        scaled_h = min(scaled_h, height)
        scaled_w = min(scaled_w, width)
        scaled_top = min(max(0, scaled_top), max(0, height - scaled_h))
        scaled_left = min(max(0, scaled_left), max(0, width - scaled_w))

        cropped = tensor[
            ...,
            scaled_top : scaled_top + scaled_h,
            scaled_left : scaled_left + scaled_w,
        ]
        if tuple(int(v) for v in cropped.shape[-2:]) == out_size:
            return cropped

        if is_mask:
            return F.interpolate(cropped.float(), size=out_size, mode="nearest")
        return F.interpolate(
            cropped.float(),
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )

    def _prepare_joint_batch(
        self,
        batch: dict[str, Any],
        *,
        training: bool,
    ) -> dict[str, Any]:
        if not self.input_sensor_configs:
            return batch

        if "image_by_sensor" not in batch:
            raise KeyError("Joint input mode expects batch['image_by_sensor'].")

        image_by_sensor = OrderedDict()
        for raw_sensor, image in batch["image_by_sensor"].items():
            sensor_name = _normalize_sensor_name(raw_sensor)
            if sensor_name not in self.input_sensor_configs:
                continue
            sensor_batch = {sensor_name: image.float()}
            band_trim = self.input_band_trims.get(sensor_name)
            if band_trim is not None:
                sensor_batch = band_trim(sensor_batch)
            image_by_sensor[sensor_name] = self._resize_image(
                sensor_batch[sensor_name],
                self.input_sensor_sizes[sensor_name],
            )

        if not image_by_sensor:
            raise ValueError("No configured input sensors were found in image_by_sensor.")

        output_size = self.input_sensor_sizes[self.joint_output_sensor]
        mask = self._resize_mask(batch["mask"], output_size)

        if training:
            ref_size = output_size
            crop = self._sample_joint_crop(ref_size)
            for sensor_name, image in list(image_by_sensor.items()):
                target_size = self.input_sensor_sizes[sensor_name]
                image_by_sensor[sensor_name] = self._crop_resize(
                    image,
                    crop,
                    ref_size,
                    target_size,
                    is_mask=False,
                )
            mask_4d = mask.unsqueeze(1)
            mask = self._crop_resize(
                mask_4d,
                crop,
                ref_size,
                output_size,
                is_mask=True,
            )[:, 0].long()

            if random.random() < 0.5:
                for sensor_name in list(image_by_sensor.keys()):
                    image_by_sensor[sensor_name] = torch.flip(
                        image_by_sensor[sensor_name], dims=[-1]
                    )
                mask = torch.flip(mask, dims=[-1])
            if random.random() < 0.5:
                for sensor_name in list(image_by_sensor.keys()):
                    image_by_sensor[sensor_name] = torch.flip(
                        image_by_sensor[sensor_name], dims=[-2]
                    )
                mask = torch.flip(mask, dims=[-2])

        for sensor_name, image in list(image_by_sensor.items()):
            normalizer = self.input_normalizers.get(sensor_name)
            if normalizer is not None:
                image = normalizer(image)
            standardizer = self.input_standardizers.get(sensor_name)
            if standardizer is not None:
                image = standardizer(image)
            image_by_sensor[sensor_name] = image.to(mask.device)

        batch["image_by_sensor"] = image_by_sensor
        batch["image"] = image_by_sensor[self.joint_output_sensor].to(mask.device)
        batch["mask"] = mask
        return batch
