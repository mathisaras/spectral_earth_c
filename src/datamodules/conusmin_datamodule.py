from typing import Any, Dict, Optional, Union

import warnings

import kornia.augmentation as K
from kornia.constants import DataKey
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from torch import Tensor
from omegaconf import OmegaConf

from ..transforms.normalize_mm import SingleSensorNormalizer
from ..transforms.sensor_projection import SensorBranchProjector
from ..transforms.spectral_resample import SpectralResampler
from ..transforms.standardize_mm import SingleSensorStandardizer
from ..datasets.conusmin_dataset import ConusminDataset  
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


class ConusminDataModule(NonGeoDataModule):  
    """LightningDataModule for the EMIT CONUSMin multi-label mineral classification dataset."""  

    def __init__(
        self,
        sensor_config: dict,  # Sensor configuration from hydra
        target_sensor_config: Optional[dict] = None,
        target_sensor_configs: Optional[Any] = None,
        target_sensor_names: Optional[Any] = None,
        projection_output_sensor: Optional[str] = None,
        projection_standardize_sensors: Optional[Any] = None,
        spectral_mapping_method: str = "auto",
        spectral_max_gap_nm: float = 80.0,
        spectral_min_coverage: float = 1.0e-6,
        spectral_uncovered_policy: str = "target_mean",
        num_classes: int = 18,  
        root: str = "data",
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        return_mask: bool = False,
        subset_percent: Optional[float] = None,
        apply_input_normalization: bool = True,
        standardize: bool = False,
        standardization_mode: str = "bandwise",
        standardization_stats_path: Optional[str] = None,
        standardization_eps: float = 1.0e-6,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            sensor: Sensor configuration from hydra
            num_classes: Number of CONUSMin mineral classes (fixed at 18)  
            root: Root directory where dataset can be found
            batch_size: Size of each mini-batch
            img_size: Size of each image
            num_workers: Number of workers for data loading
            return_mask: If True, also returns the spatial mask
            subset_percent: If set (0.0-1.0), use a random subset of samples
            **kwargs: Additional keyword arguments passed to the dataset
        """
        sensor_config = _to_plain_dict(sensor_config)
        target_sensor_config = _to_plain_dict(target_sensor_config)
        target_sensor_configs = _to_plain_container(target_sensor_configs)
        target_sensor_names = _to_plain_container(target_sensor_names)
        projection_standardize_sensors = _to_plain_container(projection_standardize_sensors)

        dataset_kwargs = {
            "sensor_config": sensor_config,
            "root": root,
            "num_classes": num_classes,
            "return_mask": return_mask,
            "subset_percent": subset_percent,
            **kwargs
        }

        super().__init__(ConusminDataset, batch_size, num_workers, **dataset_kwargs)  
        self.img_size = _to_tuple(img_size)
        self.sensor_config = sensor_config
        self.target_sensor_config = target_sensor_config
        self.target_sensor_configs = self._resolve_target_sensor_configs(
            target_sensor_configs=target_sensor_configs,
            target_sensor_names=target_sensor_names,
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

        self.normalize = (
            SingleSensorNormalizer(sensor_config=self.sensor_config)
            if self.apply_input_normalization and self.sensor_projector is None
            else None
        )

        self.spectral_resampler: Optional[SpectralResampler] = None
        source_sensor_name = str(self.sensor_config.get("name", "")).upper()
        target_sensor_name = str(
            (self.target_sensor_config or {}).get("name", "")
        ).upper()
        if (
            self.sensor_projector is None
            and self.target_sensor_config is not None
            and target_sensor_name != source_sensor_name
        ):
            self.spectral_resampler = SpectralResampler.from_sensor_configs(
                source_sensor_config=self.sensor_config,
                target_sensor_config=self.target_sensor_config,
            )

        self.standardizer: Optional[SingleSensorStandardizer] = None
        if self.standardize and self.sensor_projector is None:
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
            )

        # NOTE: For classification, we only need image transformations
        # Mask transformation is not needed since we're predicting class labels, not segmentation maps

        # We need a different setup for the mask key handling when return_mask=True
        data_keys = ["image"]
        if return_mask:
            data_keys.append("mask")

        mask_extra_args = {}
        if return_mask:
            mask_extra_args = {
                DataKey.MASK: {"resample": K.Resample.NEAREST, "align_corners": None}
            }

        self.train_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.img_size)),
            K.RandomResizedCrop(_to_tuple(self.img_size), scale=(0.4, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=data_keys,
            extra_args=mask_extra_args,
        )

        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.img_size)),
            K.CenterCrop(self.img_size),
            data_keys=data_keys,
        )

        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.img_size)),
            K.CenterCrop(self.img_size),
            data_keys=data_keys,
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations after the batch is moved to the device."""
        if self.trainer:
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
            if self.normalize is not None:
                batch["image"] = self.normalize(batch["image"])
            if self.sensor_projector is not None:
                image_by_sensor = self.sensor_projector(batch["image"])
                batch["image_by_sensor"] = image_by_sensor
                output_sensor = self.projection_output_sensor or next(iter(image_by_sensor.keys()))
                batch["image"] = image_by_sensor[output_sensor]
            elif self.spectral_resampler is not None:
                batch["image"] = self.spectral_resampler(batch["image"])
            if self.standardizer is not None:
                batch["image"] = self.standardizer(batch["image"])
            # Ensure all tensors are on the same device
            if "label" in batch:
                device = batch["image"].device
                batch["label"] = batch["label"].to(device)
            if "mask" in batch:
                batch["mask"] = batch["mask"].to(batch["image"].device)

        return batch

    @staticmethod
    def _resolve_target_sensor_configs(
        target_sensor_configs: Optional[Any],
        target_sensor_names: Optional[Any],
    ) -> dict[str, dict[str, Any]]:
        if target_sensor_configs is not None and target_sensor_names is not None:
            raise ValueError(
                "Use either `target_sensor_configs` or `target_sensor_names`, not both."
            )

        if target_sensor_names is not None:
            registry = SensorRegistry()
            configs = {}
            for sensor_name in _as_sensor_name_list(target_sensor_names):
                configs[sensor_name] = registry.get(sensor_name)
            return configs

        if target_sensor_configs is None:
            return {}

        if isinstance(target_sensor_configs, dict):
            iterable = target_sensor_configs.values()
        else:
            iterable = target_sensor_configs

        resolved: dict[str, dict[str, Any]] = {}
        for cfg in iterable:
            cfg_dict = _to_plain_dict(cfg)
            if cfg_dict is None:
                continue
            sensor_name = _normalize_sensor_name(cfg_dict.get("name", ""))
            if not sensor_name:
                raise ValueError("Every target sensor config must define `name`.")
            resolved[sensor_name] = cfg_dict
        return resolved