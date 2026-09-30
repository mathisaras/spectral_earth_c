from typing import Any, List, Dict, Optional
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.datasets.spectral_earth_mm import SpectralEarthMMDataset
from src.samplers.sensor_batch_sampler import HomogeneousSensorBatchSampler

class SpectralEarthMMDataModule(LightningDataModule):
    """
    Lightning DataModule for Spectral Earth Multi-Modal.
    
    Features:
    - Wraps SpectralEarthMMDataset.
    - Uses HomogeneousSensorBatchSampler to ensure batches have consistent keys.
    - Supports weighted sampling (e.g., oversampling Triplets).
    """
    def __init__(
        self,
        index_path: str,
        root_dir: str,
        sensors: List[Dict],                         # Hydra: ${data.sensors}
        sampling_weights: Optional[Dict[str, float]] = None, # Hydra: ${data.sampling_weights}
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        default_img_size: int = 128,
        pin_memory: bool = True,
        **kwargs: Any,                               # Capture extra args (transforms, etc.)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        """
        Load datasets. 
        Note: In a production training run, you should likely have separate 
        'index_path_train' and 'index_path_val' in your config. 
        Here we initialize the dataset based on the provided index.
        """
        if stage == "fit" or stage is None:
            self.train_ds = SpectralEarthMMDataset(
                index_path=self.hparams.index_path,
                root_dir=self.hparams.root_dir,
                sensors=self.hparams.sensors,
                normalize=self.hparams.normalize,
                default_img_size=self.hparams.default_img_size,
                **self.dataset_kwargs
            )
            
            # For this example, we use the same dataset object. 
            # In practice, you might filter the index or load a 'val.csv'.
            self.val_ds = self.train_ds 

    def train_dataloader(self):
        """
        Returns DataLoader with Homogeneous Batch Sampling (Weighted).
        """
        # Sampler handles the shuffling and group selection
        sampler = HomogeneousSensorBatchSampler(
            dataset_groups=self.train_ds.groups,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            sampling_weights=self.hparams.sampling_weights
        )

        return DataLoader(
            self.train_ds,
            batch_sampler=sampler, # Mutually exclusive with shuffle=True
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )

    def val_dataloader(self):
        """
        Returns DataLoader with Homogeneous Batch Sampling (Natural Distribution).
        Using the sampler here ensures validation batches also don't crash 
        models that expect consistent keys within a batch.
        """
        sampler = HomogeneousSensorBatchSampler(
            dataset_groups=self.val_ds.groups,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            sampling_weights=None # None = Proportional to dataset size (Natural)
        )
        
        return DataLoader(
            self.val_ds,
            batch_sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )