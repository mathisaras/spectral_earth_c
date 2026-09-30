"""Dataset package exports.

Keep imports lazy to avoid pulling optional heavy dependencies when only one
dataset implementation is needed.
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PrismaTreeDataset": ("src.datasets.prisma_tree_dataset", "PrismaTreeDataset"),
    "OxhyperDataset": ("src.datasets.oxhyper_dataset", "OxhyperDataset"),
    "ConusminDataset": ("src.datasets.conusmin_dataset", "ConusminDataset"),
    "EnMAPTreeDataset": ("src.datasets.enmap_tree_dataset", "EnMAPTreeDataset"),
    "BaseSegmentationDataset": ("src.datasets.base_segmentation_dataset", "BaseSegmentationDataset"),
    "CDLDataset": ("src.datasets.cdl_dataset", "CDLDataset"),
    "NLCDDataset": ("src.datasets.nlcd_dataset", "NLCDDataset"),
    "EurocropsDataset": ("src.datasets.eurocrops_dataset", "EurocropsDataset"),
    "TreeMapDataset": ("src.datasets.treemap_dataset", "TreeMapDataset"),
    "BDForetDataset": ("src.datasets.bdforet_dataset", "BDForetDataset"),
    "BNETDDataset": ("src.datasets.bnetd_dataset", "BNETDDataset"),
    "CorineDataset": ("src.datasets.corine_dataset", "CorineDataset"),
    "Gaofen5WuhanDataset": ("src.datasets.gaofen5_wuhan_dataset", "Gaofen5WuhanDataset"),
    "H2SRDataset": ("src.datasets.h2sr_dataset", "H2SRDataset"),
    "OxHyperMineralsEMITL2ADataset": ("src.datasets.oxhyperminerals_emit_l2a_dataset", "OxHyperMineralsEMITL2ADataset"),
    "ZarrSpectralEarthMMDataset": ("src.datasets.spectral_earth_mm_zarr", "ZarrSpectralEarthMMDataset"),
    "AlignedMultiSensorSegmentationDataset": ("src.datasets.aligned_multi_sensor_segmentation_dataset", "AlignedMultiSensorSegmentationDataset"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, symbol = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, symbol)
