"""Datamodule package exports.

Keep imports lazy to avoid importing optional dependencies (e.g. torchgeo)
when only a subset of datamodules is needed.
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PrismaTreeDataModule": ("src.datamodules.prisma_tree_datamodule", "PrismaTreeDataModule"),
    "OxhyperDataModule": ("src.datamodules.oxhyper_datamodule", "OxhyperDataModule"),
    "ConusminDataModule": ("src.datamodules.conusmin_datamodule", "ConusminDataModule"),
    "EnMAPTreeDataModule": ("src.datamodules.enmap_tree_datamodule", "EnMAPTreeDataModule"),
    "BaseSegmentationDataModule": ("src.datamodules.base_segmentation_datamodule", "BaseSegmentationDataModule"),
    "BDForetDataModule": ("src.datamodules.bdforet_datamodule", "BDForetDataModule"),
    "BNETDDataModule": ("src.datamodules.bnetd_datamodule", "BNETDDataModule"),
    "CDLDataModule": ("src.datamodules.cdl_datamodule", "CDLDataModule"),
    "NLCDDataModule": ("src.datamodules.nlcd_datamodule", "NLCDDataModule"),
    "TreeMapDataModule": ("src.datamodules.treemap_datamodule", "TreeMapDataModule"),
    "EurocropsDataModule": ("src.datamodules.eurocrops_datamodule", "EurocropsDataModule"),
    "CorineDataModule": ("src.datamodules.corine_datamodule", "CorineDataModule"),
    "Gaofen5WuhanDataModule": ("src.datamodules.gaofen5_wuhan_datamodule", "Gaofen5WuhanDataModule"),
    "H2SRDataModule": ("src.datamodules.h2sr_datamodule", "H2SRDataModule"),
    "OxHyperMineralsEMITL2ADataModule": ("src.datamodules.oxhyperminerals_emit_l2a_datamodule", "OxHyperMineralsEMITL2ADataModule"),
    "SpectralEarthMMZarrDataModule": ("src.datamodules.spectral_earth_mm_zarr", "SpectralEarthMMZarrDataModule"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, symbol = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, symbol)
