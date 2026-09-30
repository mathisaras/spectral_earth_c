# SpectralEarth Multimodal — Downstream Benchmarking

This repository contains the code used for the thesis **"Benchmarking Foundation
Models on Hyperspectral Images for Earth Observation Tasks"** (Mathimenaka
Ramasamy, 2026). It fine-tunes and evaluates several spectral /
hyperspectral foundation model backbones on 5 downstream tasks.

This README describes only the parts of the repo that are actually used for
this thesis's experiments: the tasks, backbones, and files listed below are
the complete set relevant to reproducing or extending this work.

---

## Downstream tasks used in this thesis

| Task key         | Description                                   | Type                          | Sensor(s)     |
|-------------------|-----------------------------------------------|--------------------------------|---------------|
| `conusmin`        | Mineral classification (ConUS_Min)            | Multi-label classification    | EMIT          |
| `enmap_tree`      | Tree species mapping (Spain_Tree)             | Semantic segmentation         | EnMAP         |
| `prisma_tree`     | Tree species mapping (Spain_Tree)             | Semantic segmentation         | PRISMA        |
| `oxhyper_enmap`   | Mineral segmentation (OxHyper)                | Multi-label segmentation      | EnMAP         |
| `oxhyper_emit`    | Mineral segmentation (OxHyper)                | Multi-label segmentation      | EMIT          |

Configs for these tasks live under:
```
configs/experiment/downstream/<task>/
```
Corresponding dataset / datamodule classes:
```
src/datasets/conusmin_dataset.py         + src/datamodules/conusmin_datamodule.py
src/datasets/enmap_tree_dataset.py       + src/datamodules/enmap_tree_datamodule.py
src/datasets/prisma_tree_dataset.py      + src/datamodules/prisma_tree_datamodule.py
src/datasets/oxhyper_dataset.py          + src/datamodules/oxhyper_datamodule.py   (shared by both oxhyper_* tasks)
```
Corresponding LightningModules:
```
src/models/multilabel_classification_module.py   (conusmin)
src/models/semantic_segmentation_module.py       (enmap_tree, prisma_tree)
src/models/multilabel_segmentation_module.py     (oxhyper_enmap, oxhyper_emit)
```

---

## Backbones / models benchmarked

| Backbone            | Source file(s)                                             | Weights                                      |
|---------------------|--------------------------------------------------------------|-----------------------------------------------|
| Spec. ResNet50       | `src/backbones/` (SpecResNet50)                              | Random / MoCo-V2 / DINO pretraining           |
| Spec. ViT-S / ViT-B  | `src/backbones/` (SpecViTSmall / SpecViTBase)                | MAE pretraining, from scratch (Supervised)    |
| DOFA-B               | `src/backbones/dofa_encoder_autoload.py`                     | `pretrained_models/comparison/dofa/`          |
| Panopticon           | `src/backbones/panopticon.py`                                 | `pretrained_models/comparison/panopticon/`    |
| HyperSIGMA           | `src/backbones/hypersigma.py`                                 | `pretrained_models/comparison/hypersigma_b/`  |
| SpecAware            | `src/backbones/specaware*.py`                                  | `pretrained_models/comparison/specaware/`     |

Each backbone is evaluated under some subset of the following protocols
(depending on what's meaningful for that backbone):
- **Supervised**: trained from random init, no pretrained weights
- **Frozen**: pretrained backbone frozen, only a linear/conv head trained
- **Full FT**: full backbone + head fine-tuned
- **Adapter**: backbone frozen, lightweight adapter layers fine-tuned

---

## Repo structure (relevant parts)

```
configs/                     Hydra configs (composable: backbone / data / model / experiment)
  ├── backbone/               per-backbone architecture configs
  ├── experiment/downstream/  one subfolder per task, sweep-ready configs
  └── sensor/                 per-sensor metadata (band counts, wavelengths, etc.)
src/
  ├── backbones/              backbone wrapper classes (see table above)
  ├── datasets/                per-task PyTorch Dataset classes
  ├── datamodules/             per-task LightningDataModule classes
  ├── models/                  per-task-type LightningModule (train/val/test loop, loss, metrics)
  ├── decoders/                segmentation heads (e.g. ConvHead)
  └── train.py                 Hydra entrypoint for training
```


