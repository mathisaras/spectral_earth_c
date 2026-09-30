from .spec_resnet import SpecResNet50, ModifiedResNet50
from .spec_vit import (
    SpecViTSmall,
    SpecViTBase,
    SpecViTLarge,
    SpecViTHuge,
    SpecViTGiant,
)
from .vit_mm import ViTMMBase
from .dofa_encoder import DofaBasePatch16, DofaLargePatch16
from .vit_torchvision import vit_b_torchvision

BACKBONE_REGISTRY = {
    "spec_resnet50": SpecResNet50,
    "spec_vit_small": SpecViTSmall,
    "spec_vit_base": SpecViTBase,
    "spec_vit_large": SpecViTLarge,
    "spec_vit_huge": SpecViTHuge,
    "spec_vit_giant": SpecViTGiant,
    "resnet50": ModifiedResNet50, 
    "dofa_base_patch16": DofaBasePatch16,
    "dofa_large_patch16": DofaLargePatch16,
    "vitmm_base": ViTMMBase,
    "vit_b_torchvision": vit_b_torchvision,
}
