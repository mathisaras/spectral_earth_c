"""TIMM model wrapper for standardized interface."""

import timm
import torch.nn as nn
from typing import List, Optional
from torch import Tensor


class TimmViTBackbone(nn.Module):
    """Wrapper for TIMM ViT models to standardize interface."""
    
    def __init__(
        self, 
        model_name: str,
        pretrained: bool = False,
        in_chans: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 0,
        **kwargs
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            #**kwargs
        )
        self.num_features = self.model.num_features
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)
        
    def get_intermediate_layers(self, x: Tensor, norm: bool = True, n: int = 1) -> List[Tensor]:
        """Get intermediate layers for ViT models."""
        if hasattr(self.model, 'get_intermediate_layers'):
            return self.model.get_intermediate_layers(x, norm=norm, n=n)
        else:
            # Fallback for models without this method
            return [self.forward(x)]


class TimmResNetBackbone(nn.Module):
    """Wrapper for TIMM ResNet models to standardize interface."""
    
    def __init__(
        self, 
        model_name: str,
        pretrained: bool = False,
        in_chans: int = 3,
        num_classes: int = 0,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Create model with dilation if specified
        if replace_stride_with_dilation:
            kwargs['replace_stride_with_dilation'] = replace_stride_with_dilation
            
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **kwargs
        )
        self.num_features = self.model.num_features
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward_features(x) 