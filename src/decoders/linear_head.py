"""Linear head decoder for classification and regression tasks."""

import torch.nn as nn
from torch import Tensor


class LinearHead(nn.Module):
    """Simple linear head for classification or regression."""
    
    def __init__(self, num_input_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(num_input_features, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        # Handle case where x might be a feature map (flatten to tokens)
        if x.dim() > 2:
            x = x.mean(dim=(-2, -1))  # Global average pooling
        
        x = self.dropout(x)
        return self.linear(x) 