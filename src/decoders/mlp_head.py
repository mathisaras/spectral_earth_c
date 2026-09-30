"""MLP head decoder for classification and regression tasks."""

from typing import List
import torch.nn as nn
from torch import Tensor


class MLPHead(nn.Module):
    """Multi-layer perceptron head for classification or regression."""
    
    def __init__(
        self, 
        num_input_features: int, 
        hidden_dims: List[int], 
        num_classes: int, 
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        # Choose activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        layers = []
        prev_dim = num_input_features
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        # Handle case where x might be a feature map (flatten to tokens)
        if x.dim() > 2:
            x = x.mean(dim=(-2, -1))  # Global average pooling
        
        return self.mlp(x) 