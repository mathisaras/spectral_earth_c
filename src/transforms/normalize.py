import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F



class NormalizeMeanStd(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    # Define init function
    def __init__(self, 
                 mean: Tensor, 
                 std: Tensor):
        super().__init__()
        
        self.mean = mean 
        self.std = std


    @torch.no_grad()  
    def forward(self, x):
        
        x_out = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)

        return x_out


class NormalizeMinMax(nn.Module):
    """
    Module to perform 0–max normalization on torch tensors.

    Clips values to [0, max_val], then scales so that max_val → 1.0.
    """
    def __init__(self, max_val: float):
        super().__init__()
        self.register_buffer("max_val", torch.tensor(max_val))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clip to [0, max_val], then divide
        x = x.clamp(min=0.0, max=self.max_val.to(x.device))
        return x / self.max_val.to(x.device)


class MinMaxNormalize(nn.Module):
    """Normalize a tensor using min-max normalization with clamping."""

    def __init__(self, min_val: float = 0.0, max_val: float = 10000.0) -> None:
        """
        Args:
            min_val: Minimum value for clamping
            max_val: Maximum value for normalization divisor
        """
        super().__init__()
        self.register_buffer("min_val", torch.tensor(min_val))
        self.register_buffer("max_val", torch.tensor(max_val))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        
        """Normalize input using min-max normalization.
        
        Steps:
        1. Clamp values to min_val and max_val
        2. Divide by max_val - min_val

        Args:
            sample: A dict with "image" key, where "image" is a tensor to normalize.

        Returns:
            Normalized input sample.
        """
        
        x = torch.clamp(x, min=self.min_val.to(x.device), max=self.max_val.to(x.device) )
        x = x / (self.max_val.to(x.device) - self.min_val.to(x.device))
        return x

# Divide by 10000
class DivideBy10000(nn.Module):
    """Divide a tensor by 10000."""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 10000.0