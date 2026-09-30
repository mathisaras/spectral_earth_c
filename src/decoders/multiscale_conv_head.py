import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

class MultiScaleConvHead(nn.Module):
    """
    (CORRECTED FILTER SCHEDULE)
    A convolutional decoder head that takes a SINGLE input feature map and
    applies the upsampling and filter reduction schedule correctly.
    """
    def __init__(self, in_channels: int, num_classes: int, patch_size: int = 4, initial_upsample_filters: int = 128):
        """
        Args:
            in_channels (int): The number of channels of the single input tensor.
            num_classes (int): The number of output classes for segmentation.
            patch_size (int): The total upsampling factor required.
            initial_upsample_filters (int): The filter count for the first upsampling step.
        """
        super().__init__()
        self.in_channels = in_channels
        
        if not (isinstance(patch_size, int) and patch_size >= 1):
            raise ValueError("patch_size must be an integer >= 1.")

        layers: List[nn.Module] = []
        
        if patch_size == 1:
            layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1))
        else:
            def factorize(n: int) -> list[int]:
                factors = []
                while n % 2 == 0: factors.append(2); n //= 2
                p = 3
                while p * p <= n:
                    while n % p == 0: factors.append(p); n //= p
                    p += 2
                if n > 1: factors.append(n)
                return factors

            up_factors = sorted(factorize(patch_size))
            num_steps = len(up_factors)
            
            # --- 1. PRE-CALCULATE THE FULL OUTPUT FILTER SCHEDULE ---
            # The schedule is based on `initial_upsample_filters`
            output_filter_schedule: List[int] = []
            for i in range(num_steps - 1):
                f = max(initial_upsample_filters // (2 ** i), num_classes)
                output_filter_schedule.append(f)
            output_filter_schedule.append(num_classes) # Last step must be num_classes

            # --- 2. BUILD THE LAYERS SEQUENTIALLY ---
            current_channels = in_channels
            for idx, scale in enumerate(up_factors):
                is_last_step = (idx == num_steps - 1)
                
                # Get the next target output channel count from the schedule
                out_channels = output_filter_schedule[idx] 
                
                # Upsample
                layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
                
                # Convolution: current_channels -> out_channels (from schedule)
                layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
                
                # BatchNorm + ReLU (except after last conv)
                if not is_last_step:
                    layers.append(nn.BatchNorm2d(out_channels))
                    layers.append(nn.ReLU(inplace=True))
                
                # Update current_channels for the next iteration
                current_channels = out_channels

        self.segmentation_conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass takes a single tensor."""
        # The forward logic is correct and remains simple.
        return self.segmentation_conv(x)