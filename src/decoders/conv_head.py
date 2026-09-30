import math
import torch.nn as nn
from torch import Tensor

class ConvHead(nn.Module):
    """Convolutional head for upsampling ViT features to full resolution,
       supporting arbitrary integer patch_size > 1."""
    
    def __init__(
        self,
        num_input_features: int = 384,
        num_classes: int = 5,
        patch_size: int = 4
    ):
        super().__init__()
        print(f"<<<<< CONV HEAD __INIT__ with num_input_features={num_input_features}, "
              f"num_classes={num_classes}, patch_size={patch_size} >>>>>",
              flush=True)
            
        self.in_channels = num_input_features

        if patch_size == 1:
            # Just one 3×3 conv from features → classes, no upsampling
            self.segmentation_conv = nn.Sequential(
                nn.Conv2d(num_input_features, num_classes, kernel_size=3, padding=1)
            )
            return  

        # 1) Factor patch_size into primes, e.g. 12 → [2, 2, 3]
        def factorize(n: int) -> list[int]:
            factors = []
            # pull out 2s
            while n % 2 == 0:
                factors.append(2)
                n //= 2
            # pull out odd primes
            p = 3
            while p * p <= n:
                while n % p == 0:
                    factors.append(p)
                    n //= p
                p += 2
            if n > 1:
                factors.append(n)
            return factors

        up_factors = sorted(factorize(patch_size))  # e.g. [2,2,3]
        num_steps = len(up_factors)

        # 2) Plan a filter schedule: 128 → 64 → … → num_classes
        initial_filters = 128
        filters: list[int] = []
        for i in range(num_steps - 1):
            f = max(initial_filters // (2 ** i), num_classes)
            filters.append(f)
        filters.append(num_classes)

        # 3) Build the upsample + conv blocks
        layers: list[nn.Module] = []
        in_ch = num_input_features
        for idx, scale in enumerate(up_factors):
            out_ch = filters[idx]

            # Upsample by this factor
            layers.append(nn.Upsample(scale_factor=scale,
                                      mode='bilinear',
                                      align_corners=False))

            print(f"<<<<< CONV HEAD __INIT__ with in_ch={in_ch}, out_ch={out_ch} >>>>>", flush=True)
            # Conv layer
            layers.append(nn.Conv2d(in_ch, out_ch,
                                    kernel_size=3, padding=1))
            # BatchNorm + ReLU (except after last conv)
            if idx < num_steps - 1:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        self.segmentation_conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.segmentation_conv(x)