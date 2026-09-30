"""FCN head decoder for CNN-based segmentation."""

import torch.nn as nn


class FCNHead(nn.Sequential):
    """FCN head for CNN-based segmentation models."""
    
    def __init__(self, num_input_features: int, num_classes: int) -> None:
        # num_input_features corresponds to former in_channels
        # num_classes corresponds to former out_channels
        inter_channels = num_input_features // 8
        layers = [
            nn.Conv2d(num_input_features, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, num_classes, 1),
        ]
        super().__init__(*layers) 