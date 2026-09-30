import torch.nn as nn


class SpectralAdapter(nn.Sequential):
    def __init__(self):
        super(SpectralAdapter, self).__init__(
            nn.Conv3d(
                1, 32, kernel_size=(7, 1, 1), stride=(5, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(
                32, 64, kernel_size=(7, 1, 1), stride=(5, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(
                64, 128, kernel_size=(5, 1, 1), stride=(3, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, None, None))
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = super(SpectralAdapter, self).forward(x)
        x = x.squeeze(2)  # Remove the depth dimension
        return x

