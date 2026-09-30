import torch.nn as nn
from torchvision.models import VisionTransformer

def vit_b_torchvision(
    img_size: int = 224, 
    patch_size: int = 16, 
    in_chans: int = 3, 
    name: str = "vit_b_torchvision",
    **kwargs
) -> VisionTransformer:
    """
    Factory function to create a torchvision VisionTransformer with 
    ViT-Base architecture (12 layers, 12 heads, 768 dim),
    modified to support custom input channels.
    """
    
    # Standard ViT-Base Hyperparameters
    embed_dim = 768
    mlp_dim = 3072
    num_layers = 12
    num_heads = 12

    # Instantiate the base Torchvision model
    model = VisionTransformer(
        image_size=img_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=mlp_dim,
        **kwargs
    )

    # -------------------------------------------------------------------------
    # Handle Custom Input Channels
    # -------------------------------------------------------------------------
    # Torchvision defaults to 3 channels. If your sensor data has N channels,
    # we must replace the first convolution (patch embedding).
    if in_chans != 3:
        model.conv_proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    # -------------------------------------------------------------------------
    # Add attributes for compatibility (Optional but recommended)
    # -------------------------------------------------------------------------
    # Some libraries (like TIMM or older Lightly versions) look for 'embed_dim' 
    # or 'patch_embed'. We alias them here to make the model robust.
    model.embed_dim = embed_dim 
    model.num_patches = (img_size // patch_size) ** 2

    return model

