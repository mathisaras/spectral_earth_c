# Import VisionTransformer from timm
from timm.models.vision_transformer import VisionTransformer    
from functools import partial
from inspect import signature
from torch import nn, Tensor
import torch
from typing import Union, Sequence
from typing import Tuple
from .spectral_adapter import SpectralAdapter
import lightly.models.utils as lightly_utils


class PatchEmbedWrapper(nn.Module):
    def __init__(self, spectral_adapter: nn.Module, original_patch_embed: nn.Module):
        super().__init__()
        self.spectral_adapter = spectral_adapter
        self.original_patch_embed = original_patch_embed

    def forward(self, x: Tensor) -> Tensor:
        # Apply the spectral adapter exactly once.
        x = self.spectral_adapter(x)
        # Then compute the patch embeddings.
        return self.original_patch_embed(x)

    @property
    def num_patches(self):
        return self.original_patch_embed.num_patches

    @property
    def patch_size(self):
        # Try to get patch_size, common in timm patch_embed layers
        if hasattr(self.original_patch_embed, 'patch_size'):
            return self.original_patch_embed.patch_size
        # Fallback if it's a tuple (height, width)
        if hasattr(self.original_patch_embed, 'proj') and hasattr(self.original_patch_embed.proj, 'kernel_size'):
            return self.original_patch_embed.proj.kernel_size
        raise AttributeError("PatchEmbedWrapper could not determine patch_size from original_patch_embed")

    def __getattr__(self, name):
        # Forward attribute lookups to the original patch embedding module.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.original_patch_embed, name):
                return getattr(self.original_patch_embed, name)
            # If original_patch_embed.proj exists and has the attribute
            if hasattr(self.original_patch_embed, 'proj') and hasattr(self.original_patch_embed.proj, name):
                return getattr(self.original_patch_embed.proj, name)
            raise



class SpecVisionTransformer(nn.Module):
    def __init__(self, patch_size=4, img_size=128, embed_dim=768, reduced_channels=128, dynamic_img_size=False, depth=12, num_heads=6, mlp_ratio=4, pos_embed_type="learnable", init_pos_embed: bool = True, **kwargs):
        super(SpecVisionTransformer, self).__init__()
        
        self.spectral_adapter = SpectralAdapter()

        # Initialize Vision Transformer
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=reduced_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
        if "dynamic_img_size" in signature(VisionTransformer.__init__).parameters:
            vit_kwargs["dynamic_img_size"] = dynamic_img_size
        self.vit_core = VisionTransformer(
            **vit_kwargs
        )

        # === NEW: SELF-CONTAINED POSITIONAL EMBEDDING INITIALIZATION (gated) ===
        if init_pos_embed:
            if pos_embed_type == "learnable":
                init_fn = getattr(lightly_utils, "initialize_learnable_positional_embedding", None)
                if init_fn is not None:
                    init_fn(self.vit_core.pos_embed)
                    print("SpecViT: Initialized with learnable positional embeddings.")
                else:
                    print("SpecViT: lightly learnable positional init unavailable; using timm default.")
            elif pos_embed_type == "sincos":
                init_fn = getattr(lightly_utils, "initialize_2d_sine_cosine_positional_embedding", None)
                if init_fn is not None:
                    init_fn(
                        pos_embedding=self.vit_core.pos_embed,
                        has_class_token=self.vit_core.num_prefix_tokens > 0
                    )
                    print("SpecViT: Initialized with fixed sin-cos positional embeddings.")
                else:
                    print("SpecViT: lightly sin-cos positional init unavailable; using timm default.")
        
        self.num_features = self.vit_core.num_features

        wrapped_patch_embed = PatchEmbedWrapper(
            self.spectral_adapter, 
            self.vit_core.patch_embed
        )
        # 2. Replace the vit_core's patch_embed with our wrapped version.
        #    Now, any call to vit_core.forward() will automatically use our adapter.
        self.vit_core.patch_embed = wrapped_patch_embed
        
    
    @property
    def embed_dim(self) -> int:
        return self.vit_core.embed_dim

    @property
    def patch_size(self) -> int:
        # The patch_size is on the original_patch_embed inside the wrapper
        if isinstance(self.vit_core.patch_embed.original_patch_embed.patch_size, tuple):
            return self.vit_core.patch_embed.original_patch_embed.patch_size[0]
        return self.vit_core.patch_embed.original_patch_embed.patch_size

    @property
    def num_prefix_tokens(self) -> int:
        return self.vit_core.num_prefix_tokens

    @property
    def patch_embed(self) -> nn.Module:
        # Return the wrapped patch_embed, which is what MAEModule and MaskedVisionTransformerTIMM will see.
        return self.vit_core.patch_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This forward pass is now correct and self-contained. It works for MAE
        and any other task (like classification or segmentation).
        """
        # Simply call the vit_core's forward. It will use the wrapped patch embed internally.
        return self.vit_core(x)
    
    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.vit_core.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.vit_core.patch_embed(x)
        x = self.vit_core._pos_embed(x)
        x = self.vit_core.patch_drop(x)
        x = self.vit_core.norm_pre(x)
        for i, blk in enumerate(self.vit_core.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs
    
    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.vit_core.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.vit_core.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.vit_core.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.vit_core.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)
    
    def get_classifier(self):
        return self.vit_core.head

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vit_core, name)


class SpecViTSmall(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTBase(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTLarge(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTHuge(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4,
            **kwargs
        )
        
class SpecViTGiant(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1536,
            depth=40,
            num_heads=24,
            mlp_ratio=4,
            **kwargs
        )


    
