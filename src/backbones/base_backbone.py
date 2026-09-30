# /backbones/base.py
from abc import ABC, abstractmethod
from typing import Union, Sequence, Tuple
import torch
import torch.nn as nn

class BaseBackbone(nn.Module, ABC):
    """
    Abstract Base Class for any backbone to be used with the MAEModule.
    It defines the contract that all backbones must follow.
    """
    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """The embedding dimension of the transformer."""
        raise NotImplementedError

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """The size of a single patch."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def num_prefix_tokens(self) -> int:
        """The number of prefix tokens (e.g., class token)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def patch_embed(self) -> nn.Module:
        """The patch embedding layer. Expected to have a 'num_patches' attribute."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass. For MAE compatibility, this should return the 
        sequence of token embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Returns intermediate layer outputs for tasks like segmentation.
        """
        raise NotImplementedError