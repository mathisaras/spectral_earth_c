from typing import Union, List
import urllib.request
from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Base class for encoder."""

    def __init__(
        self,
        model_name: str,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_layers: list[int],
        output_dim: Union[int, List[int]],
        multi_temporal: bool,
        multi_temporal_output: bool,
        pyramid_output: bool,
        encoder_weights: Union[str, Path],
    ) -> None:
        """Initialize the Encoder.

        Args:
            model_name (str): name of the model.
            input_bands (dict[str, list[str]]): list of the input bands for each modality.
            dictionary with keys as the modality and values as the list of bands.
            input_size (int): size of the input image.
            embed_dim (int): dimension of the embedding used by the encoder.
            output_dim (int): dimension of the embedding output by the encoder, accepted by the decoder.
            multi_temporal (bool): whether the model is multi-temporal or not.
            multi_temporal (bool): whether the model output is multi-temporal or not.
            encoder_weights (str | Path): path to the encoder weights.
            download_url (str): url to download the model.
        """
        super().__init__()
        self.model_name = model_name
        self.input_bands = input_bands
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.output_layers = output_layers
        self.output_dim = (
            [output_dim for _ in output_layers]
            if isinstance(output_dim, int)
            else list(output_dim)
        )
        self.encoder_weights = encoder_weights
        self.multi_temporal = multi_temporal
        self.multi_temporal_output = multi_temporal_output

        self.pyramid_output = pyramid_output

    def load_encoder_weights(self, logger: Logger) -> None:
        """Load the encoder weights.

        Args:
            logger (Logger): logger to log the information.

        Raises:
            NotImplementedError: raise if the method is not implemented.
        """
        raise NotImplementedError

    def enforce_single_temporal(self):
        return
        # self.multi_temporal = False
        # self.multi_temporal_fusion = False

    def parameters_warning(
        self,
        missing: dict[str, torch.Size],
        incompatible_shape: dict[str, tuple[torch.Size, torch.Size]],
        logger: Logger,
    ) -> None:
        """Print warning messages for missing or incompatible parameters

        Args:
            missing (dict[str, torch.Size]): list of missing parameters.
            incompatible_shape (dict[str, tuple[torch.Size, torch.Size]]): list of incompatible parameters.
            logger (Logger): logger to log the information.
        """
        if missing:
            logger.warning(
                "Missing parameters:\n"
                + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items()))
            )
        if incompatible_shape:
            logger.warning(
                "Incompatible parameters:\n"
                + "\n".join(
                    "%s: expected %s but found %s" % (k, v[0], v[1])
                    for k, v in sorted(incompatible_shape.items())
                )
            )

    def freeze(self) -> None:
        """Freeze encoder's parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Foward pass of the encoder.

        Args:
            x (dict[str, torch.Tensor]): encoder's input structured as a dictionary:
            x = {modality1: tensor1, modality2: tensor2, ...}, e.g. x = {"optical": tensor1, "sar": tensor2}.
            If the encoder is multi-temporal (self.multi_temporal==True), input tensor shape is (B C T H W) with C the
            number of bands required by the encoder for the given modality and T the number of time steps. If the
            encoder is not multi-temporal, input tensor shape is (B C H W) with C the number of bands required by the
            encoder for the given modality.
        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            list[torch.Tensor]: list of the embeddings for each modality. For single-temporal encoders, the list's
            elements are of shape (B, embed_dim, H', W'). For multi-temporal encoders, the list's elements are of shape
            (B, C', T, H', W') with T the number of time steps if the encoder does not have any time-merging strategy,
            else (B, C', H', W') if the encoder has a time-merging strategy (where C'==self.output_dim).
        """
        raise NotImplementedError
                    