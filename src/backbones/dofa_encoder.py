# Adapted from: https://github.com/VMarsocci/pangaea-bench
# Which is in turn based on: https://github.com/zhu-xlab/DOFA
# Weights downloaded from: https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_base_e100.pth

from functools import partial
from logging import Logger
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.vision_transformer import Block

from ..backbones.base import Encoder
from src.utils.sensor_registry import SensorRegistry
from src.utils.spectral_metadata import load_spectral_metadata


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()

        self.wv_planes = wv_planes
        self.inter_dim = inter_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim

        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)

        waves = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wvs * 1000
        ).float()
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)

        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class DOFA_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2403.15356
    Attributes:
        output_layers (int | list[int]): The layers from which to extract the output.
        img_size (int): The size of the input image.
        wv_planes (int): The number of wavelet planes.
        wave_list (dict[str, dict[str, float]]): A dictionary containing wavelet information for each band.
        return_all_tokens (bool): Whether to return all tokens or not.
        embed_dim (int): The embedding dimension.
        patch_size (int): The size of each patch.
        use_norm (bool): Whether to use normalization or not.
        wv_list (list[float]): A list of wavelet values for each band.
        norm (nn.Module): The normalization layer.
        patch_embed (Dynamic_MLP_OFA): The patch embedding layer.
        num_patches (int): The number of patches in the input image.
        cls_token (nn.Parameter): The class token parameter.
        pos_embed (nn.Parameter): The positional embedding parameter.
        blocks (nn.ModuleList): A list of Transformer blocks.
    Methods:
        __init__(encoder_weights, input_bands, input_size, embed_dim, output_layers, wave_list, patch_size=16, depth=12, num_heads=16, wv_planes=128, return_all_tokens=True, mlp_ratio=4., use_norm=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
            Initializes the DOFA_Encoder with the given parameters.
        forward(image):
            Forward pass of the encoder. Takes an input image and returns the encoded output.
        load_encoder_weights(logger):
            Loads the encoder weights from a pretrained model and logs any missing or incompatible parameters.
    """

    def __init__(
        self,
        encoder_weights: Union[str, List[Path]],
        input_bands: Optional[dict[str, list[str]]],
        input_size: int,
        embed_dim: int,
        output_dim: Union[int, List[int]],
        output_layers: Union[int, List[int]],
        wave_list: Optional[dict[str, list[float]]],
        patch_size=16,
        depth=12,
        num_heads=16,
        wv_planes=128,
        return_all_tokens=True,
        mlp_ratio=4.0,
        use_norm=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sensor="enmap",
        sensor_config_name: Optional[str] = None,
    ):
        super().__init__(
            model_name="dofa_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
        )

        self.output_layers = output_layers
        self.img_size = input_size
        self.wv_planes = wv_planes
        self.wave_list = wave_list
        self.return_all_tokens = return_all_tokens
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_norm = use_norm
        self.sensor = str(sensor_config_name or sensor).lower()
        if self.wave_list is not None:
            self.wv_list = self.wave_list[self.sensor]
        else:
            sensor_cfg = SensorRegistry.get(self.sensor)
            metadata = load_spectral_metadata(sensor_cfg, view="processed")
            self.wv_list = (metadata.band_centers_nm * 0.001).astype("float32").tolist()
        self.num_features = embed_dim
        self.in_chans = len(self.wv_list)

        self.norm = norm_layer(self.embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.num_patches = (self.img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        # Pangaea dataset loads in (C, T, H, W)
        # embed patches
        #x = [image[m].squeeze(2) for m in self.input_bands.keys()] (C, H, W)
        #x = torch.cat(x, dim=1)

        wavelist = torch.tensor(self.wv_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        # pos_embed shape is (1, num_patches+1, embed_dim)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)
            if i in self.output_layers:
                out = (
                    x[:, 1:]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.img_size // self.patch_size,
                        self.img_size // self.patch_size,
                    )
                    .contiguous()
                )

                output.append(out)
                
        if len(self.output_layers) == 0:
            # Check if we have a classifier head which could have been aded as model.classifier
            if hasattr(self, "classifier"):
                x = self.classifier(x[:, 0])
                return x
            else:
                return x[:, 0]
        else:
            return output        
        
    def _resolve_output_layers(
        self,
        n: Optional[Union[int, Sequence[int]]] = None,
    ) -> list[int]:
        if n is None:
            layers = self.output_layers
        elif isinstance(n, int):
            layers = list(range(len(self.blocks) - n, len(self.blocks))) if n > 0 else self.output_layers
        else:
            layers = list(n)
        if isinstance(layers, int):
            return [layers]
        return [int(layer) for layer in layers]

    def get_intermediate_layers(
        self,
        x,
        n: Optional[Union[int, Sequence[int]]] = None,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ):
        # Pangaea dataset loads in (C, T, H, W)
        # embed patches
        #x = [image[m].squeeze(2) for m in self.input_bands.keys()] (C, H, W)
        #x = torch.cat(x, dim=1)

        wavelist = torch.tensor(self.wv_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        # pos_embed shape is (1, num_patches+1, embed_dim)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        take_layers = set(self._resolve_output_layers(n))

        # apply Transformer blocks
        output = []
        prefix_output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)
            if i in take_layers:
                out = x[:, 1:]
                prefix_output.append(x[:, :1])
                if reshape:
                    out = (
                        out
                        .permute(0, 2, 1)
                        .view(
                            x.shape[0],
                            -1,
                            self.img_size // self.patch_size,
                            self.img_size // self.patch_size,
                        )
                        .contiguous()
                    )
                output.append(out)

        if return_prefix_tokens:
            return tuple(zip(output, prefix_output))
        return output            
        
    def load_encoder_weights(self, logger: Logger) -> None:
        print("Loading encoder weights")
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.load_state_dict(pretrained_encoder, strict=False)
        print("Loaded encoder weights")
        print("Missing parameters:")
        print(missing)
        print("Incompatible parameters:")
        print(incompatible_shape)
        #self.parameters_warning(missing, incompatible_shape, logger)


def DofaBasePatch16(**kwargs):
    kwargs.pop("patch_size", None)
    model = DOFA_Encoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), output_dim=768, **kwargs)
    return model

def DofaLargePatch16(**kwargs):
    kwargs.pop("patch_size", None)
    model = DOFA_Encoder(
        patch_size=16, embed_dim=1024, depth=12, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), output_dim=1024, **kwargs)
    return model
