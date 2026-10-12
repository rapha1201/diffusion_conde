import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .blocks import UpBlock, Flatten, Unflatten, ConvNextV2Block, ConvNextUpBlock, ConvNextDownBlock

from timm.models.layers import trunc_normal_

class ConvNextEncoder(nn.Module):
    """
    ConvNeXt V2 encoder com N downblocks definidos por listas.
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 1024,
        res_blocks:   list[int] = [3, 3, 9, 3],
        channels_list:     list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        down_rates: list[int] = None,              # se não passado, será [4] + [2]*(len(depths)-1)
        layer_scale_init_value: float = 1e-6,
        latent_spatial_dim: tuple[int,int] = (4,4)
    ):
        super().__init__()
        H, W = latent_spatial_dim
        assert len(res_blocks) == len(channels_list), "depths e dims devem ter mesmo comprimento"

        # default down_rates
        if down_rates is None:
            down_rates = [4] + [2] * (len(res_blocks) - 1)
        assert len(down_rates) == len(res_blocks), "down_rates deve ter mesmo tamanho de depths"

        # agendar drop_path para todos os blocos
        total_blocks = sum(res_blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        # criar dinamicamente os downblocks
        self.stages = nn.ModuleList()
        in_ch = in_channels
        for i, (n_blk, out_ch, dr) in enumerate(zip(res_blocks, channels_list, down_rates)):
            # fatia de dp_rates para este estágio
            rates_i = dp_rates[cur : cur + n_blk]
            cur += n_blk

            block = ConvNextDownBlock(
                in_ch=in_ch,
                out_ch=out_ch,
                num_blocks=n_blk,
                drop_path_rates=rates_i,
                layer_scale_init_value=layer_scale_init_value,
                down_rate=dr
            )
            self.stages.append(block)
            in_ch = out_ch

        # cabeça 1D
        self.flatten = nn.Flatten(1)  
        self.fc      = nn.Linear(channels_list[-1] * H * W, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.flatten(x)
        return self.fc(x)
 

class ConvNextDecoder(nn.Module):
    """
    Decoder simétrico usando apenas uma lista de ConvNextUpBlock
    e, ao final, um único Upsample×2 + Conv1×1.
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 1024,
        res_blocks:   list[int] = [3, 3, 9, 3],
        channels_list:     list[int] = [96,192,384,768],
        drop_path_rate: float = 0.0,
        latent_spatial_dim: tuple[int,int] = (4, 4),
    ):
        super().__init__()
        H, W = latent_spatial_dim

        # 1) Projeção 1D → mapa espacial
        self.fc = nn.Linear(embed_dim, channels_list[-1] * H * W)
        self.unflatten = nn.Unflatten(1, (channels_list[-1], H, W))

        # 2) Agenda invertida de drop_path
        all_dp = [x.item() for x in torch.linspace(0, drop_path_rate, sum(res_blocks))]
        dp_rev = all_dp[::-1]
        cur = 0

        # 3) Cria TODOS os up_blocks de len(depths) estágios (inclusive i=0)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(res_blocks))):
            n_blk = res_blocks[i]
            rates = dp_rev[cur:cur + n_blk]
            cur += n_blk

            in_ch  = channels_list[i]
            out_ch = channels_list[i-1] if i > 0 else channels_list[0]  
            # para i=0, out_ch=dims[0], replicando o comportamento anterior
            self.up_blocks.append(
                ConvNextUpBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    num_blocks=n_blk,
                    drop_path_rates=rates,
                    up_rate=2
                )
            )

        # 4) Último upsample×2 + conv1×1 para chegar em in_chans
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels_list[0], in_channels, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, embed_dim)
        x = self.fc(z)
        x = self.unflatten(x)            # → (B, dims[-1], H, W)
        for up in self.up_blocks:
            x = up(x)                    # ↑2 em cada bloco
        x = self.final_up(x)             # ↑2 final + conv
        return x

        

class ConvNextAutoencoder(nn.Module):
    """
    Combina o ConvNextEncoder e o ConvNextDecoder, calculando automaticamente
    o latent_spatial_dim a partir de input_size e down_rates.
    """
    def __init__(
        self,
        in_channels: int,
        input_size: tuple[int, int] = (128, 128),         # (H, W) da imagem de entrada
        embed_dim: int = 1024,
        res_blocks:   list[int] = [3, 3, 9, 3],
        channels_list:     list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        down_rates: list[int]  = None,       # se None: [4] + [2]*(len(depths)-1)
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        H_in, W_in = input_size
        N = len(res_blocks)

        # defaults para down_rates
        if down_rates is None:
            down_rates = [4] + [2] * (N - 1)
        assert len(down_rates) == N, "down_rates deve ter mesmo comprimento de depths"

        # calcula o fator total de downsampling
        total_factor = math.prod(down_rates)
        assert H_in % total_factor == 0 and W_in % total_factor == 0, (
            f"input_size {input_size} deve ser divisível por total_factor={total_factor}"
        )
        latent_H = H_in // total_factor
        latent_W = W_in // total_factor
        latent_spatial_dim = (latent_H, latent_W)
        self.embed_dim = embed_dim

        # encoder
        self.encoder = ConvNextEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            res_blocks=res_blocks,
            channels_list=channels_list,
            drop_path_rate=drop_path_rate,
            down_rates=down_rates,
            layer_scale_init_value=layer_scale_init_value,
            latent_spatial_dim=latent_spatial_dim
        )

        # decoder
        self.decoder = ConvNextDecoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            res_blocks=res_blocks,
            channels_list=channels_list,
            drop_path_rate=0,
            latent_spatial_dim=latent_spatial_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, noise_mu=0):
        z = self.encoder(x)
        z += torch.randn_like(z)*noise_mu
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    