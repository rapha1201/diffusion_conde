import torch
import torch.nn as nn
from .blocks import ResBlock, DownBlock, UpBlock, Flatten, Unflatten

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels_list=[64, 128, 256, 512, 1024],
        res_blocks=[3, 3, 5, 3],
        embed_dim=256,
        activation=nn.SiLU,
        norm=nn.BatchNorm2d,
        latent_spatial_dim=None,
        final_activation=nn.Identity
    ):
        super().__init__()
        assert latent_spatial_dim is not None, "Provide latent_spatial_dim=(C, H, W)"

        C, H, W = latent_spatial_dim

        # stem
        layers = [
            nn.Conv2d(in_channels, channels_list[0], kernel_size=7, stride=2, padding=3, bias=False),
            norm(channels_list[0]),
            activation(),
        ]
        # down blocks
        for idx in range(len(channels_list) - 1):
            layers.append(DownBlock(
                channels_list[idx], channels_list[idx+1],
                res_blocks[idx], activation, norm
            ))
        self.conv = nn.Sequential(*layers)
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Linear(C*W*H, embed_dim)
        self.final_activation = final_activation()
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.conv(x)
        #x = self.pool(x)
        x = self.flatten(x)
        z = self.fc(x)
        return self.final_activation(z)

class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        channels_list=[64, 128, 256, 512, 1024],
        res_blocks=[3, 3, 5, 3],
        embed_dim=256,
        activation=nn.SiLU,
        norm=nn.BatchNorm2d,
        final_activation=None,
        latent_spatial_dim=None
    ):
        super().__init__()
        assert latent_spatial_dim is not None, "Provide latent_spatial_dim=(C, H, W)"

        C, H, W = latent_spatial_dim
        self.fc = nn.Linear(embed_dim, C * H * W)
        self.unflatten = Unflatten(C, H, W)

        layers = []
        for idx in range(len(channels_list) - 1):
            layers.append(UpBlock(
                channels_list[idx], channels_list[idx+1],
                res_blocks[idx], activation, norm
            ))
        # final up + conv
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(
            channels_list[-1], out_channels,
            kernel_size=7, stride=1, padding=3
        ))

        layers.append(final_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        return self.net(x)

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        input_shape=(128, 128),
        in_channels=1,
        out_channels=1,
        embed_dim=256,
        channels_list=[64, 128, 256, 512, 1024],
        res_blocks=[3, 3, 5, 3],
        activation=nn.SiLU,
        norm=nn.BatchNorm2d,
        encoder_final_activation=nn.Identity,
        decoder_final_activation=nn.Identity
    ):
        super().__init__()

        self.embed_dim = embed_dim

        down_factor = 2 ** len(channels_list)
        H, W = input_shape
        latent_h = H // down_factor
        latent_w = W // down_factor
        latent_dim = channels_list[-1]
        latent_spatial = (latent_dim, latent_h, latent_w)

        
        self.encoder = Encoder(
            in_channels=in_channels,
            channels_list=channels_list,
            res_blocks=res_blocks,
            embed_dim=embed_dim,
            activation=activation,
            norm=norm,
            latent_spatial_dim=latent_spatial,
            final_activation=encoder_final_activation
        )
        # compute latent spatial dims
        

        self.decoder = Decoder(
            out_channels=out_channels,
            channels_list=list(reversed(channels_list)),
            res_blocks=list(reversed(res_blocks)),
            embed_dim=embed_dim,
            activation=activation,
            norm=norm,
            final_activation=decoder_final_activation,
            latent_spatial_dim=latent_spatial
        )

    def forward(self, x, noise_mu=0, return_emb=False):
        z = self.encoder(x)
        z += torch.randn_like(z)*noise_mu
        if return_emb:
            return self.decoder(z), z
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
