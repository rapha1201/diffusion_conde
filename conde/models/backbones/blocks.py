import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath

############################################
## ResNet Blocks
#############################################

class ResBlock(nn.Module):
    def __init__(self, channels, activation=nn.SiLU, norm=nn.BatchNorm2d, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            norm(channels),
            activation(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            activation(),  
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.net(x) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks=2,
                 activation=nn.SiLU, norm=nn.BatchNorm2d, dropout_p=0.2):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        ]
        for _ in range(res_blocks):
            layers.append(ResBlock(out_channels, activation, norm, dropout_p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_blocks=2,
        activation=nn.SiLU, norm=nn.BatchNorm2d, final_activation=None
    ):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        for _ in range(res_blocks):
            layers.append(ResBlock(out_channels, activation, norm))
        if final_activation:
            layers.append(final_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)

class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, activation = nn.SiLU, dropout_p = 0, norm = nn.BatchNorm1d):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = norm(hidden_dim)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.layer2(out)
        out = self.dropout(out)

        out = self.norm(out)
        return self.activation(out + residual)


#############################################
## ConvNext Blocks
#############################################

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextV2Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNextDownBlock(nn.Module):
    """
    Down‐block = (Norm + 2×2 stride‐2 Conv) → N × (ConvNeXtV2Block)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        drop_path_rates: list[float],
        layer_scale_init_value: float = 1e-6,
        down_rate = 2
    ):
        super().__init__()
        # 1) downsample
        self.down = nn.Sequential(
            LayerNorm(in_ch, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_ch, out_ch, kernel_size=down_rate, stride=down_rate),
        )
        # 2) ConvNeXtV2 blocks at this resolution
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ConvNextV2Block(
                    dim=out_ch,
                    drop_path=drop_path_rates[i]
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.blocks(x)
        return x


class ConvNextUpBlock(nn.Module):
    """
    Up‐block = (nearest‐upsample ×2 → 1×1 conv + Norm) → N × (ConvNeXtV2Block)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        drop_path_rates: list[float],
        up_rate=2
    ):
        super().__init__()
        # 1) upsample
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=up_rate, mode="bilinear"),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            LayerNorm(out_ch, eps=1e-6, data_format="channels_first"),
        )
        # 2) ConvNeXtV2 blocks at this resolution
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ConvNextV2Block(
                    dim=out_ch,
                    drop_path=drop_path_rates[i]
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.up(x)
        x = self.blocks(x)
        
        return x
    

