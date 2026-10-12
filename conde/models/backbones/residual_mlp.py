import torch
import torch.nn as nn
from .blocks import ResidualMLPBlock

class ResidualMLP(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 emb_size: int,
                 dropout_p: float = 0.2,
                 num_residual_layers: int = 3,
                 activation: nn.Module = nn.SiLU,
                 final_activation: nn.Module = nn.Identity,
                 norm: nn.Module = nn.BatchNorm1d):
        super().__init__()

        self.input_layer = nn.Linear(latent_dim, hidden_dim)
        self.norm = norm(hidden_dim)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_p)

        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, activation = activation, dropout_p = dropout_p, norm = norm)
            for _ in range(num_residual_layers)
        ])

        
        self.output_layer = nn.Linear(hidden_dim, emb_size)
        self.final_activation = final_activation()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.norm(self.dropout(x))

        for block in self.residual_blocks:
            x = block(x)

        x = self.output_layer(x)
        x = self.final_activation(x)
        return x


