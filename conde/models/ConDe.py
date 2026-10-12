import torch
import torch.nn as nn

class BaseMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims = None, n_layers=2):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims is not None else out_dim
        self.n_layers = n_layers
        self.in_dim=in_dim
        self.out_dims = out_dim

        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers += [nn.Linear(in_dim, self.hidden_dims), nn.SiLU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(self.hidden_dims, self.hidden_dims), nn.SiLU()]
            layers.append(nn.Linear(self.hidden_dims, out_dim))

        self.f = nn.Sequential(*layers)
        


class ProjectionHead(BaseMLP):
    def __init__(self, in_dim, out_dim, hidden_dims = None, n_layers=2):
        super().__init__(in_dim, out_dim, hidden_dims = None, n_layers=n_layers)


    def forward(self, x):
        return self.f(x)


class ResidualHead(BaseMLP):
    def __init__(self, in_dim, out_dim, hidden_dims = None, n_layers=2):
        super().__init__(in_dim, out_dim, hidden_dims = None, n_layers=n_layers)
        self.z_norm = nn.LayerNorm(in_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, z):
        gate = self.sigmoid(self.f(self.z_norm(z)))
        return gate*z, gate


class ConDe(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_semantic_heads,
        head_class=ProjectionHead,
        residual_class=ResidualHead,
        encoder = None,
        normalize=False
    ):
        """
        Args:
            latent_dim: dim of Z
            n_semantic_heads: number of semantic heads
            head_dim: dimension of semantic heads (default = latent_dim)
            residual_dim: dimension of residual vector R (default = latent_dim)
            head_class: class used to build semantic heads
            residual_class: class used to build residual head
        """
        super().__init__()

        # dimensions
        self.latent_dim = latent_dim

        self.residual_head = residual_class(latent_dim, latent_dim, n_layers=4)

        self.semantic_heads = nn.ModuleList([
            head_class(self.latent_dim, latent_dim, n_layers=4)
            for _ in range(n_semantic_heads)
        ]) if n_semantic_heads > 0 else None

        self.encoder = encoder
        self.normalize=normalize

    def forward(self, Z):
        """
        Args:
            Z: latent embedding
        Z -> {R, Z_sem, S_list, S_sum}
        """
        if self.encoder is not None:
            if self.normalize:
                mean = Z.mean(dim=0, keepdim=True)
                std = Z.std(dim=0, keepdim=True) + 1e-8
                Z = (Z - mean) / std
            Z = self.encoder(Z)
        R, gate = self.residual_head(Z)   # [B, residual_dim]
        Z_sem = Z - R 

        if self.semantic_heads is not None:
            S_list = [head(Z_sem) for head in self.semantic_heads]  # each [B, head_dim]
        else:
            S_list = [Z_sem]

        return {
            "R": R,
            "gate": gate,
            "Z_sem": Z_sem,
            "S_list": S_list,
            "Z": Z
        }
    
