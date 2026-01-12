import torch
import torch.nn as nn
from NeuralGraph.models.MLP import MLP
from typing import Optional, Tuple

class Calcium_Latent_Dynamics(nn.Module):
    """
    3-MLP model:
      - Encoder:  x∈R^{N}  -> (mu, logvar) ∈ R^{Z}×R^{Z}
      - Update:   z∈R^{Z}  -> z'∈R^{Z}  (applied latent_update_steps times if do_update=True)
      - Decoder:  z∈R^{Z}  -> x̂∈R^{N}

    Two forward modes:
      - forward(x, do_update=True):   encode -> (sample or mu) -> repeat update -> decode
      - forward_noupdate(x):          encode -> (sample or mu) -> decode

    If stochastic_latent=False or model.eval(), uses mu only (deterministic).
    Caches (mu, logvar) for optional KL regularization via kl_loss().
    """
    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        model_config = config.graph_model

        self.Z_dim = int(model_config.latent_dim)
        self.latent_update_steps = int(model_config.latent_update_steps)
        self.stochastic = bool(model_config.stochastic_latent)
        self.latent_init_std = float(model_config.latent_init_std)
        self.use_residual_update = getattr(model_config, "use_residual_update", True)

        # Encoder: x (N) -> [mu, logvar] (2Z)
        self.encoder = MLP(
            input_size=model_config.input_size_encoder,       # should be N (n_neurons)
            output_size=2 * self.Z_dim,
            nlayers=model_config.n_layers_encoder,
            hidden_size=model_config.hidden_dim_encoder,
            device=self.device,
        )

        # Update: z (Z) -> z' (Z)
        self.update_latent = MLP(
            input_size=self.Z_dim,
            output_size=self.Z_dim,
            nlayers=model_config.latent_n_layers_update,      # NEW field in config
            hidden_size=model_config.latent_hidden_dim_update,# NEW field in config
            device=self.device,
        )

        # Decoder: z (Z) -> x̂ (N)
        self.decoder = MLP(
            input_size=self.Z_dim,
            output_size=model_config.output_size_decoder,     # should be N (n_neurons)
            nlayers=model_config.n_layers_decoder,
            hidden_size=model_config.hidden_dim_decoder,
            device=self.device,
        )

        self.last_mu: Optional[torch.Tensor] = None
        self.last_logvar: Optional[torch.Tensor] = None

        self.to(self.device)

    # ---- public helpers -------------------------------------------------------
    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        mu, logvar = enc.split(self.Z_dim, dim=-1)  # FIXED: use self.Z_dim
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if (not self.stochastic) or (not self.training):
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self):
        """KL(q(z|x) || N(0, I)) for the most recent forward; returns scalar."""
        if self.last_mu is None or self.last_logvar is None:
            return torch.tensor(0.0, device=self.device)
        mu, logvar = self.last_mu, self.last_logvar
        # 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar ) averaged over batch
        kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=-1).mean()
        return kl

    # ---- forwards -------------------------------------------------------------
    @torch.no_grad()
    def forward_noupdate(self, x):

        mu, logvar = self.encode(x)
        z0 = self.reparameterize(mu, logvar)
        pred = self.decode(z0)

        return pred, mu, logvar

    def forward(self, x, do_update: bool = True):
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if do_update and self.latent_update_steps > 0:
            for _ in range(self.latent_update_steps):
                dz = self.update_latent(z)
                z = z + dz if self.use_residual_update else dz

        pred = self.decode(z)

        return pred, mu, logvar
