"""
NeuralRenderer.py - Differentiable Neural Renderer with Soft Voronoi + MLP

Simplified architecture:
- Soft Voronoi: Outputs a continuous scalar field f(x, y)
- MLP: Refines the field value to match matplotlib-style dot rendering

Design:
1. Takes neuron positions (N, 2) and activities (N,) from SIREN
2. Soft Voronoi evaluates at query points (M, 2) → (M,) field values
3. MLP processes scalar field → (M,) refined activities

Learnable parameters:
- Affine transform (4 params): ax, ay, tx, ty
- Kernel params (2): R ≡ sigma (radius), β (edge sharpness)
- MLP weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftVoronoi(nn.Module):
    """
    Soft differentiable Voronoi tessellation with smooth bump kernel.

    For each neuron i with position p_i and activity a_i, the contribution at
    query point x is:

        r_i = ||x - T(p_i)||       (T is an affine transform)
        u_i = r_i / R
        b_i = (1 - u_i^2)^p       (polynomial bump)
        m_i = sigmoid(β * (R - r_i))
        k_i = m_i * b_i           (smooth "soft disk" kernel)

    The final field value is:
        f(x) = sum_i a_i * k_i

    This gives nearly flat circular blobs with soft edges and is fully
    differentiable.
    """

    def __init__(self, resolution=512, sigma_init=0.02, beta_init=30.0, p=10):
        """
        Args:
            resolution: Default grid resolution (for grid generation)
            sigma_init: Initial disk radius R in normalized coordinates
            beta_init: Initial β (edge sharpness parameter)
            p: Exponent of the polynomial bump (controls interior flatness)
        """
        super().__init__()
        self.resolution = resolution
        self.eps = 1e-8  # For numerical stability
        self.p = float(p)

        # Learnable affine transformation (centered at 0.5, 0.5)
        # x' = ax * (x - 0.5) + tx + 0.5
        # y' = ay * (y - 0.5) + ty + 0.5
        self.affine_ax = nn.Parameter(torch.tensor(0.9))
        self.affine_ay = nn.Parameter(torch.tensor(0.9))
        self.affine_tx = nn.Parameter(torch.tensor(0.0))
        self.affine_ty = nn.Parameter(torch.tensor(0.0))

        # Learnable kernel parameters:
        # sigma ≡ disk radius R, beta ≡ edge sharpness
        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    # ------------------------------------------------------------------ #
    # Affine transform utilities
    # ------------------------------------------------------------------ #
    def apply_affine(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply the learnable affine transform to positions in [0, 1]^2.

        Args:
            positions: (N, 2) tensor in [0, 1]

        Returns:
            positions_transformed: (N, 2) tensor in [0, 1] (not strictly clamped)
        """
        x = positions[:, 0]
        y = positions[:, 1]

        # Center at 0.5, scale, then translate back
        x_centered = x - 0.5
        y_centered = y - 0.5

        x_trans = self.affine_ax * x_centered + self.affine_tx + 0.5
        y_trans = self.affine_ay * y_centered + self.affine_ty + 0.5

        return torch.stack([x_trans, y_trans], dim=-1)

    # ------------------------------------------------------------------ #
    # Forward: disk-like splatting
    # ------------------------------------------------------------------ #
    def forward(
        self,
        positions: torch.Tensor,   # (N, 2) in [0, 1]
        activities: torch.Tensor,  # (N,) or (N, 1)
        query_points: torch.Tensor # (M, 2) in [0, 1]
    ) -> torch.Tensor:
        """
        Evaluate disk-like splatting field at query points.

        Args:
            positions: (N, 2) neuron positions in [0, 1]
            activities: (N,) or (N, 1) neuron activities
            query_points: (M, 2) query coordinates in [0, 1]

        Returns:
            field_values: (M,) tensor with scalar field values
        """
        if activities.dim() == 2:
            activities = activities.squeeze(-1)  # (N, 1) -> (N,)

        N = positions.shape[0]
        M = query_points.shape[0]
        device = positions.device

        # Affine-transformed neuron positions
        positions_transformed = self.apply_affine(positions)  # (N, 2)

        # Expand for vectorized distance computation:
        # query_points: (M, 2)
        # positions_transformed: (N, 2)
        # -> dx, dy shape: (M, N)
        qp = query_points.to(device)
        pt = positions_transformed.to(device)
        acts = activities.to(device)

        dx = qp[:, None, 0] - pt[None, :, 0]  # (M, N)
        dy = qp[:, None, 1] - pt[None, :, 1]  # (M, N)

        # Distances r_i(x)
        r = torch.sqrt(dx * dx + dy * dy + self.eps)  # (M, N)

        # Disk radius R and edge sharpness β (ensure positivity by softplus)
        R = F.softplus(self.sigma) + 1e-6        # scalar > 0
        beta = F.softplus(self.beta) + 1e-6      # scalar > 0

        # Polynomial bump: (1 - (r/R)^2)^p, mainly active for r < R
        u = r / R                                # (M, N)
        bump = (1.0 - u * u).clamp(min=0.0)      # (M, N)
        bump = bump ** self.p                    # (M, N)

        # Soft disk mask: sigmoid(beta * (R - r)) ≈ 1 inside, 0 outside
        mask = torch.sigmoid(beta * (R - r))     # (M, N)

        # Final smooth disk kernel
        k = bump * mask                          # (M, N)

        # Weighted sum over neurons
        # field_values = torch.matmul(k, acts)     # (M,)

        num = torch.matmul(k, acts)             # (M,)
        den = k.sum(dim=1) + 1e-8               # (M,)
        field_values = num / den


        return field_values

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_learnable_params(self):
        """
        Return a dict of the key learnable parameters for logging / visualization.
        """
        # Use detached CPU scalars for readability
        return {
            'sigma': float(self.sigma.detach().cpu().item()),
            'beta': float(self.beta.detach().cpu().item()),
            'affine_ax': float(self.affine_ax.detach().cpu().item()),
            'affine_ay': float(self.affine_ay.detach().cpu().item()),
            'affine_tx': float(self.affine_tx.detach().cpu().item()),
            'affine_ty': float(self.affine_ty.detach().cpu().item()),
        }


class NeuralRenderer(nn.Module):
    """
    Complete neural renderer: Soft Voronoi + MLP refinement.

    Architecture:
    1. SoftVoronoi: positions, activities → scalar field f(x, y)
    2. MLP: f → refined activity (no forced output activation here; training
       objective will shape the range).

    API expected by NGP_trainer.py:
        renderer = NeuralRenderer(resolution=512, sigma_init=0.01, beta_init=100.0)

        # positions: (N, 2) in [0, 1]
        # activities: (N,) from SIREN
        # query_points: (M, 2) in [0, 1]
        pred = renderer(positions, activities, query_points)  # (M,)
    """

    def __init__(self, resolution=512, sigma_init=0.01, beta_init=100.0,
                 hidden_dim=64, bump_power=4):
        super().__init__()
        self.resolution = resolution

        # Soft Voronoi front-end
        self.splatting = SoftVoronoi(
            resolution=resolution,
            sigma_init=sigma_init,
            beta_init=beta_init,
            p=bump_power
        )

        # Simple MLP taking scalar field value as input
        # Input: 1D (field), Output: 1D (refined activity)
        # self.mlp = nn.Sequential(
        #     nn.Linear(1, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, 1)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),            # keeps low values > 0, smooth
            nn.Linear(hidden_dim, 1),
            nn.Softplus()             # second softplus avoids collapse to zero
        )
        

    # ------------------------------------------------------------------ #
    # Utility: grid creation
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_grid(resolution, device=None):
        """
        Create a regular 2D grid of coordinates in [0, 1] x [0, 1].

        Args:
            resolution: int, number of pixels along each axis
            device: torch device

        Returns:
            coords: (resolution * resolution, 2) tensor
        """
        if device is None:
            device = torch.device("cpu")

        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, resolution, device=device),
            torch.linspace(0.0, 1.0, resolution, device=device),
            indexing='ij'
        )
        coords = torch.stack([xs, ys], dim=-1)  # (res, res, 2)
        coords = coords.view(-1, 2)             # (res*res, 2)
        return coords

    # ------------------------------------------------------------------ #
    # Forward: positions, activities, query_points → predicted activity
    # ------------------------------------------------------------------ #
    def forward(
        self,
        positions: torch.Tensor,    # (N, 2) in [0, 1]
        activities: torch.Tensor,   # (N,) or (N, 1)
        query_points: torch.Tensor  # (M, 2) in [0, 1]
    ) -> torch.Tensor:
        """
        Render activities at query points using disk splatting + MLP.

        Args:
            positions: Neuron positions (N, 2) in [0, 1]
            activities: Neuron activities (N,) or (N, 1)
            query_points: Query coordinates (M, 2) in [0, 1]

        Returns:
            output: (M,) tensor of predicted activity values
        """
        # Front-end: disk-like splatting scalar field
        field_values = self.splatting(positions, activities, query_points)  # (M,)

        # MLP expects shape (M, 1)
        field_values_in = field_values.unsqueeze(-1)  # (M, 1)

        # Refined activity
        output = self.mlp(field_values_in).squeeze(-1)  # (M,)

        return output
    
    
    def forward_splatting_only(
        self,
        positions: torch.Tensor,    # (N, 2)
        activities: torch.Tensor,   # (N,) or (N, 1)
        query_points: torch.Tensor  # (M, 2)
    ) -> torch.Tensor:
        """
        Convenience wrapper used in stage 4:
        run only the splatting front-end (no MLP refinement).
        """
        # Directly use the DiskSplatting module, which has apply_affine etc.
        field_values = self.splatting(
            positions=positions,
            activities=activities,
            query_points=query_points,
        )  # (M,)

        return field_values


    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_num_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learnable_params(self):
        """Return current values of key learnable parameters."""
        splatting_params = self.splatting.get_learnable_params()
        return {
            **splatting_params,
            'total_params': self.get_num_parameters()
        }
