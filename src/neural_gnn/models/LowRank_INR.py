# Low-rank factorization for external input representation

import numpy as np
import torch
import torch.nn as nn


class LowRankINR(nn.Module):
    """
    Low-rank factorization for external_input: f(t) = U @ V

    For rank-k approximation of (T, N) signal:
        params = k × (T + N) << T × N

    This is NOT a neural network - it's a direct matrix factorization
    where U (n_frames, rank) and V (rank, n_neurons) are learnable parameters.
    """

    def __init__(self, n_frames, n_neurons, rank=64, init_data=None):
        super().__init__()
        self.n_frames = n_frames
        self.n_neurons = n_neurons
        self.rank = rank

        if init_data is not None:
            # SVD initialization for faster convergence
            U, S, Vt = np.linalg.svd(init_data, full_matrices=False)
            sqrt_S = np.sqrt(S[:rank])
            self.U = nn.Parameter(torch.tensor(U[:, :rank] * sqrt_S, dtype=torch.float32))
            self.V = nn.Parameter(torch.tensor(sqrt_S[:, None] * Vt[:rank, :], dtype=torch.float32))
        else:
            # random initialization
            self.U = nn.Parameter(torch.randn(n_frames, rank) * 0.01)
            self.V = nn.Parameter(torch.randn(rank, n_neurons) * 0.01)

    def forward(self, t_indices=None):
        """
        Args:
            t_indices: (batch,) int tensor of frame indices, or None for full matrix
        Returns:
            (batch, n_neurons) or (n_frames, n_neurons)
        """
        if t_indices is None:
            return self.U @ self.V
        return self.U[t_indices] @ self.V  # (batch, rank) @ (rank, N) = (batch, N)

    def get_compression_ratio(self):
        full_size = self.n_frames * self.n_neurons
        param_size = self.rank * (self.n_frames + self.n_neurons)
        return full_size / param_size
