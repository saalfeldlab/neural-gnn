"""
Neural ODE wrapper for Signal_Propagation.

Uses torchdiffeq's adjoint method for memory-efficient training:
- Memory O(1) in rollout steps L (vs O(L) for BPTT)
- Backward pass uses adjoint ODE solve
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from torch_geometric.loader import DataLoader


class GNNODEFunc_Signal(nn.Module):
    """
    Wraps GNN model as ODE vector field: du/dt = f(t, u).

    Column layout (as in Signal_Propagation):
        - Column 6: neural activity (u)
        - Column 8: field
    """

    def __init__(self, model, data_template, data_id, neurons_per_sample, batch_size,
                 x_list=None, run=0, device=None, k_batch=None):
        super().__init__()
        self.model = model
        self.data_template = data_template
        self.data_id = data_id
        self.neurons_per_sample = neurons_per_sample
        self.batch_size = batch_size
        self.x_list = x_list
        self.run = run
        self.device = device or torch.device('cpu')
        self.k_batch = k_batch
        self.delta_t = 1.0

    def set_time_params(self, delta_t):
        self.delta_t = delta_t

    def forward(self, t, u):
        """Compute du/dt = GNN(u). Called by ODE solver at each integration step."""
        data = self.data_template.clone()
        u_reshaped = u.view(-1, 1)

        x_new = data.x.clone()
        x_new[:, 6:7] = u_reshaped

        k_offset = int((t / self.delta_t).item()) if t.numel() == 1 else 0

        if self.x_list is not None:
            for b in range(self.batch_size):
                start_idx = b * self.neurons_per_sample
                end_idx = (b + 1) * self.neurons_per_sample
                k_current = int(self.k_batch[b].item()) + k_offset

                if k_current < len(self.x_list[self.run]):
                    x_next = torch.tensor(
                        self.x_list[self.run][k_current],
                        dtype=torch.float32,
                        device=self.device
                    )
                    x_new[start_idx:end_idx, 8:9] = x_next[:, 8:9]

        data.x = x_new

        k_current_tensor = self.k_batch + k_offset
        k_expanded = k_current_tensor.repeat_interleave(self.neurons_per_sample).unsqueeze(1)

        pred = self.model(
            data,
            data_id=self.data_id,
            k=k_expanded,
            return_all=False
        )

        return pred.view(-1)


def integrate_neural_ode_Signal(model, u0, data_template, data_id, time_steps, delta_t,
                                neurons_per_sample, batch_size, x_list=None, run=0,
                                device=None, k_batch=None, ode_method='dopri5',
                                rtol=1e-4, atol=1e-5, adjoint=True, noise_level=0.0):
    """
    Integrate GNN dynamics using Neural ODE.

    Returns:
        u_final : final state (N,)
        u_trajectory : states at all time points (time_steps+1, N)
    """

    solver = odeint_adjoint if adjoint else odeint

    ode_func = GNNODEFunc_Signal(
        model=model,
        data_template=data_template,
        data_id=data_id,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        x_list=x_list,
        run=run,
        device=device,
        k_batch=k_batch
    )
    ode_func.set_time_params(delta_t)

    t_span = torch.linspace(
        0, time_steps * delta_t, time_steps + 1,
        device=device, dtype=u0.dtype
    )

    u_trajectory = solver(
        ode_func,
        u0.flatten(),
        t_span,
        method=ode_method,
        rtol=rtol,
        atol=atol
    )

    if noise_level > 0 and model.training:
        u_trajectory = u_trajectory + noise_level * torch.randn_like(u_trajectory)

    u_final = u_trajectory[-1]

    return u_final, u_trajectory


def neural_ode_loss_Signal(model, dataset_batch, x_list, run, k_batch,
                           time_step, batch_size, n_neurons, ids_batch,
                           delta_t, device, data_id=None, y_batch=None,
                           noise_level=0.0, ode_method='dopri5',
                           rtol=1e-4, atol=1e-5, adjoint=True):
    """
    Compute loss using Neural ODE integration.
    Replaces explicit autoregressive rollout in data_train_signal.
    """

    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
    data_template = next(iter(batch_loader))

    u0 = data_template.x[:, 3].flatten()
    neurons_per_sample = dataset_batch[0].x.shape[0]

    k_per_sample = torch.tensor([
        k_batch[b * neurons_per_sample, 0].item()
        for b in range(batch_size)
    ], device=device)

    u_final, u_trajectory = integrate_neural_ode_Signal(
        model=model,
        u0=u0,
        data_template=data_template,
        data_id=data_id,
        time_steps=time_step,
        delta_t=delta_t,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        x_list=x_list,
        run=run,
        device=device,
        k_batch=k_per_sample,
        ode_method=ode_method,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        noise_level=noise_level
    )

    pred_x = u_final.view(-1, 1)

    loss = ((pred_x[ids_batch] - y_batch[ids_batch]) / (delta_t * time_step)).norm(2)

    return loss, pred_x
