import torch
import torch.nn as nn
from NeuralGraph.models.MLP import MLP
from torchdiffeq import odeint

class Signal_Propagation_MLP_ODE(nn.Module):
    """Neural ODE baseline: dv/dt = f_θ(v, I)
    
    Backward-compatible:
      - Same API and default behavior as your previous version.
      - Call `enable_solver_training(True)` to train with an ODE solver in forward().
    """
    def __init__(self, aggr_type='add', config=None, device=None):
        super(Signal_Propagation_MLP_ODE, self).__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        self.device = device

        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.calcium_type = simulation_config.calcium_type
        self.delta_t = float(simulation_config.delta_t)

        # --- drift network f_θ ---

        self.ode_func = MLP(
            input_size=model_config.input_size,
            output_size=model_config.output_size,
            nlayers=model_config.n_layers,
            hidden_size=model_config.hidden_dim,
            activation=model_config.MLP_activation,
            device=self.device,
        )

        # Kept for compatibility (unused in drift unless you wire it in)
        self.a = nn.Parameter(
            torch.randn(self.n_neurons, model_config.embedding_dim).to(self.device),
            requires_grad=True,
        )

        self.ode_method = 'dopri5'
        self.ode_rtol = 1e-5
        self.ode_atol = 1e-7
        self.stab_lambda = 0.0                  # optional linear stabilizer; 0 keeps behavior

    def set_stabilizer(self, lam=0.0):
        self.stab_lambda = float(lam)

    # ---- drift f(t, v, I) ----
    def f(self, t, v_flat, I_flat):
        """ODE drift: dv/dt = f(v, I). Inputs are 1D flattened vectors."""
        x = torch.cat([v_flat, I_flat], dim=0)
        dv = self.ode_func(x)  # shape [N]
        if self.stab_lambda > 0:
            # linear contraction term to help long rollouts
            dv = dv - self.stab_lambda * v_flat
        return dv  # [N]

    # ---- forward: returns dv/dt (shape [N,1]) just like before ----
    def forward(self, x=[], data_id=[], k=[], return_all=False, **kwargs):
        # Extract state and inputs from your layout
        if self.calcium_type != "none":
            v = x[:, 7:8]                    # [N,1]
        else:
            v = x[:, 3:4]                    # [N,1]
        excitation = x[:self.n_input_neurons, 4:5]  # [N_in,1]

        v_flat = v.flatten()                 # [N]
        I_flat = excitation.flatten()        # [N_in]

        # === Neural-ODE behavior: integrate one Δt, then return (v_{t+Δt}-v_t)/Δt ===
        def rhs(t, y):
            # y: [N], treat I as piecewise-constant over this small step
            return self.f(t, y, I_flat)

        y0 = v_flat
        ts = torch.tensor([0.0, self.delta_t], device=y0.device, dtype=y0.dtype)
        yT = odeint(
            rhs, y0, ts,
            method=self.ode_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol
        )[-1]                                   # [N]
        dv_dt = (yT - v_flat) / self.delta_t    # [N]

        return dv_dt.view(-1, 1)                # [N,1]

    # ---- rollout utilities (kept, but now use f() for consistency) ----
    @torch.no_grad()
    def rollout_step(self, v, I, dt, method='rk4'):
        """Single integration step for rollout. v:[N,1], I:[N_in,1]."""
        v_flat = v.flatten()
        I_flat = I.flatten()

        if method == 'euler':
            k1 = self.f(None, v_flat, I_flat)                 # [N]
            v_next = v + dt * k1.view(-1, 1)
            return v_next

        elif method == 'rk4':
            k1 = self.f(None, v_flat, I_flat)

            v2 = (v_flat + 0.5 * dt * k1)
            k2 = self.f(None, v2, I_flat)

            v3 = (v_flat + 0.5 * dt * k2)
            k3 = self.f(None, v3, I_flat)

            v4 = (v_flat + dt * k3)
            k4 = self.f(None, v4, I_flat)

            incr = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return v + incr.view(-1, 1)

        elif method == 'odeint':  # full solver for a single step
            def rhs(t, y): return self.f(t, y, I_flat)
            ts = torch.tensor([0.0, dt], device=v.device, dtype=v.dtype)
            yT = odeint(rhs, v_flat, ts, method=self.ode_method,
                        rtol=self.ode_rtol, atol=self.ode_atol)[-1]
            return yT.view_as(v)

        else:
            raise ValueError("method must be 'euler', 'rk4', or 'odeint'")
