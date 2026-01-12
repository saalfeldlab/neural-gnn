import torch
import torch.nn as nn

class Signal_Propagation_LSTM(nn.Module):
    """LSTM baseline for neural dynamics prediction
    
    Long Short-Term Memory network for time series modeling.
    Better at learning long-term dependencies than GRU.
    
    Model: (h_t, c_t) = LSTM((h_{t-1}, c_{t-1}), [v_t, I_t])
           dv/dt = MLP(h_t)
    """
    
    def __init__(self, aggr_type='add', config=None, device=None):
        super(Signal_Propagation_LSTM, self).__init__()
        
        simulation_config = config.simulation
        model_config = config.graph_model
        
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.calcium_type = simulation_config.calcium_type
        self.input_size = model_config.input_size
        self.device = device
        
        hidden_dim = model_config.hidden_dim  # e.g., 512
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        ).to(device)
        
        # Output MLP
        self.fc = nn.Linear(hidden_dim, self.n_neurons).to(device)
        
        # Dummy embedding (for compatibility with existing code)
        self.a = nn.Parameter(
            torch.randn(self.n_neurons, model_config.embedding_dim).to(device),
            requires_grad=True,
        )
    
    def forward(self, x=[], h=None, c=None, data_id=[], mask=[], k=[], return_all=False):
        """
        Args:
            x: State tensor (n_neurons, features)
            h: Hidden state (num_layers, 1, hidden_dim) or None
            c: Cell state (num_layers, 1, hidden_dim) or None
            return_all: If True, return (prediction, hidden_state, cell_state)
        
        Returns:
            dv_dt: Predicted derivative (n_neurons, 1)
            h, c: Updated states (if return_all=True)
        """
        
        if self.calcium_type != "none":
            v = x[:, 7:8]
        else:
            v = x[:, 3:4]
        
        excitation = x[:self.n_input_neurons, 4:5]
        
        # Flatten and concatenate: (n_neurons + n_input_neurons,)
        inp = torch.cat([v.flatten(), excitation.flatten()])
        
        # Reshape for LSTM: (batch=1, seq=1, features)
        inp = inp.unsqueeze(0).unsqueeze(0)
        
        # Combine hidden and cell states if provided
        hc = None
        if h is not None and c is not None:
            hc = (h, c)
        
        # LSTM forward
        out, (h_next, c_next) = self.lstm(inp, hc)
        
        # Predict dv/dt
        dv_dt = self.fc(out.squeeze(0)).view(-1, 1)
        
        if return_all:
            return dv_dt, h_next, c_next
        return dv_dt