import torch
import torch.nn as nn

class Signal_Propagation_RNN(nn.Module):
    """RNN baseline for neural dynamics prediction
    
    Standard Recurrent Neural Network for time series modeling.
    Maintains hidden state across timesteps during sequential processing.
    
    Model: h_t = GRU(h_{t-1}, [v_t, I_t])
           dv/dt = MLP(h_t)
    """
    
    def __init__(self, aggr_type='add', config=None, device=None):
        super(Signal_Propagation_RNN, self).__init__()
        
        simulation_config = config.simulation
        model_config = config.graph_model
        
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.calcium_type = simulation_config.calcium_type
        self.input_size = model_config.input_size
        self.device = device
        
        hidden_dim = model_config.hidden_dim  # e.g., 512
        
        # GRU layers
        self.gru = nn.GRU(
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
    
    def forward(self, x=[], h=None, data_id=[], mask=[], k=[], return_all=False):
        """
        Args:
            x: State tensor (n_neurons, features)
            h: Hidden state (num_layers, 1, hidden_dim) or None
            return_all: If True, return (prediction, hidden_state)
        
        Returns:
            dv_dt: Predicted derivative (n_neurons, 1)
            h: Updated hidden state (if return_all=True)
        """
        
        if self.calcium_type != "none":
            v = x[:, 7:8]
        else:
            v = x[:, 3:4]
        
        excitation = x[:self.n_input_neurons, 4:5]
        
        # Flatten and concatenate: (n_neurons + n_input_neurons,)
        inp = torch.cat([v.flatten(), excitation.flatten()])
        
        # Reshape for GRU: (batch=1, seq=1, features)
        inp = inp.unsqueeze(0).unsqueeze(0)
        
        # GRU forward
        out, h_next = self.gru(inp, h)
        
        # Predict dv/dt
        dv_dt = self.fc(out.squeeze(0)).view(-1, 1)
        
        if return_all:
            return dv_dt, h_next
        return dv_dt