import torch
import torch.nn as nn
from NeuralGraph.models.MLP import MLP
# Siren_Network import not needed - removed star import


class Signal_Propagation_MLP(nn.Module):  # NOT MessagePassing
    """MLP baseline that predicts dv/dt from flattened (v, I)
    
    Standard feedforward neural network baseline for comparison with GNN.
    
    Model: dv/dt = MLP_θ([v₁, ..., vₙ, I₁, ..., Iₘ])
    
    Architecture:
    - Input: Concatenated neuron states (v) and external inputs (I)
    - Output: Predicted derivatives dv/dt for all neurons
    - No graph structure, no connectivity information
    
    This is the naive baseline to demonstrate the value of incorporating
    connectivity structure (as done in GNN) for neural circuit modeling.
    """
    
    def __init__(self, aggr_type='add', config=None, device=None):
        super(Signal_Propagation_MLP, self).__init__()
        
        simulation_config = config.simulation
        model_config = config.graph_model
        self.device = device
        
        # Extract relevant params
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.calcium_type = simulation_config.calcium_type
        self.MLP_activation = model_config.MLP_activation

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers


        # MLP predicts dv/dt for all neurons
        self.mlp = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            nlayers=self.n_layers,
            hidden_size=self.hidden_dim,
            activation=self.MLP_activation,
            device=self.device,
        )

        self.a = nn.Parameter(
            torch.randn(self.n_neurons, model_config.embedding_dim).to(self.device),
            requires_grad=True,
        )
    
    def forward(self, x=[], data_id=[], k=[], return_all=False, **kwargs):
        # Extract features

        if self.calcium_type != "none":
            v = x[:, 7:8]  # calcium
        else:
            v = x[:, 3:4]  # voltage
        
        excitation = x[:self.n_input_neurons, 4:5]
        
        # Flatten: concatenate all neurons and inputs
        # Assuming data.x has shape (n_neurons, n_features)
        in_features = torch.cat([v.flatten(), excitation.flatten()])
        
        # Predict dv/dt for all neurons
        pred = self.mlp(in_features)
        
        return pred.view(-1, 1) # Match GNN output shape