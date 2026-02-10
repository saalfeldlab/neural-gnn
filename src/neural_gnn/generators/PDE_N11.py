
import torch
import torch_geometric as pyg

class PDE_N11(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling with configurable activation functions per neuron type

    X tensor layout:
    x[:, 0]   = index (neuron ID)
    x[:, 1:3] = positions (x, y)
    x[:, 3]   = signal u (state)
    x[:, 4]   = external_input
    x[:, 5]   = plasticity p (PDE_N6/N7)
    x[:, 6]   = neuron_type
    x[:, 7]   = calcium

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    du : float
    the update rate of the signals (dim 1)

    """

    def __init__(self, config = [],aggr_type=[], p=[], W=[], phi=[], func_p=None, device=[]):
        super(PDE_N11, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi
        self.func_p = func_p

        self.device = device
        self.n_neurons = config.simulation.n_neurons
        self.n_neuron_types = config.simulation.n_neuron_types

        if self.func_p is None:
            self.func_p = [['tanh', 1.0, 1.0] for n in range(self.n_neuron_types)]

    def forward(self, data=[], has_field=False, data_id=[], frame=None):
        x, _edge_index = data.x, data.edge_index
        neuron_type = x[:, 6].long()
        parameters = self.p[neuron_type]

        g = parameters[:, 0:1]
        c = parameters[:, 1:2]
        u = x[:, 3:4]  # signal state
        external_input = x[:, 4:5]  # external input

        msg = torch.zeros_like(u)

        for n in range(self.n_neuron_types):
            func_name = self.func_p[n][0]
            amplitude = self.func_p[n][1]
            slope = self.func_p[n][2]
            type_mask = (neuron_type == n)
            if type_mask.any():
                if func_name == 'tanh':
                    activated_u = amplitude * torch.tanh(u / slope)
                elif func_name == 'relu':
                    activated_u = amplitude * torch.relu(u / slope)
                elif func_name == 'sigmoid':
                    activated_u = amplitude * torch.sigmoid(u / slope)
                elif func_name == 'identity':
                    activated_u = amplitude * (u / slope)
                else:
                    activated_u = amplitude * self.phi(u / slope)
                msg_n = torch.matmul(self.W, activated_u)
                msg[type_mask] = msg_n[type_mask]

        du = -c*u + g * msg + external_input

        return du

    def message(self, u_j, edge_attr):

        self.activation = self.phi(u_j)
        self.u_j = u_j

        return edge_attr[:,None] * self.phi(u_j)


    def func(self, u, type, function):
        if function=='phi':
            if self.func_p is not None and type < len(self.func_p):
                func_name = self.func_p[type][0]
                amplitude = self.func_p[type][1] 
                slope = self.func_p[type][2] 
                if func_name == 'tanh':
                    return amplitude * torch.tanh(u / slope)
                elif func_name == 'relu':
                    return amplitude * torch.relu(u / slope)
                elif func_name == 'sigmoid':
                    return amplitude * torch.sigmoid(u / slope)
                elif func_name == 'identity':
                    return amplitude * (u / slope)
                else:
                    return amplitude * self.phi(u / slope)
            return self.phi(u)

        elif function=='update':
            _g, c = self.p[type, 0:1], self.p[type, 1:2]
            return -c * u

