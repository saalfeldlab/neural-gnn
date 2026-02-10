
import torch_geometric as pyg
from neural_gnn.utils import to_numpy
import torch

class PDE_N7(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling with short-term plasticity and scaled activation

    X tensor layout:
    x[:, 0]   = index (neuron ID)
    x[:, 1:3] = positions (x, y)
    x[:, 3]   = signal u (state)
    x[:, 4]   = external_input
    x[:, 5]   = plasticity p
    x[:, 6]   = neuron_type
    x[:, 7]   = calcium

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    du : float - the update rate of the signals (dim 1)
    dp : float - the update rate of the plasticity (dim 1)

    """

    def __init__(self, config=None, aggr_type=[], p=[], W=[], phi=[], short_term_plasticity_mode='', device=None):
        super(PDE_N7, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi
        self.short_term_plasticity_mode = short_term_plasticity_mode
        self.device = device
        self.n_neurons = config.simulation.n_neurons

    def forward(self, data=[], has_field=False, data_id=[], frame=None):
        x, edge_index = data.x, data.edge_index
        neuron_type = to_numpy(x[:, 6])
        parameters = self.p[neuron_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]
        c = parameters[:, 2:3]
        t = parameters[:, 3:4]
        tau = parameters[:, 4:5]
        alpha = parameters[:, 5:6]

        u = x[:, 3:4]  # signal state
        p = x[:, 5:6]  # plasticity
        external_input = x[:, 4:5]  # external input

        msg = self.propagate(edge_index, u=u, t=t)

        du = -c * u + s * torch.tanh(u) + g * p * msg + external_input
        dp = (1-p)/tau - alpha * p * torch.abs(u)

        return du, dp

    def message(self, edge_index_i, edge_index_j, u_j, t_i):

        T = self.W
        return T[edge_index_i, edge_index_j][:, None] * self.phi(u_j / t_i)


    def func(self, u, type, function):

        if function=='phi':
            return self.phi(u)

        elif function=='update':
            _g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * torch.tanh(u)
