
import torch_geometric as pyg
import torch

class PDE_N2(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-dependent

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

    def __init__(self, config=None, aggr_type=[], p=[], W=[], phi=[], device=None):
        super(PDE_N2, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi
        self.device = device
        self.n_neurons = config.simulation.n_neurons

    def forward(self, data=[], has_field=False, data_id=[], frame=None):
        x, _edge_index = data.x, data.edge_index
        neuron_type = x[:, 6].long()
        parameters = self.p[neuron_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]
        c = parameters[:, 2:3]

        u = x[:, 3:4]  # signal state
        external_input = x[:, 4:5]  # external input

        if has_field:
            field = external_input
        else:
            field = torch.ones_like(u)

        msg = torch.matmul(self.W, self.phi(u))

        du = -c * u + s * torch.tanh(u) + g * msg * field + external_input

        return du


    def message(self, u_j, edge_attr):

        self.activation = self.phi(u_j)
        self.u_j = u_j

        return edge_attr[:,None] * self.phi(u_j)


    def func(self, u, type, function):

        if function=='phi':
            return self.phi(u)

        elif function=='update':
            _g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * torch.tanh(u)
