
import torch_geometric as pyg
import torch

class PDE_N5(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-neuron-dependent

    X tensor layout:
    x[:, 0]   = index (neuron ID)
    x[:, 1:3] = positions (x, y)
    x[:, 3]   = signal u (state)
    x[:, 4]   = external_input
    x[:, 5]   = plasticity p (PDE_N6/N7)
    x[:, 6]   = neuron_type
    x[:, 7]   = calcium

    params: [a, b, g, s, w, h]
    a: decay, b: offset, g: gain, s: self-recurrence, w: width, h: threshold
    MLP1((u-h)/w)

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    du : float
    the update rate of the signals (dim 1)

    """

    def __init__(self, config=None, aggr_type=[], p=[], W=[], phi=[], device=None):
        super(PDE_N5, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi
        self.device = device
        self.n_neurons = config.simulation.n_neurons

    def forward(self, data=[], has_field=False, data_id=[], frame=None):
        x, edge_index = data.x, data.edge_index

        neuron_type = x[:, 6].long()
        parameters = self.p[neuron_type]
        # params: [a, b, g, s, w, h]
        a = parameters[:, 0:1]  # decay
        b = parameters[:, 1:2]  # offset
        g = parameters[:, 2:3]  # gain
        s = parameters[:, 3:4]  # self-recurrence
        w = parameters[:, 4:5]  # width
        l = torch.log(parameters[:, 4:5])  # log(width)

        if parameters.shape[1] < 6:
            h = torch.zeros_like(w)  # threshold defaults to 0
        else:
            h = parameters[:, 5:6]  # threshold

        u = x[:, 3:4]  # signal state

        if has_field:
            field = x[:, 4:5]
            msg = self.propagate(edge_index, u=u, w=w, l=l, h=h, field=field)
            du = -a * u + b + s * torch.tanh(u) + g * msg
        else:
            field = torch.ones_like(u)
            external_input = x[:, 4:5]  # external input
            msg = self.propagate(edge_index, u=u, w=w, l=l, h=h, field=field)
            du = -a * u + b + s * torch.tanh(u) + g * msg + external_input

        return du


    def message(self, edge_index_i, edge_index_j, u_j, w_i, l_j, h_j, field_i):

        T = self.W
        return T[edge_index_i, edge_index_j][:, None] * (self.phi((u_j - h_j) / w_i) - u_j * l_j / 50) * field_i


    def func(self, u, type, function):

        if function == 'phi':
            # params: [a, b, g, s, w, h]
            w = self.p[type, 4:5]  # width
            l = torch.log(self.p[type, 4:5])  # log(width)
            if self.p.shape[1] < 6:
                h = torch.zeros_like(w)
            else:
                h = self.p[type, 5:6]  # threshold

            return self.phi((u - h) / w) - u * l / 50

        elif function == 'update':
            # params: [a, b, g, s, w, h]
            a = self.p[type, 0:1]  # decay
            b = self.p[type, 1:2]  # offset
            s = self.p[type, 3:4]  # self-recurrence
            return -a * u + b + s * torch.tanh(u)
