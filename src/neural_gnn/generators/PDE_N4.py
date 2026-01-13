
import torch_geometric as pyg
import torch


class PDE_N4(pyg.nn.MessagePassing):
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
        super(PDE_N4, self).__init__(aggr=aggr_type)

        # validate params has 6 columns: [a, b, g, s, w, h]
        if p.shape[1] != 6:
            raise ValueError(
                f"params must have 6 columns [a, b, g, s, w, h], got {p.shape[1]}. "
                f"Check your config 'params' field."
            )

        self.p = p
        self.W = W
        self.phi = phi
        self.device = device
        self.n_neurons = config.simulation.n_neurons
        self.external_input_mode = getattr(config.simulation, 'external_input_mode', 'none')

    def forward(self, data=[], has_field=False, data_id=[], frame=None):
        x, edge_index = data.x, data.edge_index
        neuron_type = x[:, 6].long()

        # validate neuron_type indices against params size
        max_type = neuron_type.max().item()
        n_param_types = self.p.shape[0]
        if max_type >= n_param_types:
            raise ValueError(
                f"neuron_type index {max_type} out of bounds for params with {n_param_types} types. "
                f"Config has n_neuron_types={max_type + 1} but params only defines {n_param_types} sets. "
                f"Add {max_type + 1 - n_param_types} more parameter sets to 'params' in your config."
            )

        # extract neuron-type-dependent parameters
        # params order: [a, b, g, s, w, h]
        parameters = self.p[neuron_type]
        a = parameters[:, 0:1]  # decay: rate of signal decay (-a*u term)
        b = parameters[:, 1:2]  # offset: constant input/drive term
        g = parameters[:, 2:3]  # gain: coupling strength for incoming messages
        s = parameters[:, 3:4]  # self-recurrence: strength of self-feedback
        w = parameters[:, 4:5]  # width: temperature/scaling for activation function MLP1((u-h)/w)
        h = parameters[:, 5:6]  # threshold: baseline for activation function MLP1((u-h)/w)

        u = x[:, 3:4]  # signal state
        external_input = x[:, 4:5]

        msg = self.propagate(edge_index, u=u, w=w, h=h)

        if self.external_input_mode == "multiplicative":
            # multiplicative mode: external_input modulates the message
            du = -a * u + b + s * torch.tanh(u) + g * msg * external_input
        elif self.external_input_mode == "additive":
            # additive mode: external_input is added directly
            du = -a * u + b + s * torch.tanh(u) + g * msg + external_input
        else:  # none
            du = -a * u + b + s * torch.tanh(u) + g * msg

        return du

    def message(self, edge_index_i, edge_index_j, u_j, w_j, h_j):
        # Message from neuron j to neuron i: W_ij * φ((u_j - h_j) / w_j)
        # W: connectivity matrix, φ: activation function, h: threshold, w: width
        T = self.W

        return T[edge_index_i, edge_index_j][:, None] * self.phi((u_j - h_j) / w_j)

    def func(self, u, type, function):

        if function=='phi':
            # φ((u - h) / w): activation function with threshold h and width w
            # params order: [a, b, g, s, w, h] = indices [0, 1, 2, 3, 4, 5]
            w = self.p[type, 4:5]  # width
            h = self.p[type, 5:6]  # threshold

            return self.phi((u - h) / w)

        elif function=='update':
            # Self-dynamics: -a*u + s*tanh(u)
            # params order: [a, b, g, s, w, h]
            a, s = self.p[type, 0:1], self.p[type, 3:4]
            return -a * u + s * torch.tanh(u)
