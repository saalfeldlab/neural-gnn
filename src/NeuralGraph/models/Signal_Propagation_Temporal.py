import torch
import torch.nn as nn
import torch_geometric as pyg
from NeuralGraph.models.MLP import MLP
import numpy as np
from NeuralGraph.models.Siren_Network import Siren

class Signal_Propagation_Temporal(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the first derivative of a scalar field on a mesh.
    The node embedding is defined by a table self.a
    Note the Laplacian coeeficients are in data.edge_attr

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the first derivative of a scalar field on a mesh (dimension 3).
    """

    def __init__(
        self, aggr_type='add', config=None, device=None ):
        super(Signal_Propagation_Temporal, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.model = model_config.signal_model_name
        self.embedding_dim = model_config.embedding_dim
        self.n_neurons = simulation_config.n_neurons
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.field_type = model_config.field_type
        self.embedding_trial = config.training.embedding_trial
        self.multi_connectivity = config.training.multi_connectivity
        self.calcium_type = simulation_config.calcium_type

        self.training_time_window = config.training.time_window

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.lin_edge_positive = model_config.lin_edge_positive

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        self.lin_edge = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            nlayers=self.n_layers,
            hidden_size=self.hidden_dim,
            device=self.device,
        )

        self.lin_phi = MLP(
            input_size=self.input_size_update,
            output_size=self.output_size,
            nlayers=self.n_layers_update,
            hidden_size=self.hidden_dim_update,
            device=self.device,
        )

        # embedding
        self.a = nn.Parameter(
            torch.tensor(
                np.ones((int(self.n_neurons), self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))

        self.W = nn.Parameter(
            torch.zeros(
                self.n_edges + self.n_extra_null_edges,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )[:, None]
        )

        if 'visual' in model_config.field_type:
            self.visual_NNR = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                  hidden_features=model_config.hidden_dim_nnr,
                  hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                  hidden_omega_0=model_config.omega,
                  outermost_linear=model_config.outermost_linear_nnr)
            self.visual_NNR.to(self.device)


    def forward(self, data=[], data_id=[], mask=[], return_all=False):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        self.data_id = data_id.squeeze().long().clone().detach()
        self.mask = mask.squeeze().long().clone().detach()

        v = data.x[:, -5:]

        # print(data.x.shape, v.shape)

        excitation = data.x[:, 4:5]

        particle_id = x[:, 0].long()
        embedding = self.a[particle_id].squeeze()

        msg = self.propagate(
            edge_index, v=v, embedding=embedding, data_id=self.data_id[:, None]
        )

        in_features = torch.cat([v, embedding, msg, excitation], dim=1)
        pred = self.lin_phi(in_features)
        
        if return_all:
            return pred, in_features, msg
        else:
            return pred

    def message(self, edge_index_i, edge_index_j, v_i, v_j, embedding_i, embedding_j, data_id_i):

        # Flatten to [N*D, 1]
        reshaped = v_j.reshape(-1, 1)
        # print (v_j.shape, reshaped.shape, embedding_j.shape)
        embedding_j_expanded = embedding_j.repeat_interleave(self.training_time_window, dim=0)
        # print (embedding_j_expanded.shape)
        in_features = torch.cat([reshaped, embedding_j_expanded], dim=1)
        # print (in_features.shape)

        lin_edge = self.lin_edge(in_features)
        if self.lin_edge_positive:
            lin_edge = lin_edge**2

        # Restore to [N, D]
        lin_edge = lin_edge.reshape(v_j.shape[0], v_j.shape[1])

        return self.W[self.mask % (self.n_edges+ self.n_extra_null_edges)] * lin_edge

    def update(self, aggr_out):
        return aggr_out
