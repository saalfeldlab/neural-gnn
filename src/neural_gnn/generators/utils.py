
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import subprocess
import torch
import xarray as xr
from neural_gnn.generators import PDE_N2, PDE_N3, PDE_N4, PDE_N5, PDE_N6, PDE_N7, PDE_N11
from neural_gnn.utils import choose_boundary_values, get_equidistant_points, to_numpy, large_tensor_nonzero
from scipy import stats
from scipy.spatial import Delaunay
from time import sleep
from tifffile import imread
from torch_geometric.utils import get_mesh_laplacian, dense_to_sparse
from tqdm import trange
import seaborn as sns

# Optional imports
try:
    from fa2_modified import ForceAtlas2
except ImportError:
    ForceAtlas2 = None


def choose_model(config=[], W=[], device=[]):
    model_signal_name = config.graph_model.signal_model_name
    aggr_type = config.graph_model.aggr_type
    short_term_plasticity_mode = config.simulation.short_term_plasticity_mode

    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)

    params = config.simulation.params
    p = torch.tensor(params, dtype=torch.float32, device=device).squeeze()


    match config.simulation.phi:
        case 'tanh':
            phi=torch.tanh
        case 'relu':
            phi=torch.relu
        case 'sigmoid':
            phi=torch.sigmoid
        case _:
            phi=torch.sigmoid

    match model_signal_name:
        case 'PDE_N2':
            model = PDE_N2(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, device=device)
        case 'PDE_N3':
            model = PDE_N3(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, device=device)
        case 'PDE_N4':
            model = PDE_N4(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, device=device)
        case 'PDE_N5':
            model = PDE_N5(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, device=device)
        case 'PDE_N6':
            model = PDE_N6(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode=short_term_plasticity_mode, device=device)
        case 'PDE_N7':
            model = PDE_N7(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode=short_term_plasticity_mode, device=device)
        case 'PDE_N11':
            func_p = config.simulation.func_params
            model = PDE_N11(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, func_p=func_p, device=device)



    return model, bc_pos, bc_dpos


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)


def init_neurons(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    n_neurons = simulation_config.n_neurons * ratio
    n_neuron_types = simulation_config.n_neuron_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init


    xc, yc = get_equidistant_points(n_points=n_neurons)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0), device=device)
    pos = pos[perm]

    dpos = dpos_init * torch.randn((n_neurons, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))

    type = torch.zeros(int(n_neurons / n_neuron_types), device=device)

    for n in range(1, n_neuron_types):
        type = torch.cat((type, n * torch.ones(int(n_neurons / n_neuron_types), device=device)), 0)
    if type.shape[0] < n_neurons:
        type = torch.cat((type, n * torch.ones(n_neurons - type.shape[0], device=device)), 0)

    if (config.graph_model.signal_model_name == 'PDE_N6') | (config.graph_model.signal_model_name == 'PDE_N7'):
        features = torch.cat((torch.rand((n_neurons, 1), device=device), 0.1 * torch.randn((n_neurons, 1), device=device),
                              torch.ones((n_neurons, 1), device=device), torch.zeros((n_neurons, 1), device=device)), 1)
    elif 'excitation_single' in config.graph_model.field_type:
        features = torch.zeros((n_neurons, 2), device=device)
    else:
        features = torch.cat((torch.randn((n_neurons, 1), device=device) * 5 , 0.1 * torch.randn((n_neurons, 1), device=device)), 1)

    type = type[:, None]
    particle_id = torch.arange(n_neurons, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_neurons,1), device=device)

    return pos, dpos, type, features, age, particle_id


def random_rotation_matrix(device='cpu'):
    # Random Euler angles
    roll = torch.rand(1, device=device) * 2 * torch.pi
    pitch = torch.rand(1, device=device) * 2 * torch.pi
    yaw = torch.rand(1, device=device) * 2 * torch.pi

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices around each axis
    R_x = torch.tensor([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ], device=device).squeeze()

    R_y = torch.tensor([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], device=device).squeeze()

    R_z = torch.tensor([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ], device=device).squeeze()

    # Combined rotation matrix: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R


def get_index(n_neurons, n_neuron_types):
    index_particles = []
    for n in range(n_neuron_types):
        index_particles.append(
            np.arange((n_neurons // n_neuron_types) * n, (n_neurons // n_neuron_types) * (n + 1)))
    return index_particles


def get_time_series(x_list, cell_id, feature):

    match feature:
        case 'velocity_x':
            feature = 3
        case 'velocity_y':
            feature = 4
        case 'type' | 'state':
            feature = 5
        case 'age':
            feature = 8
        case 'mass':
            feature = 10

        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))

    return to_numpy(torch.stack(time_series))


def init_mesh(config, device):

    simulation_config = config.simulation
    model_config = config.graph_model

    n_input_neurons = simulation_config.n_input_neurons
    n_neurons = simulation_config.n_neurons
    node_value_map = simulation_config.node_value_map
    field_grid = model_config.field_grid
    max_radius = simulation_config.max_radius

    n_input_neurons_per_axis = int(np.sqrt(n_input_neurons))
    xs = torch.linspace(1 / (2 * n_input_neurons_per_axis), 1 - 1 / (2 * n_input_neurons_per_axis), steps=n_input_neurons_per_axis)
    ys = torch.linspace(1 / (2 * n_input_neurons_per_axis), 1 - 1 / (2 * n_input_neurons_per_axis), steps=n_input_neurons_per_axis)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_input_neurons_per_axis ** 2, 1))
    y_mesh = torch.reshape(y_mesh, (n_input_neurons_per_axis ** 2, 1))
    mesh_size = 1 / n_input_neurons_per_axis
    pos_mesh = torch.zeros((n_input_neurons, 2), device=device)
    pos_mesh[0:n_input_neurons, 0:1] = x_mesh[0:n_input_neurons]
    pos_mesh[0:n_input_neurons, 1:2] = y_mesh[0:n_input_neurons]

    i0 = imread(f'graphs_data/{node_value_map}')
    if len(i0.shape) == 2:
        # i0 = i0[0,:, :]
        i0 = np.flipud(i0)
        values = i0[(to_numpy(pos_mesh[:, 1]) * 255).astype(int), (to_numpy(pos_mesh[:, 0]) * 255).astype(int)]

    mask_mesh = (x_mesh > torch.min(x_mesh) + 0.02) & (x_mesh < torch.max(x_mesh) - 0.02) & (y_mesh > torch.min(y_mesh) + 0.02) & (y_mesh < torch.max(y_mesh) - 0.02)

    if 'grid' in field_grid:
        pos_mesh = pos_mesh
    else:
        if 'pattern_Null.tif' in simulation_config.node_value_map:
            pos_mesh = pos_mesh + torch.randn(n_input_neurons, 2, device=device) * mesh_size / 24
        else:
            pos_mesh = pos_mesh + torch.randn(n_input_neurons, 2, device=device) * mesh_size / 8

    match config.graph_model.mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            node_value = torch.zeros((n_input_neurons, 2), device=device)
            node_value[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            node_value[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case 'RD_FitzHugh_Nagumo_Mesh':
            node_value = torch.zeros((n_input_neurons, 2), device=device) + torch.rand((n_input_neurons, 2), device=device) * 0.1
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' :
            node_value = torch.rand((n_input_neurons, 3), device=device)
            s = torch.sum(node_value, dim=1)
            for k in range(3):
                node_value[:, k] = node_value[:, k] / s
        case 'DiffMesh' | 'WaveMesh' | 'Particle_Mesh_A' | 'Particle_Mesh_B' | 'WaveSmoothParticle':
            node_value = torch.zeros((n_input_neurons, 2), device=device)
            node_value[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case 'PDE_O_Mesh':
            node_value = torch.zeros((n_neurons, 5), device=device)
            node_value[0:n_neurons, 0:1] = x_mesh[0:n_neurons]
            node_value[0:n_neurons, 1:2] = y_mesh[0:n_neurons]
            node_value[0:n_neurons, 2:3] = torch.randn(n_neurons, 1, device=device) * 2 * np.pi  # theta
            node_value[0:n_neurons, 3:4] = torch.ones(n_neurons, 1, device=device) * np.pi / 200  # d_theta
            node_value[0:n_neurons, 4:5] = node_value[0:n_neurons, 3:4]  # d_theta0
            pos_mesh[:, 0] = node_value[:, 0] + (3 / 8) * mesh_size * torch.cos(node_value[:, 2])
            pos_mesh[:, 1] = node_value[:, 1] + (3 / 8) * mesh_size * torch.sin(node_value[:, 2])
        case '' :
            node_value = torch.zeros((n_input_neurons, 2), device=device)



    type_mesh = torch.zeros((n_input_neurons, 1), device=device)

    node_id_mesh = torch.arange(n_input_neurons, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((n_input_neurons, 2), device=device)

    x_mesh = torch.concatenate((node_id_mesh.clone().detach(), pos_mesh.clone().detach(), dpos_mesh.clone().detach(),
                                type_mesh.clone().detach(), node_value.clone().detach()), 1)

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    sleep(0.5)
    for k in trange(face.shape[0], ncols=100):
        # compute edge distances
        x1 = pos[face[k, 0], :]
        x2 = pos[face[k, 1], :]
        x3 = pos[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
    edge_index_mesh, edge_weight_mesh = get_mesh_laplacian(pos=pos_3d, face=face, normalization="None")
    edge_weight_mesh = edge_weight_mesh.to(dtype=torch.float32)
    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}

    if (config.graph_model.particle_model_name == 'PDE_ParticleField_A')  | (config.graph_model.particle_model_name == 'PDE_ParticleField_B'):
        type_mesh = 0 * type_mesh

    a_mesh = torch.zeros_like(type_mesh)
    type_mesh = type_mesh.to(dtype=torch.float32)

    if 'Smooth' in config.graph_model.mesh_model_name:
        distance = torch.sum((pos_mesh[:, None, :] - pos_mesh[None, :, :]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= 0)).float() * 1
        mesh_data['edge_index'] = adj_t.nonzero().t().contiguous()


    return pos_mesh, dpos_mesh, type_mesh, node_value, a_mesh, node_id_mesh, mesh_data


def init_connectivity(connectivity_file, connectivity_type, connectivity_filling_factor, T1, n_neurons, n_neuron_types, dataset_name, device, connectivity_rank=1, Dale_law=False, Dale_law_factor=0.5):

    if 'pt' in connectivity_file:
        connectivity = torch.load(connectivity_file, map_location=device)
    elif 'mat' in connectivity_file:
        mat = scipy.io.loadmat(connectivity_file)
        connectivity = torch.tensor(mat['A'], device=device)
    elif 'zarr' in connectivity_file:
        print('loading zarr ...')
        dataset = xr.open_zarr(connectivity_file)
        trained_weights = dataset["trained"]  # alpha * sign * N
        print(f'weights {trained_weights.shape}')
        dataset["untrained"]  # sign * N
        values = trained_weights[0:n_neurons,0:n_neurons]
        values = np.array(values)
        values = values / np.max(values)
        connectivity = torch.tensor(values, dtype=torch.float32, device=device)
        values=[]
    elif 'tif' in connectivity_file:
        # TODO: constructRandomMatrices function not implemented
        raise NotImplementedError("constructRandomMatrices function not implemented for tif files")
        # connectivity = constructRandomMatrices(n_neurons=n_neurons, density=1.0, connectivity_mask=f"./graphs_data/{connectivity_file}" ,device=device)
        # n_neurons = connectivity.shape[0]
        # TODO: config parameter not passed to this function
        # config.simulation.n_neurons = n_neurons
    elif connectivity_type != 'none':

        if 'chaotic' in connectivity_type:
            # Chaotic network 
            connectivity = np.random.randn(n_neurons,n_neurons) * np.sqrt(1/n_neurons)
        elif 'ring attractor' in connectivity_type:
            # Ring attractor network 
            th = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)   # Preferred firing location (angle)
            J1 = 1.0
            J0 = 0.5
            connectivity = (J1 * np.cos(th[:, None] - th[None, :]) + J0) / n_neurons   # Synaptic weight matrix
        elif 'low_rank' in connectivity_type:
            # Low rank network: W = U @ V where U is (N x rank) and V is (rank x N)
            U = np.random.randn(n_neurons, connectivity_rank)
            V = np.random.randn(connectivity_rank, n_neurons)
            connectivity = U @ V / np.sqrt(connectivity_rank * n_neurons)
        elif 'successor' in connectivity_type:
            # Successor Representation
            T = np.eye(n_neurons, k=1)
            gamma = 0.98
            connectivity = np.linalg.inv(np.eye(n_neurons) - gamma*T)
        elif 'null' in connectivity_type:
            connectivity = np.zeros((n_neurons, n_neurons))
        elif 'Gaussian' in connectivity_type:
            connectivity = torch.randn((n_neurons, n_neurons), dtype=torch.float32, device=device)
            connectivity = connectivity / np.sqrt(n_neurons)
            print(f"Gaussian   1/sqrt(N)  {1/np.sqrt(n_neurons)}    std {torch.std(connectivity.flatten())}")
        elif 'Lorentz' in connectivity_type:
            s = np.random.standard_cauchy(n_neurons**2)
            s[(s < -25) | (s > 25)] = 0
            if n_neurons < 2000:
                s = s / n_neurons**0.7
            elif n_neurons <4000:
                s = s / n_neurons**0.675
            elif n_neurons < 8000:
                s = s / n_neurons**0.67
            elif n_neurons == 8000:
                s = s / n_neurons**0.66
            elif n_neurons > 8000:
                s = s / n_neurons**0.5
            print(f"Lorentz   1/sqrt(N)  {1/np.sqrt(n_neurons):0.3f}    std {np.std(s):0.3f}")
            connectivity = torch.tensor(s, dtype=torch.float32, device=device)
            connectivity = torch.reshape(connectivity, (n_neurons, n_neurons))
        elif 'uniform' in connectivity_type:
            connectivity = torch.rand((n_neurons, n_neurons), dtype=torch.float32, device=device)
            connectivity = connectivity - 0.5

        connectivity = torch.tensor(connectivity, dtype=torch.float32, device=device)
        connectivity.fill_diagonal_(0)

    # Apply Dale's law: each neuron (column) is either excitatory or inhibitory
    if Dale_law:
        n_excitatory = int(n_neurons * Dale_law_factor)
        n_inhibitory = n_neurons - n_excitatory

        # Take absolute values
        connectivity = torch.abs(connectivity)

        # First n_excitatory columns are positive (excitatory), rest are negative (inhibitory)
        # Columns represent presynaptic neurons in W[post, pre] convention
        connectivity[:, n_excitatory:] = -connectivity[:, n_excitatory:]

        print(f"Dale's law applied: {n_excitatory} excitatory columns, {n_inhibitory} inhibitory columns")

    if connectivity_filling_factor != 1:
        mask = torch.rand(connectivity.shape) > connectivity_filling_factor
        connectivity[mask] = 0
        mask = (connectivity != 0).float()

        # Calculate effective filling factor
        total_possible = connectivity.shape[0] * connectivity.shape[1]
        actual_connections = mask.sum().item()
        effective_filling_factor = actual_connections / total_possible

        print(f"target filling factor: {connectivity_filling_factor}")
        print(f"effective filling factor: {effective_filling_factor:.6f}")
        print(f"actual connections: {int(actual_connections)}/{total_possible}")

        if n_neurons > 10000:
            edge_index = large_tensor_nonzero(mask)
            print(f'edge_index {edge_index.shape}')
        else:
            edge_index = mask.nonzero().t().contiguous()

    else:
        adj_matrix = torch.ones((n_neurons)) - torch.eye(n_neurons)
        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        mask = (adj_matrix != 0).float()

    if 'structured' in connectivity_type:
        parts = connectivity_type.split('_')
        float_value1 = float(parts[-2])  # repartition pos/neg
        float_value2 = float(parts[-1])  # filling factor

        matrix_sign = torch.tensor(stats.bernoulli(float_value1).rvs(n_neuron_types ** 2) * 2 - 1,
                                   dtype=torch.float32, device=device)
        matrix_sign = matrix_sign.reshape(n_neuron_types, n_neuron_types)

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_0.png', dpi=300)
        plt.close()

        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        connectivity = torch.abs(connectivity)
        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        sign_matrix = matrix_sign[xy_grid[..., 0], xy_grid[..., 1]]
        connectivity *= sign_matrix

        plt.imshow(to_numpy(sign_matrix))
        plt.savefig(f"graphs_data/{dataset_name}/large_connectivity_sign.png", dpi=130)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_1.png', dpi=300)
        plt.close()

        flat_sign_matrix = sign_matrix.flatten()
        num_elements = len(flat_sign_matrix)
        num_ones = int(num_elements * float_value2)
        indices = np.random.choice(num_elements, num_ones, replace=False)
        flat_sign_matrix[:] = 0
        flat_sign_matrix[indices] = 1
        sign_matrix = flat_sign_matrix.reshape(sign_matrix.shape)

        connectivity *= sign_matrix

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_2.png', dpi=300)
        plt.close()

        total_possible = connectivity.shape[0] * connectivity.shape[1]
        actual_connections = (connectivity != 0).sum().item()
        effective_filling_factor = actual_connections / total_possible

        print(f"target filling factor: {float_value2}")
        print(f"effective filling factor: {effective_filling_factor:.6f}")
        print(f"actual connections: {actual_connections}/{total_possible}")

    edge_index = edge_index.to(device=device)

    return edge_index, connectivity, mask


def generate_compressed_video_mp4(output_dir, run=0, framerate=10, output_name=None, crf=23, log_dir=None):
    """
    Generate a compressed video using ffmpeg's libx264 codec in MP4 format.
    Automatically handles odd dimensions by scaling to even dimensions.

    Parameters:
        output_dir (str): Path to directory containing Fig/Fig_*.png.
        run (int): Run index to use in filename pattern.
        framerate (int): Desired video framerate.
        output_name (str): Name of output .mp4 file.
        crf (int): Constant Rate Factor for quality (0-51, lower = better quality, 23 is default).
        log_dir (str): If provided, save mp4 to log_dir instead of output_dir.
    """

    fig_dir = os.path.join(output_dir, "Fig")
    input_pattern = os.path.join(fig_dir, f"Fig_{run}_%06d.png")

    # Save to log_dir if provided, otherwise to output_dir
    save_dir = log_dir if log_dir is not None else output_dir
    output_path = os.path.join(save_dir, f"{output_name}.mp4")

    # Video filter to ensure even dimensions (required for yuv420p)
    # This scales the video so both width and height are divisible by 2

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",  # Suppress verbose output
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-vf", "scale='trunc(iw/2)*2:trunc(ih/2)*2'",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"compressed video (libx264) saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to generate videos.")


def plot_synaptic_frame_visual(X1, A1, H1, dataset_name, run, num):
    """Plot frame for visual field type."""
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.subplot(211)
    plt.axis("off")
    plt.title("$\Omega_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[0:1024, 1]) * 0.95,
        to_numpy(X1[0:1024, 0]) * 0.95,
        s=15,
        c=to_numpy(A1[0:1024, 0]),
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    plt.scatter(
        to_numpy(X1[1024:, 1]) * 0.95 + 0.2,
        to_numpy(X1[1024:, 0]) * 0.95,
        s=15,
        c=to_numpy(A1[1024:, 0]),
        cmap="viridis",
        vmin=-4,
        vmax=4,
    )
    plt.xticks([])
    plt.yticks([])
    plt.subplot(212)
    plt.axis("off")
    plt.title("$x_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[0:1024, 1]),
        to_numpy(X1[0:1024, 0]),
        s=15,
        c=to_numpy(H1[0:1024, 0]),
        cmap="viridis",
        vmin=-10,
        vmax=10,
    )
    plt.scatter(
        to_numpy(X1[1024:, 1]) + 0.2,
        to_numpy(X1[1024:, 0]),
        s=15,
        c=to_numpy(H1[1024:, 0]),
        cmap="viridis",
        vmin=-10,
        vmax=10,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80)
    plt.close()


def plot_synaptic_frame_modulation(X1, A1, H1, dataset_name, run, num):
    """Plot frame for modulation field type."""
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.scatter(
        to_numpy(X1[:, 1]),
        to_numpy(X1[:, 0]),
        s=100,
        c=to_numpy(A1[:, 0]),
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    plt.subplot(222)
    plt.scatter(
        to_numpy(X1[:, 1]),
        to_numpy(X1[:, 0]),
        s=100,
        c=to_numpy(H1[:, 0]),
        cmap="viridis",
        vmin=-5,
        vmax=5,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80)
    plt.close()


def plot_synaptic_frame_plasticity(X1, x, dataset_name, run, num):
    """Plot frame for PDE_N6/PDE_N7 with short term plasticity."""
    plt.figure(figsize=(12, 5.6))
    plt.axis("off")
    plt.subplot(121)
    plt.title("activity $x_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[:, 0]),
        to_numpy(X1[:, 1]),
        s=200,
        c=to_numpy(x[:, 3]),
        cmap="viridis",
        vmin=-5,
        vmax=5,
        edgecolors="k",
        alpha=1,
    )
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.title("short term plasticity $y_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[:, 0]),
        to_numpy(X1[:, 1]),
        s=200,
        c=to_numpy(x[:, 5]),
        cmap="grey",
        vmin=0,
        vmax=1,
        edgecolors="k",
        alpha=1,
    )
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=170)
    plt.close()


def plot_synaptic_frame_default(X1, x, dataset_name, run, num):
    """Plot default frame for synaptic simulation."""
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.scatter(
        to_numpy(X1[:, 0]),
        to_numpy(X1[:, 1]),
        s=100,
        c=to_numpy(x[:, 3]),
        cmap="viridis",
        vmin=-40,
        vmax=40,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=170)
    plt.close()

    # Read back and create zoomed subplot
    im_ = imread(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(im_)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 1)
    plt.imshow(im_[800:1000, 800:1000, :])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80)
    plt.close()


def plot_synaptic_activity_traces(x_list, n_neurons, n_frames, dataset_name, model=None):
    """Plot activity traces for synaptic simulation."""
    print('plot activity ...')
    activity = x_list[:, :, 3:4]
    activity = activity.squeeze()
    activity = activity.T

    # Sample 100 traces if n_neurons > 100
    if n_neurons > 100:
        sampled_indices = np.random.choice(n_neurons, 100, replace=False)
        sampled_indices = np.sort(sampled_indices)
        activity_plot = activity[sampled_indices]
        n_plot = 100
    else:
        activity_plot = activity
        sampled_indices = np.arange(n_neurons)
        n_plot = n_neurons

    activity_plot = activity_plot - 10 * np.arange(n_plot)[:, None] + 200
    plt.figure(figsize=(10, 10))

    # Plot all traces
    plt.plot(activity_plot.T, linewidth=2, alpha=0.7)

    ax = plt.gca()
    ax.set_xlabel('time', fontsize=24)
    ax.set_ylabel('neuron index', fontsize=24)
    ax.tick_params(labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim([0, min(n_frames, 10000)])
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/activity.png", dpi=100)
    plt.close()


def plot_external_input_field(x_list, n_neurons, dataset_name, frame_indices=None, n_input_neurons=None):
    """Plot external input field Omega(t) at specific time frames (for Figure 3a).

    Args:
        x_list: Array of shape (n_frames, n_neurons, n_features)
        n_neurons: Total number of neurons
        dataset_name: Name of dataset for saving
        frame_indices: List of frame indices to plot (default: [0, n_frames//4, n_frames//2, 3*n_frames//4])
        n_input_neurons: Number of neurons with external input (default: n_neurons//2)
    """
    print('plot external input field ...')

    n_frames = x_list.shape[0]
    if n_input_neurons is None:
        n_input_neurons = n_neurons // 2

    if frame_indices is None:
        frame_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4]

    # External input is stored in column 4
    external_input = x_list[:, :, 4]  # (n_frames, n_neurons)

    # Check if there's actual external input variation
    if np.abs(external_input).max() < 1e-6:
        print('  no external input detected, skipping plot')
        return

    # Get the grid size for input neurons
    n_per_axis = int(np.sqrt(n_input_neurons))

    fig, axes = plt.subplots(1, len(frame_indices), figsize=(4*len(frame_indices), 4))
    if len(frame_indices) == 1:
        axes = [axes]

    for idx, frame in enumerate(frame_indices):
        if frame >= n_frames:
            frame = n_frames - 1

        # Get external input at this frame for input neurons only
        ext_at_frame = external_input[frame, :n_input_neurons]

        # Reshape to 2D grid
        ext_field = ext_at_frame.reshape(n_per_axis, n_per_axis)
        ext_field = np.rot90(ext_field, k=1)

        im = axes[idx].imshow(ext_field, cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(f't = {frame}', fontsize=14)
        axes[idx].axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label=r'$\Omega_i(t)$')
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/external_input.png", dpi=100)
    plt.close()


def plot_activity_sample(x_list, n_neurons, n_frames, dataset_name, n_traces=10):
    """Plot a sample of activity traces (for Figure 3c).

    Args:
        x_list: Array of shape (n_frames, n_neurons, n_features)
        n_neurons: Total number of neurons
        n_frames: Number of frames
        dataset_name: Name of dataset for saving
        n_traces: Number of traces to plot (default: 10)
    """
    print('plot activity sample ...')

    activity = x_list[:, :, 3:4]
    activity = activity.squeeze()
    activity = activity.T  # (n_neurons, n_frames)

    # Sample n_traces neurons
    np.random.seed(42)  # For reproducibility
    sampled_indices = np.random.choice(n_neurons, n_traces, replace=False)
    sampled_indices = np.sort(sampled_indices)
    activity_plot = activity[sampled_indices]

    # Offset traces for visibility
    offset = np.abs(activity_plot).max() * 1.5
    activity_plot = activity_plot - offset * np.arange(n_traces)[:, None]

    plt.figure(figsize=(12, 6))
    for i in range(n_traces):
        plt.plot(activity_plot[i], linewidth=1.5, alpha=0.8)
        plt.text(-50, activity_plot[i, 0], str(sampled_indices[i]), fontsize=10, va='center', ha='right')

    plt.xlabel("time", fontsize=16)
    plt.ylabel("neuron index", fontsize=16)
    plt.xlim([0, min(n_frames, 10000)])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/activity_sample.png", dpi=100)
    plt.close()


def plot_synaptic_mlp_functions(model, x_list, n_neurons, dataset_name, colormap, device, signal_model_name=None):
    """Plot MLP0 and MLP1 functions for synaptic simulation.

    For PDE_N5, plots a 2x2 montage showing neuron-neuron dependent transfer functions.
    Each subplot shows target neuron type k receiving from all source neuron types.
    """
    if not hasattr(model, 'func'):
        return

    print('plot MLP0 and MLP1 functions ...')
    xnorm = np.std(x_list[:, :, 3])
    import torch
    rr = torch.linspace(-xnorm, xnorm, 1000).to(device)
    neuron_types = x_list[0, :, 6].astype(int)  # neuron_type is at column 6
    n_neuron_types = int(neuron_types.max()) + 1
    cmap = plt.cm.get_cmap(colormap)

    # For PDE_N5: plot 2x2 montage of neuron-neuron dependent MLP1
    if signal_model_name == 'PDE_N5' and n_neuron_types == 4:
        print('  PDE_N5: plotting 2x2 neuron-neuron dependent MLP1 montage ...')
        fig = plt.figure(figsize=(16, 16))
        plt.axis('off')

        for k in range(n_neuron_types):  # target neuron type
            ax = fig.add_subplot(2, 2, k + 1)
            # Color the subplot border by target neuron type
            for spine in ax.spines.values():
                spine.set_edgecolor(cmap(k))
                spine.set_linewidth(3)

            if k == 0:
                plt.ylabel(r'$\psi^*(a_i, a_j, x_i)$', fontsize=32)

            # Plot MLP1 for all source neuron types -> target type k
            for n in range(n_neuron_types):  # source neuron type
                # Get width (w) from target neuron type k
                w_target = model.p[k, 4:5]  # width of target

                # Sample multiple neurons of source type n
                for m in range(250):
                    # Get threshold (h) from source neuron type n
                    if model.p.shape[1] >= 6:
                        h_source = model.p[n, 5:6]
                    else:
                        h_source = torch.zeros_like(w_target)

                    # Compute phi((u - h_source) / w_target)
                    func_phi = model.phi((rr[:, None] - h_source) / w_target)
                    # Add the log term: - u * log(w_source) / 50
                    l_source = torch.log(model.p[n, 4:5])
                    func_phi = func_phi - rr[:, None] * l_source / 50

                    plt.plot(to_numpy(rr), to_numpy(func_phi), color=cmap(n),
                             linewidth=2, alpha=0.25)

            plt.ylim([-1.1, 1.1])
            plt.xlim([-5, 5])
            if k >= 2:  # bottom row
                plt.xlabel(r'$x_j$', fontsize=32)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/MLP1_neuron_neuron.png", dpi=150)
        plt.close()

    # Plot MLP1 (message/phi function) - all neurons (standard plot)
    plt.figure(figsize=(10, 8))
    for n in range(n_neurons):
        neuron_type = neuron_types[n]
        func_phi = model.func(rr, neuron_type, 'phi')
        plt.plot(to_numpy(rr), to_numpy(func_phi), color=cmap(neuron_type), linewidth=1, alpha=0.5)
    plt.xlabel('$x$', fontsize=32)
    plt.ylabel(r'$\mathrm{MLP}_1(x)$', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/MLP1_function.png", dpi=300)
    plt.close()

    # Plot MLP0 (update function) - all neurons
    plt.figure(figsize=(10, 8))
    for n in range(n_neurons):
        neuron_type = neuron_types[n]
        func_update = model.func(rr, neuron_type, 'update')
        plt.plot(to_numpy(rr), to_numpy(func_update), color=cmap(neuron_type), linewidth=1, alpha=0.5)
    plt.xlabel('$x$', fontsize=32)
    plt.ylabel(r'$\mathrm{MLP}_0(x)$', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/MLP0_function.png", dpi=300)
    plt.close()


def plot_eigenvalue_spectrum(connectivity, dataset_name, mc='k', log_file=None):
    """Plot eigenvalue spectrum of connectivity matrix (3 panels)."""
    gt_weight = to_numpy(connectivity)
    eig_true, _ = np.linalg.eig(gt_weight)

    # Sort eigenvalues by magnitude
    idx_true = np.argsort(-np.abs(eig_true))
    eig_true_sorted = eig_true[idx_true]
    spectral_radius = np.max(np.abs(eig_true))

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # (0) eigenvalues in complex plane
    axes[0].scatter(eig_true.real, eig_true.imag, s=50, c=mc, alpha=0.7, edgecolors='none')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('real', fontsize=32)
    axes[0].set_ylabel('imag', fontsize=32)
    axes[0].tick_params(labelsize=20)
    axes[0].set_title('eigenvalues in complex plane', fontsize=28)
    axes[0].text(0.05, 0.95, f'spectral radius: {spectral_radius:.3f}',
            transform=axes[0].transAxes, fontsize=20, verticalalignment='top')

    # (1) eigenvalue magnitude (sorted)
    axes[1].scatter(range(len(eig_true_sorted)), np.abs(eig_true_sorted), s=50, c=mc, alpha=0.7, edgecolors='none')
    axes[1].set_xlabel('index', fontsize=32)
    axes[1].set_ylabel('|eigenvalue|', fontsize=32)
    axes[1].tick_params(labelsize=20)
    axes[1].set_title('eigenvalue magnitude (sorted)', fontsize=28)

    # (2) eigenvalue spectrum (log scale)
    axes[2].plot(np.abs(eig_true_sorted), c=mc, linewidth=2)
    axes[2].set_xlabel('index', fontsize=32)
    axes[2].set_ylabel('|eigenvalue|', fontsize=32)
    axes[2].set_yscale('log')
    axes[2].tick_params(labelsize=20)
    axes[2].set_title('eigenvalue spectrum (log scale)', fontsize=28)

    plt.tight_layout()
    plt.savefig(f"./graphs_data/{dataset_name}/eigenvalues.png", dpi=150)
    plt.close()

    msg = f'spectral radius: {spectral_radius:.3f}'
    print(msg)
    if log_file:
        log_file.write(msg + '\n')
    return spectral_radius


def plot_connectivity_matrix(connectivity, output_path, vmin_vmax_method='minmax',
                              percentile=99, vmin=None, vmax=None,
                              show_labels=True, show_title=True,
                              zoom_size=20, dpi=100, cbar_fontsize=32, label_fontsize=48):
    """Plot connectivity matrix heatmap with zoom inset.

    Args:
        connectivity: Connectivity matrix (torch tensor or numpy array)
        output_path: Path to save the figure
        vmin_vmax_method: 'minmax' for full range, 'percentile' for percentile-based
        percentile: Percentile value if vmin_vmax_method='percentile' (default: 99)
        vmin: Explicit vmin value (overrides vmin_vmax_method if provided)
        vmax: Explicit vmax value (overrides vmin_vmax_method if provided)
        show_labels: Whether to show x/y axis labels (default: True)
        show_title: Whether to show title (default: True)
        zoom_size: Size of zoom inset (top-left NxN block, default: 20)
        dpi: Output DPI (default: 100)
        cbar_fontsize: Colorbar tick font size (default: 32)
        label_fontsize: Axis label font size (default: 48)
    """
    gt_weight = to_numpy(connectivity)
    n_neurons = gt_weight.shape[0]

    # Use explicit vmin/vmax if provided, otherwise compute based on method
    if vmin is None or vmax is None:
        if vmin_vmax_method == 'percentile':
            weight_pct = np.percentile(np.abs(gt_weight.flatten()), percentile)
            vmin, vmax = -weight_pct * 1.1, weight_pct * 1.1
        else:  # minmax
            weight_max = np.max(np.abs(gt_weight))
            vmin, vmax = -weight_max, weight_max

    # Main heatmap
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(gt_weight, center=0, square=True, cmap='bwr',
                     cbar_kws={'fraction': 0.046}, vmin=vmin, vmax=vmax)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)

    if show_labels:
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=label_fontsize)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=label_fontsize)
        plt.xticks(rotation=0)
    else:
        plt.xticks([])
        plt.yticks([])

    if show_title:
        plt.title('connectivity matrix', fontsize=28)

    # Zoom inset (top-left corner)
    if zoom_size > 0 and n_neurons >= zoom_size:
        plt.subplot(2, 2, 1)
        sns.heatmap(gt_weight[0:zoom_size, 0:zoom_size], cbar=False,
                    center=0, square=True, cmap='bwr', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_signal_loss(loss_dict, log_dir, epoch=None, Niter=None, debug=False,
                     current_loss=None, current_regul=None, total_loss=None,
                     total_loss_regul=None):
    """
    Plot stratified loss components over training iterations.

    Creates a two-panel figure showing loss and regularization terms in both
    linear and log scale. Saves to {log_dir}/tmp_training/loss.tif.

    Parameters:
    -----------
    loss_dict : dict
        Dictionary containing loss component lists with keys:
        - 'loss': Loss without regularization
        - 'regul_total': Total regularization loss
        - 'W_L1': W L1 sparsity penalty
        - 'W_L2': W L2 regularization penalty
        - 'edge_diff': Edge monotonicity penalty
        - 'edge_norm': Edge normalization
        - 'edge_weight': Edge MLP weight regularization
        - 'phi_weight': Phi MLP weight regularization
        - 'W_sign': W sign consistency penalty
    log_dir : str
        Directory to save the figure
    epoch : int, optional
        Current epoch number
    Niter : int, optional
        Number of iterations per epoch
    debug : bool, optional
        If True, print debug information about loss components
    current_loss : float, optional
        Current iteration total loss (for debug)
    current_regul : float, optional
        Current iteration regularization (for debug)
    total_loss : float, optional
        Accumulated total loss (for debug)
    total_loss_regul : float, optional
        Accumulated regularization loss (for debug)
    """
    if len(loss_dict['loss']) == 0:
        return

    # Debug output if requested
    if debug and current_loss is not None and current_regul is not None:
        current_pred_loss = current_loss - current_regul

        # Get current iteration component values (last element in each list)
        comp_sum = (loss_dict['W_L1'][-1] + loss_dict['W_L2'][-1] +
                   loss_dict['edge_diff'][-1] + loss_dict['edge_norm'][-1] +
                   loss_dict['edge_weight'][-1] + loss_dict['phi_weight'][-1] +
                   loss_dict['W_sign'][-1])

        print(f"\n=== DEBUG Loss Components (Epoch {epoch}, Iter {Niter}) ===")
        print("Current iteration:")
        print(f"  loss.item() (total): {current_loss:.6f}")
        print(f"  regul_this_iter: {current_regul:.6f}")
        print(f"  prediction_loss (loss - regul): {current_pred_loss:.6f}")
        print("\nRegularization breakdown:")
        print(f"  W_L1: {loss_dict['W_L1'][-1]:.6f}")
        print(f"  W_L2: {loss_dict['W_L2'][-1]:.6f}")
        print(f"  W_sign: {loss_dict['W_sign'][-1]:.6f}")
        print(f"  edge_diff: {loss_dict['edge_diff'][-1]:.6f}")
        print(f"  edge_norm: {loss_dict['edge_norm'][-1]:.6f}")
        print(f"  edge_weight: {loss_dict['edge_weight'][-1]:.6f}")
        print(f"  phi_weight: {loss_dict['phi_weight'][-1]:.6f}")
        print(f"  Sum of components: {comp_sum:.6f}")
        if total_loss is not None and total_loss_regul is not None:
            print("\nAccumulated (for reference):")
            print(f"  total_loss (accumulated): {total_loss:.6f}")
            print(f"  total_loss_regul (accumulated): {total_loss_regul:.6f}")
        if current_loss > 0:
            print(f"\nRatio: regul / loss (current iter) = {current_regul / current_loss:.4f}")
        if current_pred_loss < 0:
            print("\n⚠️  WARNING: Negative prediction loss! regul > total loss")
        print("="*60)

    fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Add epoch and iteration info as text annotation
    info_text = ""
    if epoch is not None:
        info_text += f"epoch: {epoch}"
    if Niter is not None:
        if info_text:
            info_text += " | "
        info_text += f"iterations/epoch: {Niter}"
    if info_text:
        fig_loss.suptitle(info_text, fontsize=20, y=0.995)

    # Linear scale
    ax1.plot(loss_dict['loss'], color='b', linewidth=4, label='loss (no regul)', alpha=0.8)
    ax1.plot(loss_dict['regul_total'], color='b', linewidth=2, label='total regularization', alpha=0.8)
    ax1.plot(loss_dict['W_L1'], color='r', linewidth=1.5, label='W L1 sparsity', alpha=0.7)
    ax1.plot(loss_dict['W_L2'], color='darkred', linewidth=1.5, label='W L2 regul', alpha=0.7)
    ax1.plot(loss_dict['W_sign'], color='navy', linewidth=1.5, label='W sign (Dale)', alpha=0.7)
    ax1.plot(loss_dict['phi_weight'], color='lime', linewidth=1.5, label='MLP0 Weight Regul', alpha=0.7)
    ax1.plot(loss_dict['edge_diff'], color='orange', linewidth=1.5, label='MLP1 monotonicity', alpha=0.7)
    ax1.plot(loss_dict['edge_norm'], color='brown', linewidth=1.5, label='MLP1 norm', alpha=0.7)
    ax1.plot(loss_dict['edge_weight'], color='pink', linewidth=1.5, label='MLP1 weight regul', alpha=0.7)
    ax1.set_xlabel('iteration', fontsize=16)
    ax1.set_ylabel('loss', fontsize=16)
    ax1.set_title('loss vs iteration', fontsize=18)
    ax1.legend(fontsize=10, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)

    # Log scale
    ax2.plot(loss_dict['loss'], color='b', linewidth=4, label='loss (no regul)', alpha=0.8)
    ax2.plot(loss_dict['regul_total'], color='b', linewidth=2, label='total regularization', alpha=0.8)
    ax2.plot(loss_dict['W_L1'], color='r', linewidth=1.5, label='W L1 sparsity', alpha=0.7)
    ax2.plot(loss_dict['W_L2'], color='darkred', linewidth=1.5, label='W L2 regul', alpha=0.7)
    ax2.plot(loss_dict['W_sign'], color='navy', linewidth=1.5, label='W sign (Dale)', alpha=0.7)
    ax2.plot(loss_dict['phi_weight'], color='lime', linewidth=1.5, label='MLP0 Weight Regul', alpha=0.7)
    ax2.plot(loss_dict['edge_diff'], color='orange', linewidth=1.5, label='MLP1 monotonicity', alpha=0.7)
    ax2.plot(loss_dict['edge_norm'], color='brown', linewidth=1.5, label='MLP1 norm', alpha=0.7)
    ax2.plot(loss_dict['edge_weight'], color='pink', linewidth=1.5, label='MLP1 weight regul', alpha=0.7)
    ax2.set_xlabel('iteration', fontsize=16)
    ax2.set_ylabel('loss', fontsize=16)
    ax2.set_yscale('log')
    ax2.set_title('loss vs iteration (Log)', fontsize=18)
    ax2.legend(fontsize=10, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(f'{log_dir}/tmp_training/loss.png', dpi=150)
    plt.close()

