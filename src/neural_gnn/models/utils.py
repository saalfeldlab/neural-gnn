import matplotlib.pyplot as plt
import os
import torch
import torch_geometric.data as data

from neural_gnn.models import Signal_Propagation
from neural_gnn.models.Siren_Network import Siren, Siren_Network
from neural_gnn.models.LowRank_INR import LowRankINR
from neural_gnn.models.HashEncoding_Network import HashEncodingMLP
from neural_gnn.utils import to_numpy, fig_init, map_matrix, choose_boundary_values
import warnings
import numpy as np

# Optional import
try:
    import umap
except ImportError:
    umap = None

import seaborn as sns
from scipy.optimize import curve_fit
import json
from pathlib import Path
from collections import Counter

def linear_model(x, a, b):
    return a * x + b


def compute_normalization_value(func_values, x_values, method='plateau',
                                 x_start=None, x_stop=None, derivative_threshold=0.01,
                                 per_neuron=False):
    """
    Compute normalization value for MLP output (e.g., transfer function psi).

    Args:
        func_values: tensor of shape (n_neurons, n_points) - function values
        x_values: tensor of shape (n_points,) - x coordinates
        method: str - 'max', 'median', 'mean', or 'plateau'
        x_start: float - start of range for normalization (default: min(x_values))
        x_stop: float - end of range for normalization (default: max(x_values))
        derivative_threshold: float - relative threshold for plateau detection
        per_neuron: bool - if True, return per-neuron values (tensor), else single scalar

    Returns:
        If per_neuron=False: normalization_value (float) - single value to normalize by
        If per_neuron=True: normalization_values (tensor) - per-neuron values, shape (n_neurons,)
    """
    func_values = func_values.detach()
    x_values = x_values.detach()

    # Default range
    if x_start is None:
        x_start = x_values.min().item()
    if x_stop is None:
        x_stop = x_values.max().item()

    # Filter to range [x_start, x_stop]
    mask = (x_values >= x_start) & (x_values <= x_stop)
    x_range = x_values[mask]
    func_range = func_values[:, mask]  # (n_neurons, n_points_in_range)

    if func_range.shape[1] < 2:
        # Not enough points, fall back to max
        if per_neuron:
            return func_values.abs().max(dim=1)[0]
        return func_values.abs().max().item()

    if method == 'max':
        # Maximum absolute value in range
        if per_neuron:
            return func_range.abs().max(dim=1)[0]
        return func_range.abs().max().item()

    elif method == 'median':
        if per_neuron:
            # Per-neuron median across points in range
            return func_range.median(dim=1)[0]
        # Median of mean across neurons
        neuron_means = func_range.mean(dim=1)
        return neuron_means.median().item()

    elif method == 'mean':
        if per_neuron:
            # Per-neuron mean across points in range
            return func_range.mean(dim=1)
        # Mean value in range (across all neurons and points)
        return func_range.mean().item()

    elif method == 'plateau':
        # Detect plateau by finding where derivative is flat
        # Compute finite differences along x
        dx = x_range[1] - x_range[0]
        if dx == 0:
            if per_neuron:
                return func_range.mean(dim=1)
            return func_range.mean().item()

        if per_neuron:
            # Per-neuron plateau detection
            n_neurons = func_range.shape[0]
            norm_values = torch.zeros(n_neurons, device=func_values.device)

            for n in range(n_neurons):
                func_n = func_range[n]  # (n_points_in_range,)
                d_func = (func_n[1:] - func_n[:-1]) / dx
                max_derivative = d_func.abs().max().item()

                if max_derivative < 1e-10:
                    norm_values[n] = func_n.mean()
                    continue

                threshold = derivative_threshold * max_derivative
                plateau_mask = d_func.abs() < threshold

                if plateau_mask.sum() < 2:
                    # No plateau, use mean in range
                    norm_values[n] = func_n.mean()
                else:
                    plateau_indices = torch.where(plateau_mask)[0]
                    norm_values[n] = func_n[plateau_indices].mean()

            return norm_values
        else:
            # Global plateau detection (mean across neurons)
            func_mean = func_range.mean(dim=0)  # (n_points_in_range,)

            # Compute derivative (finite difference)
            d_func = (func_mean[1:] - func_mean[:-1]) / dx

            # Find where derivative is small (plateau region)
            max_derivative = d_func.abs().max().item()
            if max_derivative < 1e-10:
                # Function is essentially flat everywhere
                return func_mean.mean().item()

            threshold = derivative_threshold * max_derivative
            plateau_mask = d_func.abs() < threshold

            if plateau_mask.sum() < 2:
                # No clear plateau found, use max
                print(f"  normalization: no plateau detected, using max value")
                return func_range.abs().max().item()

            # Get indices where plateau exists
            plateau_indices = torch.where(plateau_mask)[0]

            # Use the values in plateau region (offset by 1 due to derivative)
            plateau_values = func_mean[plateau_indices]
            norm_value = plateau_values.mean().item()

            print(f"  normalization: plateau detected at {plateau_mask.sum().item()} points, value={norm_value:.4f}")
            return norm_value

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_mlp_normalization(model, config, device, method='plateau',
                          x_start=None, x_stop=None, xnorm=None):
    """
    Compute normalization correction for MLP1 (transfer function psi).

    Args:
        model: trained GNN model
        config: NeuralGraphConfig
        device: torch device
        method: 'max', 'median', 'mean', or 'plateau'
        x_start: start of x range (default: xnorm)
        x_stop: end of x range (default: 3*xnorm)
        xnorm: normalization reference (auto-detected if None)

    Returns:
        correction: tensor of shape (n_neurons,) - per-neuron correction factors
    """
    from tqdm import trange

    n_neurons = config.simulation.n_neurons
    model_config = config.graph_model

    # Auto-detect xnorm from data if not provided
    if xnorm is None:
        xnorm = torch.tensor(5.0, device=device)  # Default fallback

    # Set default range for plateau detection
    if x_start is None:
        x_start = xnorm.item()
    if x_stop is None:
        x_stop = 3.0 * xnorm.item()

    print(f"Computing MLP1 normalization: method={method}, x_range=[{x_start:.2f}, {x_stop:.2f}]")

    # Sample x values
    n_samples = 1000
    rr = torch.linspace(x_start, x_stop, n_samples).to(device)

    # Evaluate MLP1 for each neuron
    func_list = []
    for n in trange(0, n_neurons, ncols=90, desc="Evaluating MLP1"):
        if model_config.signal_model_name in ['PDE_N4', 'PDE_N5', 'PDE_N7', 'PDE_N11']:
            embedding_ = model.a[n, :] * torch.ones((n_samples, config.graph_model.embedding_dim), device=device)
            if model_config.signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
            elif model_config.signal_model_name == 'PDE_N5':
                in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
        else:
            in_features = rr[:, None]

        with torch.no_grad():
            func = model.lin_edge(in_features.float())

        if config.graph_model.lin_edge_positive:
            func = func ** 2

        func_list.append(func)

    func_list = torch.stack(func_list).squeeze()  # (n_neurons, n_samples)

    # Compute normalization value
    norm_value = compute_normalization_value(
        func_list, rr, method=method,
        x_start=x_start, x_stop=x_stop
    )

    # Per-neuron correction: use mean per neuron in plateau region
    neuron_plateau_means = func_list.mean(dim=1)  # (n_neurons,)
    correction = 1.0 / (neuron_plateau_means + 1e-16)

    print(f"  global norm_value={norm_value:.4f}")
    print(f"  per-neuron correction: min={correction.min():.4f}, max={correction.max():.4f}, mean={correction.mean():.4f}")

    return correction

def get_embedding(model_a=None, dataset_number = 0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    return embedding

def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_neurons=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    indexes = np.arange(n_frames) * n_neurons + cell_id

    return embedding[indexes]

def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_neurons=None, n_frames=None, has_cell_division=None):

    indexes = np.arange(n_frames) * n_neurons + cell_id

    return new_labels[indexes]

def get_in_features_update(rr=None, model=None, embedding = None, device=None):

    n_neurons = model.n_neurons
    model_update_type = model.update_type

    if embedding == None:
        embedding = model.a[0:n_neurons]
        if model.embedding_trial:
            embedding = torch.cat((embedding, model.b[0].repeat(n_neurons, 1)), dim=1)


    if rr == None:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.zeros((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device),
                    torch.zeros((n_neurons, model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.ones((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((torch.zeros((n_neurons, 1), device=device), embedding), dim=1)
    else:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.zeros((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.zeros((rr.shape[0], model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((rr, embedding), dim=1)

    return in_features

def get_in_features_lin_edge(x, model, model_config, xnorm, n_neurons, device):

    signal_model_name = model_config.signal_model_name

    if signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
        # in_features for lin_edge: [u_j, embedding_j] where u is x[:,3:4]
        in_features_prev = torch.cat((x[:n_neurons, 3:4] - xnorm / 150, model.a[:n_neurons]), dim=1)
        in_features = torch.cat((x[:n_neurons, 3:4], model.a[:n_neurons]), dim=1)
        in_features_next = torch.cat((x[:n_neurons, 3:4] + xnorm / 150, model.a[:n_neurons]), dim=1)
        if model.embedding_trial:
            in_features_prev = torch.cat((in_features_prev, model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features = torch.cat((in_features, model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((in_features_next, model.b[0].repeat(n_neurons, 1)), dim=1)
    elif signal_model_name == 'PDE_N5':
        # in_features for lin_edge: [u_j, embedding_i, embedding_j] where u is x[:,3:4]
        if model.embedding_trial:
            in_features = torch.cat((x[:n_neurons, 3:4], model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[:n_neurons], model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 3:4] + xnorm / 150, model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[:n_neurons], model.b[0].repeat(n_neurons, 1)), dim=1)
        else:
            in_features = torch.cat((x[:n_neurons, 3:4], model.a[:n_neurons], model.a[:n_neurons]), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 3:4] + xnorm / 150, model.a[:n_neurons], model.a[:n_neurons]), dim=1)
    elif ('PDE_N9_A' in signal_model_name) | (signal_model_name == 'PDE_N9_C') | (signal_model_name == 'PDE_N9_D') :
        in_features = torch.cat((x[:, 3:4], model.a), dim=1)
        in_features_next = torch.cat((x[:,3:4] * 1.05, model.a), dim=1)
    elif signal_model_name == 'PDE_N9_B':
        perm_indices = torch.randperm(n_neurons, device=model.a.device)
        in_features = torch.cat((x[:, 3:4], x[:, 3:4], model.a, model.a[perm_indices]), dim=1)
        in_features_next = torch.cat((x[:, 3:4], x[:, 3:4] * 1.05, model.a, model.a[perm_indices]), dim=1)
    elif signal_model_name == 'PDE_N8':
        # in_features for lin_edge: [u_i, u_j, embedding_i, embedding_j] where u is x[:,3:4]
        if model.embedding_trial:
            perm_indices = torch.randperm(n_neurons, device=model.a.device)
            in_features = torch.cat((x[:n_neurons, 3:4], x[:n_neurons, 3:4], model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[perm_indices[:n_neurons]], model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 3:4], x[:n_neurons, 3:4]*1.05, model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[perm_indices[:n_neurons]], model.b[0].repeat(n_neurons, 1)), dim=1)
        else:
            perm_indices = torch.randperm(n_neurons, device=model.a.device)
            in_features = torch.cat((x[:n_neurons, 3:4], x[:n_neurons, 3:4], model.a[:n_neurons], model.a[perm_indices[:n_neurons]]), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 3:4], x[:n_neurons, 3:4] * 1.05, model.a[:n_neurons], model.a[perm_indices[:n_neurons]]), dim=1)
    else:
        # default: just u (signal) where u is x[:,3:4]
        in_features = x[:n_neurons, 3:4]
        in_features_next = x[:n_neurons, 3:4] + xnorm / 150
        in_features_prev = x[:n_neurons, 3:4] - xnorm / 150

    return in_features, in_features_next

def get_in_features(rr=None, embedding=None, model=[], model_name = [], max_radius=[]):

    if model.embedding_trial:
        embedding = torch.cat((embedding, model.b[0].repeat(embedding.shape[0], 1)), dim=1)

    match model_name:
        case 'PDE_A' | 'PDE_Cell_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_ParticleField_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_A_bis':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_B' | 'PDE_Cell_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_ParticleField_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_GS':
            in_features = torch.cat(
                (rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, 10 ** embedding), dim=1)
        case 'PDE_G':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_E':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_N2' | 'PDE_N3' | 'PDE_N6' :
            in_features = rr[:, None]
        case 'PDE_N4' | 'PDE_N7' | 'PDE_N11':
            in_features = torch.cat((rr[:, None], embedding), dim=1)
        case 'PDE_N8':
            in_features = torch.cat((rr[:, None]*0, rr[:, None], embedding, embedding), dim=1)
        case 'PDE_N5':
            in_features = torch.cat((rr[:, None], embedding, embedding), dim=1)
        case 'PDE_K':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius), dim=1)
        case 'PDE_F':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius, rr[:, None] / max_radius, embedding, embedding), dim=-1)
        case 'PDE_M':
            in_features = torch.cat((rr[:, None] / max_radius, rr[:, None] / max_radius, embedding, embedding), dim=-1)

    return in_features

def plot_training_signal(config, model, x, connectivity, log_dir, epoch, N, n_neurons, type_list, cmap, mc, device):

    if 'PDE_N3' in config.graph_model.signal_model_name:

        fig, ax = fig_init()
        plt.scatter(to_numpy(model.a[:-200, 0]), to_numpy(model.a[:-200, 1]), s=1, color='k', alpha=0.1, edgecolor='none')

    else:
        fig = plt.figure(figsize=(8, 8))
        # For large neuron counts, subsample for plotting
        if n_neurons > 5000:
            step = max(1, n_neurons // 1000)  # Plot ~1000 neurons max
        else:
            step = 1
        for n in range(0, n_neurons, step):
            if x[n, 3] != config.simulation.baseline_value:
                plt.scatter(to_numpy(model.a[n, 0]), to_numpy(model.a[n, 1]), s=100 if n_neurons <= 5000 else 20,
                            color=cmap.color(int(type_list[n])), alpha=1.0 if n_neurons <= 5000 else 0.5, edgecolors='none')

    plt.xlabel(r'$a_0$', fontsize=48)
    plt.ylabel(r'$a_1$', fontsize=48)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    gt_weight = to_numpy(connectivity)

    if config.training.multi_connectivity:
        pred_weight = to_numpy(model.W[0, :n_neurons, :n_neurons].clone().detach())
    else:
        pred_weight = to_numpy(model.W[:n_neurons, :n_neurons].clone().detach())
    np.fill_diagonal(pred_weight, 0)

    if 'PDE_N11' in config.graph_model.signal_model_name:
        weight_variable = '$J_{ij}$'
        signal_variable = '$h_i$'
    else:
        weight_variable = '$W_{ij}$'
        signal_variable = '$v_i$'

    if n_neurons<1000:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        ax = sns.heatmap(gt_weight, center=0, vmin=-0.5, vmax=0.5, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.ylabel('postsynaptic', fontsize=16)
        plt.xlabel('presynaptic', fontsize=16)
        plt.title(f'true {weight_variable}', fontsize=16)
        ax = fig.add_subplot(122)
        ax = sns.heatmap(pred_weight / 10, center=0, vmin=-0.5, vmax=0.5, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.ylabel('postsynaptic', fontsize=16)
        plt.xlabel('presynaptic', fontsize=16)
        plt.title(f'learned {weight_variable}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/matrix_{epoch}_{N}.tif", dpi=80)
        plt.close()

    fig, ax = fig_init()
    # Flatten data for scatterplot and R² computation
    x_data = gt_weight.flatten()
    y_data = (pred_weight / 10).flatten()

    # For large neuron counts, subsample for plotting (but compute R² on full data)
    if n_neurons > 5000:
        # Subsample to ~1M points for plotting (full data has n_neurons^2 points)
        n_total = len(x_data)
        n_sample = min(1_000_000, n_total)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        indices = rng.choice(n_total, size=n_sample, replace=False)
        x_plot = x_data[indices]
        y_plot = y_data[indices]
        ax.scatter(x_plot, y_plot, s=0.05, c=mc, alpha=0.05)
    elif n_neurons < 1000:
        ax.scatter(x_data, y_data, s=1.0, c=mc, alpha=1.0)
    else:
        ax.scatter(x_data, y_data, s=0.1, c=mc, alpha=0.1)
    ax.set_xlabel(r'true $J_{ij}$', fontsize=48)
    ax.set_ylabel(r'learned $J_{ij}$', fontsize=48)
    if n_neurons >= 5000:
        ax.set_xlim([-0.05, 0.05])
    else:
        ax.set_xlim([-0.2, 0.2])
    # Compute and display R² and slope (on FULL data, not subsampled)
    lin_fit, _ = curve_fit(linear_model, x_data, y_data)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    ax.text(0.05, 0.95, f'$R^2$: {r_squared:.3f}', transform=ax.transAxes,
            fontsize=24, verticalalignment='top', color=mc)
    ax.text(0.05, 0.9, f'slope: {lin_fit[0]:.3f}', transform=ax.transAxes,
            fontsize=24, verticalalignment='top', color=mc)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif", dpi=87)
    plt.close()

    # Return r_squared for progress tracking
    connectivity_r2 = r_squared

    if ('PDE_N8' in config.graph_model.signal_model_name):
        os.makedirs(f"./{log_dir}/tmp_training/matrix/larynx", exist_ok=True)
        data_folder_name = f'./graphs_data/{config.dataset}/'
        with open(data_folder_name+"all_neuron_list.json", "r") as f:
            all_neuron_list = json.load(f)
        with open(data_folder_name+"larynx_neuron_list.json", "r") as f:
            larynx_neuron_list = json.load(f)
        larynx_pred_weight, index_larynx = map_matrix(larynx_neuron_list, all_neuron_list, pred_weight)
        larynx_gt_weight, _ = map_matrix(larynx_neuron_list, all_neuron_list, gt_weight)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        ax = sns.heatmap(larynx_pred_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        ax.set_xticks(range(len(larynx_neuron_list)))
        ax.set_xticklabels(larynx_neuron_list, fontsize=12, rotation=90)
        ax.set_yticks(range(len(larynx_neuron_list)))
        ax.set_yticklabels(larynx_neuron_list, fontsize=12)
        plt.ylabel('postsynaptic')
        plt.xlabel('presynaptic')
        ax = fig.add_subplot(122)
        ax = sns.heatmap(larynx_gt_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        ax.set_xticks(range(len(larynx_neuron_list)))
        ax.set_xticklabels(larynx_neuron_list, fontsize=12, rotation=90)
        ax.set_yticks(range(len(larynx_neuron_list)))
        ax.set_yticklabels(larynx_neuron_list, fontsize=12)
        plt.ylabel('postsynaptic')
        plt.xlabel('presynaptic')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/larynx/matrix_{epoch}_{N}.tif", dpi=87)
        plt.close()

        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
        fig = plt.figure(figsize=(8, 8))
        for idx, k in enumerate(np.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 10)):  # Corrected step size to generate 13 evenly spaced values
            for n in range(0, n_neurons, 4):
                embedding_i = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                embedding_j = model.a[np.random.randint(n_neurons), :] * torch.ones(
                    (1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat((torch.ones_like(rr[:, None]) * k, rr[:, None], embedding_i, embedding_j,model.b[0].repeat(1000, 1)), dim=1)
                else:
                    in_features = torch.cat((rr[:, None], torch.ones_like(rr[:, None]) * k, embedding_i, embedding_j), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                plt.plot(to_numpy(rr - k), to_numpy(func), 2, color=cmap.color(idx), linewidth=2, alpha=0.25)
        plt.xlabel(r'$x_i-x_j$', fontsize=18)
        plt.ylabel(r'$MLP_1(a_i, a_j, x_i, x_j)$', fontsize=18)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/func_{epoch}_{N}.tif", dpi=87)
        plt.close()

    else:

        fig = plt.figure(figsize=(8, 8))
        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
        func_list = []
        # For large neuron counts, subsample for plotting
        mlp1_step = max(1, n_neurons // 500) if n_neurons > 5000 else 1
        for n in range(0, n_neurons, mlp1_step):
            if config.graph_model.signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    embedding_ = torch.cat((embedding_, model.b[0].repeat(1000, 1)), dim=1)
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
            elif 'PDE_N5' in config.graph_model.signal_model_name:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat(
                        (rr[:, None], embedding_, model.b[0].repeat(1000, 1), embedding_, model.b[0].repeat(1000, 1)),
                        dim=1)
                else:
                    in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
            elif 'PDE_N8' in config.graph_model.signal_model_name:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, model.b[0].repeat(1000, 1),
                                             embedding_, model.b[0].repeat(1000, 1)), dim=1)
                else:
                    in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
            else:
                in_features = rr[:, None]
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            if config.graph_model.lin_edge_positive:
                func = func ** 2
            func_list.append(to_numpy(func))
            if (n % (2 * mlp1_step) == 0):
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=2 if n_neurons <= 5000 else 1, alpha=0.25 if n_neurons <= 5000 else 0.1)
        plt.xlim(config.plotting.xlim)
        all_func = np.concatenate(func_list)
        plt.ylim([np.min(all_func), np.max(all_func)])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        xlabel = signal_variable.replace('_i', '_j')
        if config.training.training_single_type:
            ylabel = rf'$\mathrm{{MLP_1}}({xlabel[1:-1]})$'
        else:
            ylabel = rf'$\mathrm{{MLP_1}}(a_j, {xlabel[1:-1]})$'
        plt.xlabel(xlabel, fontsize=48)
        plt.ylabel(ylabel, fontsize=48)
        plt.tight_layout()

        plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/func_{epoch}_{N}.tif", dpi=87)
        plt.close()

    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    fig = plt.figure(figsize=(8, 8))
    func_list = []
    # For large neuron counts, subsample for plotting
    mlp0_step = max(1, n_neurons // 500) if n_neurons > 5000 else 1
    for n in range(0, n_neurons, mlp0_step):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if 'generic' in config.graph_model.update_type:
            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, rr[:, None] * 0), dim=1)
        else:
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        func_list.append(to_numpy(func))
        if (n % (2 * mlp0_step) == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1 if n_neurons <= 5000 else 0.5, alpha=0.1 if n_neurons <= 5000 else 0.05)
    plt.xlim(config.plotting.xlim)
    all_func = np.concatenate(func_list)
    plt.ylim([np.min(all_func), np.max(all_func)])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    if config.training.training_single_type:
        ylabel = rf'$\mathrm{{MLP_0}}({signal_variable[1:-1]})$'
    else:
        ylabel = rf'$\mathrm{{MLP_0}}(a_i, {signal_variable[1:-1]})$'
    plt.xlabel(signal_variable, fontsize=48)
    plt.ylabel(ylabel, fontsize=48)

    plt.tight_layout()

    plt.savefig(f"./{log_dir}/tmp_training/function/MLP0/func_{epoch}_{N}.tif", dpi=87)
    plt.close()

    return connectivity_r2


def plot_training_signal_visual_input(x, n_input_neurons, external_input_type, log_dir, epoch, N):

    n_input_neurons_per_axis = int(np.sqrt(n_input_neurons))
    if 'visual' in external_input_type:
        tmp = torch.reshape(x[:n_input_neurons, 4:5], (n_input_neurons_per_axis, n_input_neurons_per_axis))
    else:
        tmp = torch.reshape(x[:, 4:5], (n_input_neurons_per_axis, n_input_neurons_per_axis))
    tmp = to_numpy(tmp)

    # compute stats for sanity check
    val_min = np.min(tmp)
    val_max = np.max(tmp)
    val_std = np.std(tmp)

    tmp = np.rot90(tmp, k=1)
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(tmp, cmap='gray')
    plt.text(0.02, 0.98, f'min={val_min:.2f} max={val_max:.2f} std={val_std:.2f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/external_input/external_input_{epoch}_{N}.tif", dpi=80)
    plt.close()

def plot_training_signal_missing_activity(n_frames, k, x_list, baseline_value, model_missing_activity, log_dir, epoch, N, device):

        if n_frames > 1000:
            t = torch.linspace(0, 1, n_frames//100, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
        prediction = model_missing_activity[0](t)
        prediction = prediction.t()
        fig = plt.figure(figsize=(16, 16))
        fig.add_subplot(2, 2, 1)
        plt.title('neural field')
        plt.imshow(to_numpy(prediction), aspect='auto', cmap='viridis')
        fig.add_subplot(2, 2, 2)
        plt.title('true activity')
        activity = torch.tensor(x_list[0][:, :, 6:7], device=device)
        activity = activity.squeeze()
        activity = activity.t()
        plt.imshow(to_numpy(activity), aspect='auto', cmap='viridis')
        plt.tight_layout()
        fig.add_subplot(2, 2, 3)
        plt.title('learned missing activity')
        pos = np.argwhere(x_list[0][k][:, 6] != baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        fig.add_subplot(2, 2, 4)
        plt.title('learned observed activity')
        pos = np.argwhere(x_list[0][k][:, 6] == baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/external_input/missing_activity_{epoch}_{N}.tif", dpi=80)
        plt.close()

def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model=None, n_nodes=0, n_neurons=None, ynorm=None, type_list=None, cmap=None, update_type=None, device=None):

    dimension = config.simulation.dimension

    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    if rr==[]:
        if 'PDE_N' in config_model:
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device) # noqa: F821

    print('interaction functions ...')
    func_list = []
    # For large neuron counts, subsample the loop itself
    analyze_step = max(1, n_neurons // 500) if n_neurons > 5000 else 1
    neuron_indices = range(0, n_neurons, analyze_step)
    for n in neuron_indices:

        if len(model.a.shape)==3:
            model_a= model.a[1, n, :]
        else:
            model_a = model.a[n, :]

        if (update_type != 'NA') & model.embedding_trial:
            embedding_ = torch.cat((model_a, model.b[0].clone().detach().repeat(n_neurons, 1)), dim=1) * torch.ones((1000, 2*dimension), device=device)
        else:
            embedding_ = model_a * torch.ones((1000, dimension), device=device)

        if update_type == 'NA':
            in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config_model, max_radius=max_radius) # noqa: F821
        else:
            in_features = get_in_features_update(rr=rr[:, None], embedding=embedding_, model=model, device=device)
        with torch.no_grad():
            func = model_MLP(in_features.float())[:, 0]

        func_list.append(func)

        should_plot = vizualize and (
                n_neurons <= 200 or
                (n % max(1, n_neurons // 200) == 0) or
                (config.graph_model.particle_model_name == 'PDE_GS') or
                ('PDE_N' in config_model and n_neurons <= 5000)
        )

        if should_plot:
            plt.plot(
                to_numpy(rr),
                to_numpy(func) * to_numpy(ynorm),
                2,
                color=cmap.color(type_list[n].astype(int)),
                linewidth=1,
                alpha=0.25
            )

    func_list = torch.stack(func_list)
    func_list_ = to_numpy(func_list)

    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_GS':
            plt.xscale('log')
            plt.yscale('log')
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xlim([1E-3, 0.02])
        if config.graph_model.particle_model_name == 'PDE_E':
            plt.xlim([0, 0.05])
        if 'PDE_N' in config.graph_model.particle_model_name:
            plt.xlim(config.plotting.xlim)


        # ylim = [np.min(func_list_)/1.05, np.max(func_list_)*1.05]
        plt.ylim(config.plotting.ylim)

    print('UMAP reduction ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if func_list_.shape[0] > 1000:
            new_index = np.random.permutation(func_list_.shape[0])
            new_index = new_index[0:min(1000, func_list_.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(func_list_[new_index])
            proj_interaction = trans.transform(func_list_)
        else:
            trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0).fit(func_list_)
            proj_interaction = trans.transform(func_list_)

    return func_list, proj_interaction

def choose_training_model(model_config=None, device=None):

    dataset_name = model_config.dataset
    aggr_type = model_config.graph_model.aggr_type
    dimension = model_config.simulation.dimension

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model=[]
    model_name = model_config.graph_model.particle_model_name
    match model_name:
        case 'PDE_R':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                     dimension=dimension)
        case 'PDE_MPM' | 'PDE_MPM_A':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                    dimension=dimension)
        case  'PDE_Cell' | 'PDE_Cell_area':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_ParticleField_A' | 'PDE_ParticleField_B':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_Agents' | 'PDE_Agents_A' | 'PDE_Agents_B' | 'PDE_Agents_C':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_A' | 'PDE_A_bis' | 'PDE_B' | 'PDE_B_mass' | 'PDE_B_bis' | 'PDE_E' | 'PDE_G' | 'PDE_K' | 'PDE_T':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
            if 'PDE_K' in model_name:
                model.connection_matrix = torch.load(f'./graphs_data/{dataset_name}/connection_matrix_list.pt', map_location=device)
        case 'PDE_GS':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_neurons)
            t1 = np.repeat(t, model_config.simulation.n_neurons)
            t2 = np.tile(t, model_config.simulation.n_neurons)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_GS2':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_neurons)
            t1 = np.repeat(t, model_config.simulation.n_neurons)
            t2 = np.tile(t, model_config.simulation.n_neurons)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_Cell_A' | 'PDE_Cell_B' | 'PDE_Cell_B_area' | 'PDE_Cell_A_area':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_F_A' |'PDE_F_B'|'PDE_F_C'|'PDE_F_D'|'PDE_F_E' :
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_MLPs' | 'PDE_MLPs_A' | 'PDE_MLPs_A_bis' | 'PDE_MLPs_A_ter' | 'PDE_MLPs_B'| 'PDE_MLPs_B_0' |'PDE_MLPs_B_1' | 'PDE_MLPs_B_4'| 'PDE_MLPs_B_10' |'PDE_MLPs_C' | 'PDE_MLPs_D' | 'PDE_MLPs_E' | 'PDE_MLPs_F':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device,
                                                bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_M' | 'PDE_M2':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MM' | 'PDE_MM_1layer' | 'PDE_MM_2layers' | 'PDE_MM_3layers':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MS':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config,bc_dpos=bc_dpos, dimension=dimension, device=device)

    model_name = model_config.graph_model.mesh_model_name
    match model_name:
        case 'DiffMesh':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMesh':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMeshSmooth':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' | 'RD_Mesh4':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    model_name = model_config.graph_model.signal_model_name
    match model_name:
        case 'PDE_N2' | 'PDE_N3' | 'PDE_N4' | 'PDE_N5' | 'PDE_N6' | 'PDE_N7' | 'PDE_N9' | 'PDE_N8' | 'PDE_N11':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'PDE_WBI':
            from neural_gnn.models import WBI_Communication
            model = WBI_Communication(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    if model==[]:
        raise ValueError(f'Unknown model {model_name}')

    return model, bc_pos, bc_dpos


def choose_inr_model(config=None, n_neurons=None, n_frames=None, x_list=None, device=None):
    """
    create INR model for external input reconstruction.

    hierarchy: visual > signal > none
    for signal input: use inr_type to select representation (siren_t, siren_id, siren_x, ngp, lowrank)
    for visual input: use Siren_Network with nnr_f params

    returns None if learn_external_input is False or external_input_type is 'none'
    """
    simulation_config = config.simulation
    model_config = config.graph_model
    train_config = config.training

    external_input_type = simulation_config.external_input_type
    learn_external_input = train_config.learn_external_input
    inr_type = model_config.inr_type

    if not learn_external_input or external_input_type == 'none':
        return None

    model_f = None

    if external_input_type == 'visual':
        # visual input: use Siren_Network with nnr_f params
        n_input_neurons = simulation_config.n_input_neurons
        n_input_neurons_per_axis = int(np.sqrt(n_input_neurons))
        model_f = Siren_Network(
            image_width=n_input_neurons_per_axis,
            in_features=model_config.input_size_nnr_f,
            out_features=model_config.output_size_nnr_f,
            hidden_features=model_config.hidden_dim_nnr_f,
            hidden_layers=model_config.n_layers_nnr_f,
            outermost_linear=model_config.outermost_linear_nnr_f,
            device=device,
            first_omega_0=model_config.omega_f,
            hidden_omega_0=model_config.omega_f
        )

    elif external_input_type == 'signal':
        # signal input: use inr_type to select representation
        learnable_omega = model_config.omega_f_learning
        if inr_type == 'siren_t':
            model_f = Siren(
                in_features=1,
                hidden_features=model_config.hidden_dim_nnr_f,
                hidden_layers=model_config.n_layers_nnr_f,
                out_features=n_neurons,
                outermost_linear=model_config.outermost_linear_nnr_f,
                first_omega_0=model_config.omega_f,
                hidden_omega_0=model_config.omega_f,
                learnable_omega=learnable_omega
            )
        elif inr_type == 'siren_id':
            model_f = Siren(
                in_features=2,  # (t, id)
                hidden_features=model_config.hidden_dim_nnr_f,
                hidden_layers=model_config.n_layers_nnr_f,
                out_features=1,
                outermost_linear=model_config.outermost_linear_nnr_f,
                first_omega_0=model_config.omega_f,
                hidden_omega_0=model_config.omega_f,
                learnable_omega=learnable_omega
            )
        elif inr_type == 'siren_x':
            model_f = Siren(
                in_features=3,  # (t, x, y)
                hidden_features=model_config.hidden_dim_nnr_f,
                hidden_layers=model_config.n_layers_nnr_f,
                out_features=1,
                outermost_linear=model_config.outermost_linear_nnr_f,
                first_omega_0=model_config.omega_f,
                hidden_omega_0=model_config.omega_f,
                learnable_omega=learnable_omega
            )
        elif inr_type == 'ngp':
            model_f = HashEncodingMLP(
                n_input_dims=1,
                n_output_dims=n_neurons,
                n_levels=model_config.ngp_n_levels,
                n_features_per_level=model_config.ngp_n_features_per_level,
                log2_hashmap_size=model_config.ngp_log2_hashmap_size,
                base_resolution=model_config.ngp_base_resolution,
                per_level_scale=model_config.ngp_per_level_scale,
                n_neurons=model_config.ngp_n_neurons,
                n_hidden_layers=model_config.ngp_n_hidden_layers,
                output_activation='none'
            )
        elif inr_type == 'lowrank':
            # extract external_input for SVD init
            external_input_data = x_list[0][:, :, 4] if x_list is not None else None
            init_data = external_input_data if model_config.lowrank_svd_init else None
            model_f = LowRankINR(
                n_frames=n_frames,
                n_neurons=n_neurons,
                rank=model_config.lowrank_rank,
                init_data=init_data
            )

    if model_f is not None:
        model_f.to(device=device)
        print(f'external input model: {inr_type}, external_input_type={external_input_type}')

    return model_f


def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size

    return get_batch_size

def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size

    return get_batch_size


def set_trainable_parameters(model=[], lr_embedding=[], lr=[],  lr_update=[], lr_W=[], lr_modulation=[], learning_rate_NNR=[], learning_rate_NNR_f=[], learning_rate_NNR_E=[], learning_rate_NNR_b=[]):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params)

    # Only count model.a if it exists and requires gradients (not frozen by training_single_type)
    if hasattr(model, 'a') and model.a.requires_grad:
        n_total_params = n_total_params + torch.numel(model.a)


    if lr_update==[]:
        lr_update = lr

    param_groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name == 'a':
                param_groups.append({'params': parameter, 'lr': lr_embedding})
            elif (name=='b') or ('lin_modulation' in name):
                param_groups.append({'params': parameter, 'lr': lr_modulation})
            elif 'lin_phi' in name:
                param_groups.append({'params': parameter, 'lr': lr_update})
            elif 'W' in name:
                param_groups.append({'params': parameter, 'lr': lr_W})
            elif 'NNR_f' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR_f})
            elif 'NNR' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR})
            else:
                param_groups.append({'params': parameter, 'lr': lr})

    # Use foreach=False to avoid CUDA device mismatch issues with multi-GPU setups
    optimizer = torch.optim.Adam(param_groups, foreach=False)

    return optimizer, n_total_params


def set_trainable_division_parameters(model, lr):
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.t)

    embedding = model.t
    optimizer = torch.optim.Adam([embedding], lr=lr)

    _, *parameters = trainable_params
    for parameter in parameters:
        optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params

def get_index_particles(x, n_neuron_types, dimension):
    index_particles = []
    for n in range(n_neuron_types):
        if dimension == 2:
            index = np.argwhere(x[:, 6].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    return index_particles

def get_type_list(x, dimension):
    type_list = x[:, 1 + 2 * dimension:2 + 2 * dimension].clone().detach()
    return type_list

def sample_synaptic_data_and_predict(model, x_list, edges, n_runs, n_frames, time_step, device,
                            has_missing_activity=False, model_missing_activity=None,
                            has_neural_field=False, model_f=None,
                            run=None, k=None):
    """
    Sample data from x_list and get model predictions

    Args:
        model: trained GNN model
        x_list: list of data arrays [n_runs][n_frames]
        edges: edge indices for graph
        n_runs, n_frames, time_step: data dimensions
        device: torch device
        has_missing_activity: whether to fill missing activity
        model_missing_activity: model for missing activity (if needed)
        has_neural_field: whether to compute neural field
        model_f: field model (if needed)
        run: specific run index (if None, random)
        k: specific frame index (if None, random)

    Returns:
        dict with pred, in_features, x, dataset, data_id, k_batch
    """
    # Sample random run and frame if not specified
    if run is None:
        run = np.random.randint(n_runs)
    if k is None:
        k = np.random.randint(n_frames - 4 - time_step)

    # Get data
    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)

    # Handle missing activity if needed
    if has_missing_activity and model_missing_activity is not None:
        pos = torch.argwhere(x[:, 3] == 6)
        if len(pos) > 0:
            t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
            missing_activity = model_missing_activity[run](t).squeeze()
            x[pos, 3] = missing_activity[pos]

    # Handle neural field if needed
    if has_neural_field and model_f is not None:
        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
        x[:, 4] = model_f[run](t) ** 2

    # Create dataset
    dataset = data.Data(x=x, edge_index=edges)
    data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k

    # Get predictions
    pred, in_features = model(dataset, data_id=data_id, k=k_batch, return_all=True)

    return {
        'pred': pred,
        'in_features': in_features,
        'x': x,
        'dataset': dataset,
        'data_id': data_id,
        'k_batch': k_batch,
        'run': run,
        'k': k
    }

def analyze_odor_responses_by_neuron(model, x_list, edges, n_runs, n_frames, time_step, device,
                                     all_neuron_list, has_missing_activity=False, model_missing_activity=None,
                                     has_neural_field=False, model_f=None, n_samples=50, run=0):
    """
    Analyze odor responses by comparing lin_phi output with and without excitation
    Returns top responding neurons by name for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']

    # Store responses: difference between excitation and baseline
    odor_responses = {odor: [] for odor in odor_list}
    valid_samples = 0

    model.eval()
    with torch.no_grad():
        sample = 0
        while valid_samples < n_samples:
            result = sample_synaptic_data_and_predict(
                model, x_list, edges, n_runs, n_frames, time_step, device,
                has_missing_activity, model_missing_activity,
                has_neural_field, model_f, run
            )

            if not (torch.isnan(result['x']).any()):
                # Get baseline response (no excitation)
                x_baseline = result['x'].clone()
                x_baseline[:, 10:13] = 0  # no excitation
                dataset_baseline = data.Data(x=x_baseline, edge_index=edges)
                pred_baseline = model(dataset_baseline, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                for i, odor in enumerate(odor_list):
                    x_odor = result['x'].clone()
                    x_odor[:, 10:13] = 0
                    x_odor[:, 10 + i] = 1  # activate specific odor

                    dataset_odor = data.Data(x=x_odor, edge_index=edges)
                    pred_odor = model(dataset_odor, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                    odor_diff = pred_odor - pred_baseline
                    odor_responses[odor].append(odor_diff.cpu())

                valid_samples += 1

            sample += 1
            if sample > n_samples * 10:
                break

        # Convert to tensors [n_samples, n_neurons]
        for odor in odor_list:
            odor_responses[odor] = torch.stack(odor_responses[odor]).squeeze()

    # Identify top responding neurons for each odor
    top_neurons = {}
    for odor in odor_list:
        # Calculate mean response across samples for each neuron
        mean_response = torch.mean(odor_responses[odor], dim=0)  # [n_neurons]

        # Get top 3 responding neurons (highest positive response)
        top_20_indices = torch.topk(mean_response, k=20).indices.cpu().numpy()
        top_20_names = [all_neuron_list[idx] for idx in top_20_indices]
        top_20_values = [mean_response[idx].item() for idx in top_20_indices]

        top_neurons[odor] = {
            'names': top_20_names,
            'indices': top_20_indices.tolist(),
            'values': top_20_values
        }

        print(f"\ntop 20 responding neurons for {odor}:")
        for i, (name, idx, val) in enumerate(zip(top_20_names, top_20_indices, top_20_values)):
            print(f"  {i + 1}. {name} : {val:.4f}")

    return odor_responses  # Return only odor_responses to match original function signature

def plot_odor_heatmaps(odor_responses):
    """
    Plot 3 separate heatmaps showing mean response per neuron for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']
    n_neurons = odor_responses['butanone'].shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, odor in enumerate(odor_list):
        # Compute mean response per neuron
        mean_responses = torch.mean(odor_responses[odor], dim=0).numpy()  # [n_neurons]

        # Reshape to 2D for heatmap (assuming square-ish layout)
        side_length = int(np.ceil(np.sqrt(n_neurons)))
        padded_responses = np.pad(mean_responses, (0, side_length ** 2 - n_neurons), 'constant')
        response_matrix = padded_responses.reshape(side_length, side_length)

        # Plot heatmap
        sns.heatmap(response_matrix, ax=axes[i], cmap='bwr', center=0,
                    cbar=False, square=True, xticklabels=False, yticklabels=False)
        axes[i].set_title(f'{odor} mean response')

    plt.tight_layout()
    return fig

def overlay_umap_refit_with_W_list(
    w_list,
    out_prefix="/groups/saalfeld/home/allierc/Py/NeuralGraph/graphs_data/fly/",   # folder containing flyvis_connectomes_W.npz
    figure_path=None,                                            # e.g. ".../overlay_all.png"
    show=True,
    # UMAP params
    neighbors=15,
    min_dist=0.1,
    metric="cosine",
    seed=0,
    # labeling
    label_bg=True,
    labels=None,                 # list of text labels (same length as number of new vectors)
    label_fontsize=7,
    label_y_offset_frac=0.015,
    # styling for new points
    markers=None,                # list like ["*", "D", "o", ...]
    sizes=None,                  # list like [140, 120, 120, ...]
    edgecolors="k",
    linewidths=1.2,
    colors=None,                 # list of facecolors
    verbose=True,
):
    """
    Load saved training W, append multiple new vector(s), refit UMAP on [W_ref; w_new...], and plot.

    Parameters
    ----------
    w_list : array-like
        One of:
          - list/tuple of 1D arrays, each shaped [E]
          - 2D array shaped [n_new, E]
          - single 1D array [E] (treated as one vector)
    labels : list of str (optional)
        Text labels for new points. Defaults to ["NEW_0", "NEW_1", ...].
    markers/sizes/colors : per-point style lists (optional)

    Returns
    -------
    dict with keys:
        emb_bg  : (n_train, 2) UMAP coords of saved training points
        emb_new : (n_new, 2)   UMAP coords of the new points
        ids_bg  : (n_train,)    ids for background (if present in npz)
        reducer : fitted UMAP object
    """
    out = Path(out_prefix)
    W_file = out / "flyvis_connectomes_W.npz"
    if not W_file.exists():
        raise FileNotFoundError(
            f"Missing training matrix: {W_file}\n"
            "Save it once in the collector, e.g.: "
            "np.savez_compressed(f'{out}_W.npz', W=W.astype(np.float32), model_ids=np.array(ok_ids,'<U3'))"
        )

    # --- load saved training matrix (and ids if present) ---
    W_npz = np.load(W_file, allow_pickle=False)
    if "W" in W_npz:
        W_ref = np.asarray(W_npz["W"], dtype=np.float32)
    elif "w" in W_npz:  # fallback key
        W_ref = np.asarray(W_npz["w"], dtype=np.float32)
    else:
        raise KeyError(f"{W_file} must contain array 'W' (or 'w').")
    ids_bg = W_npz.get("model_ids", np.array([f"{i:03d}" for i in range(W_ref.shape[0])], dtype="<U8"))

    # --- normalize incoming w_list to a 2D array (n_new, E) ---
    if isinstance(w_list, (list, tuple)):
        new_vecs = [np.asarray(w, dtype=np.float32).reshape(1, -1) for w in w_list]
        w_new = np.vstack(new_vecs) if len(new_vecs) > 0 else np.zeros((0, W_ref.shape[1]), np.float32)
    else:
        w_arr = np.asarray(w_list, dtype=np.float32)
        if w_arr.ndim == 1:
            w_new = w_arr.reshape(1, -1)
        elif w_arr.ndim == 2:
            w_new = w_arr
        else:
            raise ValueError("w_list must be (list/tuple of 1D), a 1D array, or a 2D array.")
    if w_new.shape[1] != W_ref.shape[1]:
        raise ValueError(f"Feature mismatch: new has {w_new.shape[1]} features; saved W has {W_ref.shape[1]}.")

    n_new = w_new.shape[0]
    if labels is None:
        labels = [f"NEW_{i}" for i in range(n_new)]
    # default styles
    if markers is None:
        # cycle a few nice markers
        base = ["*", "D", "o", "s", "^", "P", "X", "v"]
        markers = [base[i % len(base)] for i in range(n_new)]
    if sizes is None:
        sizes = [140] * n_new
    if colors is None:
        # None -> matplotlib cycles; we’ll pass no color and let scatter choose per call
        colors = [None] * n_new

    # --- concatenate and fit UMAP fresh on [W_ref; w_new] ---
    W_all = np.vstack([W_ref, w_new])
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=seed,
        init="spectral",
        verbose=verbose
    ).fit(W_all)

    emb_all = reducer.embedding_.astype(np.float32, copy=False)
    n_train = W_ref.shape[0]
    emb_bg  = emb_all[:n_train]
    emb_new = emb_all[n_train:]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(emb_bg[:, 0], emb_bg[:, 1], s=28, alpha=0.65, label="FlyVis (refit)")
    if label_bg:
        y_range = float(np.ptp(emb_all[:, 1])) if emb_all.size else 0.0
        dy = (label_y_offset_frac * y_range) if y_range > 0 else 0.02
        for i in range(n_train):
            ax.text(emb_bg[i, 0], emb_bg[i, 1] + dy, str(ids_bg[i]),
                    fontsize=label_fontsize, ha="center", va="bottom")

    # plot each new point with its own style + label
    for i in range(n_new):
        ax.scatter(
            emb_new[i:i+1, 0], emb_new[i:i+1, 1],
            s=sizes[i], marker=markers[i],
            edgecolors=edgecolors, linewidths=linewidths,
            label=labels[i],
            c=None if colors[i] is None else [colors[i]],
            zorder=3
        )

    ax.set_title("UMAP (refit) — FlyVis + new vector(s)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    if figure_path:
        fig.savefig(figure_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"emb_bg": emb_bg, "emb_new": emb_new, "ids_bg": ids_bg, "reducer": reducer}

from sklearn.neighbors import NearestNeighbors
import joblib

def overlay_barycentric_into_umap(
    w_list,
    out_prefix="/groups/saalfeld/home/allierc/Py/NeuralGraph/flyvis_connectomes",
    figure_path=None,
    show=True,
    metric="cosine",
    k=15,
    label_bg=True,
    labels=None,
    label_fontsize=7,
    label_y_offset_frac=0.015,
    eps_self=1e-12,   # for exact match snapping
):
    """
    Project new vectors into an existing UMAP background using K-NN barycentric weights.
    Requires the same W_ref the reducer was trained on (order must match).

    Files expected (from your collector):
      - {out_prefix}_W.npz               with arrays: W (or w), model_ids
      - {out_prefix}_umap_model.joblib   reducer fitted on W_ref
    """
    out = Path(out_prefix)
    W_file = out.with_name(out.name + "_W.npz")
    model_file = out.with_name(out.name + "_umap_model.joblib")

    # load training matrix + ids
    W_npz = np.load(W_file, allow_pickle=False)
    W_ref = np.asarray(W_npz["W"] if "W" in W_npz else W_npz["w"], dtype=np.float32)
    ids_bg = W_npz.get("model_ids", np.array([f"{i:03d}" for i in range(W_ref.shape[0])], dtype="<U8"))

    # load reducer to get the background embedding
    reducer = joblib.load(model_file)
    emb_bg = reducer.embedding_.astype(np.float32, copy=False)

    # prepare new vectors
    if isinstance(w_list, (list, tuple)):
        w_new = np.vstack([np.asarray(w, np.float32).reshape(1, -1) for w in w_list])
    else:
        arr = np.asarray(w_list, np.float32)
        w_new = arr.reshape(1, -1) if arr.ndim == 1 else arr
    if w_new.shape[1] != W_ref.shape[1]:
        raise ValueError(f"Feature mismatch: new has {w_new.shape[1]}, saved W has {W_ref.shape[1]}.")

    # KNN on the original high-dim space
    nbrs = NearestNeighbors(n_neighbors=min(k, W_ref.shape[0]), metric=metric)
    nbrs.fit(W_ref)

    emb_new = np.zeros((w_new.shape[0], 2), dtype=np.float32)
    for i, v in enumerate(w_new):
        # exact/self match snapping
        dists, idxs = nbrs.kneighbors(v[None, :], return_distance=True)
        dists = dists.ravel(); idxs = idxs.ravel()

        # if the closest neighbor is *itself* (zero distance), snap
        if dists[0] <= eps_self:
            emb_new[i] = emb_bg[idxs[0]]
            continue

        # inverse-distance weights (add tiny epsilon to avoid div by 0)
        wts = 1.0 / (dists + 1e-12)
        wts = wts / (wts.sum() + 1e-12)

        # barycentric combination of neighbor coordinates
        emb_new[i] = (wts[:, None] * emb_bg[idxs]).sum(axis=0)

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(emb_bg[:, 0], emb_bg[:, 1], s=28, alpha=0.65, label="FlyVis (fixed)")
    if label_bg:
        y_range = float(np.ptp(emb_bg[:, 1])) if emb_bg.size else 0.0
        dy = (label_y_offset_frac * y_range) if y_range > 0 else 0.02
        for i in range(emb_bg.shape[0]):
            ax.text(emb_bg[i, 0], emb_bg[i, 1] + dy, str(ids_bg[i]),
                    fontsize=label_fontsize, ha="center", va="bottom")

    if labels is None:
        labels = [f"NEW_{i}" for i in range(w_new.shape[0])]
    for i in range(w_new.shape[0]):
        ax.scatter(emb_new[i, 0], emb_new[i, 1], s=160, marker="*",
                   edgecolors="k", linewidths=1.2, label=labels[i])

    ax.set_title("UMAP (fixed) + KNN barycentric projection")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    if figure_path: fig.savefig(figure_path, dpi=220)
    if show: plt.show()
    else: plt.close(fig)

    return {"emb_bg": emb_bg, "emb_new": emb_new, "ids_bg": ids_bg}

def get_n_hop_neighborhood_with_stats(target_ids, edges_all, n_hops, verbose=False):
    """Get n-hop neighborhood with optional detailed statistics per hop"""

    current = set(target_ids)
    all_neurons = set(target_ids)

    if verbose:
        print("\n=== N-hop Neighborhood Expansion ===")
        print(f"Starting with {len(target_ids)} core neurons")

    # Track stats per hop
    hop_stats = []

    for hop in range(n_hops):
        next_hop = set()
        edge_count = 0

        for node in current:
            # Find predecessors (neurons that send to current)
            mask = edges_all[1, :] == node
            predecessors = edges_all[0, mask].cpu().numpy()
            next_hop.update(predecessors)
            edge_count += len(predecessors)

        # New neurons added at this hop
        new_neurons = next_hop - all_neurons
        all_neurons.update(next_hop)

        if verbose:
            # Calculate edges to neurons at this hop
            edges_to_current = torch.isin(edges_all[1, :],
                                         torch.tensor(list(all_neurons), device=edges_all.device))
            total_edges = edges_to_current.sum().item()

            # Store stats
            hop_stats.append({
                'hop': hop + 1,
                'new_neurons': len(new_neurons),
                'total_neurons': len(all_neurons),
                'edges_this_hop': edge_count,
                'total_edges': total_edges,
                'expansion_factor': len(all_neurons) / len(target_ids)
            })

            print(f"\nHop {hop + 1}:")
            print(f"  New neurons added: {len(new_neurons):,}")
            print(f"  Total neurons now: {len(all_neurons):,} ({100*len(all_neurons)/13741:.1f}% of network)")
            print(f"  Edges from this hop: {edge_count:,}")
            print(f"  Total edges needed: {total_edges:,} ({100*total_edges/edges_all.shape[1]:.1f}% of all edges)")
            print(f"  Expansion factor: {len(all_neurons)/len(target_ids):.2f}x")
            print(f"  Compute cost estimate: {len(all_neurons) * total_edges / 1e6:.2f}M operations")

        current = next_hop

        if len(current) == 0:
            if verbose:
                print("  -> No more expansion possible")
            break

    if verbose:
        print("\n=== Summary ===")
        print(f"Total neurons: {len(all_neurons):,} / 13,741 ({100*len(all_neurons)/13741:.1f}%)")
        print(f"Total edges: {total_edges:,} / {edges_all.shape[1]:,} ({100*total_edges/edges_all.shape[1]:.1f}%)")
        print(f"Memory estimate: {(len(all_neurons) * 8 + total_edges * 8) / 1e6:.2f} MB")

    return np.array(sorted(all_neurons))

def analyze_type_neighbors(
    type_name: str,
    edges_all: torch.Tensor,        # shape (2, E); row0=pre, row1=post; on some device
    type_list: torch.Tensor,        # shape (N,1) or (N,); integer type indices aligned with node IDs
    n_hops: int = 3,
    direction: str = 'in',          # 'in' | 'out' | 'both'
    verbose: bool = False
):
    """
    Pick one neuron of the given type and expand n hops to collect per-hop type compositions.
    Returns a dict with the target info, per-hop stats, and a short summary.
    """

    device = edges_all.device
    type_vec = type_list.squeeze(-1).long().to(device)  # (N,)

    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14',
        18: 'Mi15', 19: 'Mi2', 20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
        27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
        36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1',
        44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y',
        52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14',
        59: 'TmY15', 60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'
    }

    # --- map type_name -> type_index (simple, case-insensitive) ---
    def _norm(s): return ''.join(ch for ch in s.lower() if ch.isalnum())
    name_to_index = {}
    for k, v in index_to_name.items():
        name_to_index[_norm(v)] = int(k)

    tkey = _norm(type_name)
    if tkey not in name_to_index:
        raise ValueError(f"Unknown type name: {type_name}")

    target_type_idx = name_to_index[tkey]
    target_type_name = index_to_name.get(target_type_idx, f"Type{target_type_idx}")

    # --- pick one neuron of that type (first occurrence) ---
    cand = torch.nonzero(type_vec == target_type_idx, as_tuple=True)[0]
    if cand.numel() == 0:
        return {
            "target_type_idx": target_type_idx,
            "target_type_name": target_type_name,
            "note": "No neuron of this type present.",
            "per_hop": [],
            "summary": {"total_neurons": 0, "total_hops_realized": 0, "direction": direction}
        }
    target_id = int(cand[0].item())

    # --- neighborhood expansion ---
    visited = torch.tensor([target_id], device=device, dtype=torch.long)
    current = visited.clone()
    per_hop = []

    for hop in range(1, n_hops + 1):
        if direction == 'in':
            mask = torch.isin(edges_all[1], current)
            nxt = edges_all[0, mask]
        elif direction == 'out':
            mask = torch.isin(edges_all[0], current)
            nxt = edges_all[1, mask]
        else:  # 'both'
            mask_in = torch.isin(edges_all[1], current)
            mask_out = torch.isin(edges_all[0], current)
            nxt = torch.cat([edges_all[0, mask_in], edges_all[1, mask_out]], dim=0)

        if nxt.numel() == 0:
            break

        nxt = torch.unique(nxt)
        # remove already visited
        new = nxt[~torch.isin(nxt, visited)]
        if new.numel() == 0:
            break

        # types for newly discovered nodes
        new_types = type_vec[new]
        new_ids = new.tolist()
        new_type_idxs = new_types.tolist()
        new_type_names = [index_to_name.get(int(t), f"Type{int(t)}") for t in new_type_idxs]

        # per-hop counts
        cnt = Counter(new_type_names)
        n_new = int(new.numel())
        type_counts = dict(cnt)
        type_perc = {k: v / n_new for k, v in type_counts.items()}

        per_hop.append({
            "hop": hop,
            "new_neuron_ids": new_ids,
            "type_indices": new_type_idxs,
            "type_names": new_type_names,
            "type_counts": type_counts,
            "type_perc": type_perc,
            "n_new": n_new,
        })

        if verbose:
            print(f"hop {hop}: new={n_new}  unique types={len(cnt)}  top={cnt.most_common(3)}")

        # advance
        visited = torch.unique(torch.cat([visited, new], dim=0))
        current = new

    # --- summary (simple) ---
    cumulative = Counter()
    total_new = 0
    for h in per_hop:
        cumulative.update(h["type_counts"])
        total_new += h["n_new"]
    cumulative_perc = {k: (v / total_new if total_new else 0.0) for k, v in cumulative.items()}

    return {
        "target_id": target_id,
        "target_type_idx": target_type_idx,
        "target_type_name": target_type_name,
        "per_hop": per_hop,
        "summary": {
            "total_neurons": int(visited.numel()),
            "total_hops_realized": len(per_hop),
            "direction": direction,
            "cumulative_type_counts": dict(cumulative),
            "cumulative_type_perc": cumulative_perc,
        },
    }


def plot_weight_comparison(w_true, w_modified, output_path, xlabel='true $W$', ylabel='modified $W$', color='white'):
    w_true_np = w_true.detach().cpu().numpy().flatten()
    w_modified_np = w_modified.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(w_true_np, w_modified_np, s=8, alpha=0.5, color=color, edgecolors='none')
    # Fit linear model
    lin_fit, _ = curve_fit(linear_model, w_true_np, w_modified_np)
    slope = lin_fit[0]
    lin_fit[1]
    # R2 calculation
    residuals = w_modified_np - linear_model(w_true_np, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((w_modified_np - np.mean(w_modified_np)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # Plot identity line
    plt.plot([w_true_np.min(), w_true_np.max()], [w_true_np.min(), w_true_np.max()], 'r--', linewidth=2, label='identity')
    # Add text
    plt.text(w_true_np.min(), w_true_np.max(), f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}', fontsize=18, va='top', ha='left')
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return slope, r_squared


def check_dales_law(edges, weights, type_list=None, n_neurons=None, verbose=True, logger=None):
    """
    Check if synaptic weights satisfy Dale's Law.

    Dale's Law: Each neuron releases the same neurotransmitter at all synapses.
    This means all outgoing weights from a neuron should have the same sign.

    Parameters:
    -----------
    edges : torch.Tensor
        Edge index tensor of shape [2, n_edges] where edges[0] are source neurons
    weights : torch.Tensor
        Weight tensor of shape [n_edges] or [n_edges, 1]
    type_list : torch.Tensor, optional
        Neuron type indices of shape [n_neurons] or [n_neurons, 1]
    n_neurons : int, optional
        Total number of neurons (inferred from edges if not provided)
    verbose : bool, default=True
        If True, print detailed statistics
    logger : logging.Logger, optional
        Logger for recording results

    Returns:
    --------
    dict with keys:
        - 'n_excitatory': Number of purely excitatory neurons (all W>0)
        - 'n_inhibitory': Number of purely inhibitory neurons (all W<0)
        - 'n_mixed': Number of mixed neurons (violates Dale's Law)
        - 'n_violations': Number of Dale's Law violations
        - 'violations': List of dicts with violation details
        - 'neuron_signs': Dict mapping neuron_idx to sign (1=excitatory, -1=inhibitory, 0=mixed)
    """
    # Neuron type name mapping (from FlyVis connectome)
    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)',
        5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi15', 14: 'Mi4',
        15: 'Mi9', 16: 'T1', 17: 'T2', 18: 'T2a', 19: 'T3',
        20: 'T4a', 21: 'T4b', 22: 'T4c', 23: 'T4d', 24: 'T5a',
        25: 'T5b', 26: 'T5c', 27: 'T5d', 28: 'Tm1', 29: 'Tm2',
        30: 'Tm3', 31: 'Tm4', 32: 'Tm9', 33: 'TmY10', 34: 'TmY13',
        35: 'TmY14', 36: 'TmY15', 37: 'TmY18', 38: 'TmY3',
        39: 'TmY4', 40: 'TmY5a', 41: 'TmY9'
    }

    # Flatten weights if needed
    if weights.dim() > 1:
        weights = weights.squeeze()

    # Infer n_neurons if not provided
    if n_neurons is None:
        n_neurons = int(edges.max().item()) + 1

    # Check Dale's Law for each neuron
    dale_violations = []
    neuron_signs = {}

    for neuron_idx in range(n_neurons):
        # Find all outgoing edges from this neuron
        outgoing_mask = edges[0, :] == neuron_idx
        outgoing_weights = weights[outgoing_mask]

        if len(outgoing_weights) > 0:
            n_positive = (outgoing_weights > 0).sum().item()
            n_negative = (outgoing_weights < 0).sum().item()
            n_zero = (outgoing_weights == 0).sum().item()

            # Dale's Law: all non-zero weights should have same sign
            if n_positive > 0 and n_negative > 0:
                violation_info = {
                    'neuron': neuron_idx,
                    'n_positive': n_positive,
                    'n_negative': n_negative,
                    'n_zero': n_zero
                }

                # Add type information if available
                if type_list is not None:
                    type_id = type_list[neuron_idx].item()
                    type_name = index_to_name.get(type_id, f'Unknown_{type_id}')
                    violation_info['type_id'] = type_id
                    violation_info['type_name'] = type_name

                dale_violations.append(violation_info)
                neuron_signs[neuron_idx] = 0  # Mixed
            elif n_positive > 0:
                neuron_signs[neuron_idx] = 1  # Excitatory
            elif n_negative > 0:
                neuron_signs[neuron_idx] = -1  # Inhibitory
            else:
                neuron_signs[neuron_idx] = 0  # All zero

    # Compute statistics
    n_excitatory = sum(1 for s in neuron_signs.values() if s == 1)
    n_inhibitory = sum(1 for s in neuron_signs.values() if s == -1)
    n_mixed = sum(1 for s in neuron_signs.values() if s == 0)

    # Print results if verbose
    if verbose:
        print("\n=== Dale's Law Check ===")
        print(f"Total neurons: {n_neurons}")
        print(f"Excitatory neurons (all W>0): {n_excitatory} ({100*n_excitatory/n_neurons:.1f}%)")
        print(f"Inhibitory neurons (all W<0): {n_inhibitory} ({100*n_inhibitory/n_neurons:.1f}%)")
        print(f"Mixed/zero neurons (violates Dale's Law): {n_mixed} ({100*n_mixed/n_neurons:.1f}%)")
        print(f"Dale's Law violations: {len(dale_violations)}")

        if logger:
            logger.info("=== Dale's Law Check ===")
            logger.info(f"Total neurons: {n_neurons}")
            logger.info(f"Excitatory: {n_excitatory} ({100*n_excitatory/n_neurons:.1f}%)")
            logger.info(f"Inhibitory: {n_inhibitory} ({100*n_inhibitory/n_neurons:.1f}%)")
            logger.info(f"Violations: {len(dale_violations)}")

        if len(dale_violations) > 0:
            print("\nFirst 10 violations:")
            for i, v in enumerate(dale_violations[:10]):
                if 'type_name' in v:
                    print(f"  Neuron {v['neuron']} ({v['type_name']}): "
                          f"{v['n_positive']} positive, {v['n_negative']} negative, {v['n_zero']} zero weights")
                    if logger:
                        logger.info(f"  Neuron {v['neuron']} ({v['type_name']}): "
                                    f"{v['n_positive']} positive, {v['n_negative']} negative")
                else:
                    print(f"  Neuron {v['neuron']}: "
                          f"{v['n_positive']} positive, {v['n_negative']} negative, {v['n_zero']} zero weights")

            # Group violations by neuron type if available
            if type_list is not None and any('type_name' in v for v in dale_violations):
                type_violations = Counter([v['type_name'] for v in dale_violations if 'type_name' in v])
                print("\nViolations by neuron type:")
                for type_name, count in sorted(type_violations.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {type_name}: {count} violations")
                    if logger:
                        logger.info(f"  {type_name}: {count} violations")
        else:
            print("✓ Weights perfectly satisfy Dale's Law!")
            if logger:
                logger.info("✓ Weights perfectly satisfy Dale's Law!")

        print("=" * 60 + "\n")

    return {
        'n_excitatory': n_excitatory,
        'n_inhibitory': n_inhibitory,
        'n_mixed': n_mixed,
        'n_violations': len(dale_violations),
        'violations': dale_violations,
        'neuron_signs': neuron_signs
    }


def analyze_data_svd(x_list, output_folder, config=None, max_components=100, logger=None, max_data_size=10_000_000, max_neurons=1024, is_flyvis=False, style=None, save_in_subfolder=True, log_file=None):
    """
    Perform SVD analysis on activity data and external_input/visual stimuli (if present).
    Uses randomized SVD for large datasets for efficiency.
    Subsamples frames if data is too large.

    Args:
        x_list: numpy array of shape (n_frames, n_neurons, n_features)
                features: [id, x, y, u, external_input, ...]
        output_folder: path to save plots
        config: config object (optional, for metadata)
        max_components: maximum number of SVD components to compute
        logger: optional logger (for training)
        max_data_size: maximum data size before subsampling (default 10M elements)
        max_neurons: maximum number of neurons before subsampling (default 1024)
        is_flyvis: if True, use "visual stimuli" label instead of "external input"
        style: matplotlib style to use (e.g., 'dark_background' for dark mode)
        save_in_subfolder: if True, save to results/ subfolder; if False, save directly to output_folder
        log_file: optional file handle to write results

    Returns:
        dict with SVD analysis results
    """
    from sklearn.utils.extmath import randomized_svd

    n_frames, n_neurons, n_features = x_list.shape
    results = {}

    import re
    def log_print(msg):
        if logger:
            logger.info(msg)
        if log_file:
            # strip ANSI color codes for log file
            clean_msg = re.sub(r'\033\[[0-9;]*m', '', msg)
            log_file.write(clean_msg + '\n')

    # subsample neurons if too many
    if n_neurons > max_neurons:
        neuron_subsample = int(np.ceil(n_neurons / max_neurons))
        neuron_indices = np.arange(0, n_neurons, neuron_subsample)
        x_list = x_list[:, neuron_indices, :]
        n_neurons_sampled = len(neuron_indices)
        log_print(f"subsampling neurons: {n_neurons} -> {n_neurons_sampled} (every {neuron_subsample}th)")
        n_neurons = n_neurons_sampled

    # subsample frames if data is too large
    data_size = n_frames * n_neurons
    if data_size > max_data_size:
        subsample_factor = int(np.ceil(data_size / max_data_size))
        frame_indices = np.arange(0, n_frames, subsample_factor)
        x_list_sampled = x_list[frame_indices]
        n_frames_sampled = len(frame_indices)
        log_print(f"subsampling frames: {n_frames} -> {n_frames_sampled} (every {subsample_factor}th)")
        data_size_sampled = n_frames_sampled * n_neurons
    else:
        x_list_sampled = x_list
        n_frames_sampled = n_frames
        data_size_sampled = data_size
        subsample_factor = 1

    # decide whether to use randomized SVD
    use_randomized = data_size_sampled > 1e6  # use randomized for > 1M elements

    # store data size info for later printing with results
    if subsample_factor > 1:
        data_info = f"using {n_frames_sampled:,} of {n_frames:,} frames ({n_neurons:,} neurons)"
    else:
        data_info = f"using full data ({n_frames:,} frames, {n_neurons:,} neurons)"

    # save current style context and apply new style if provided
    # We use context manager approach to avoid resetting global style
    if style:
        plt.style.use(style)

    # main color based on style
    mc = 'w' if style == 'dark_background' else 'k'
    bg_color = 'k' if style == 'dark_background' else 'w'

    # font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12

    # prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=bg_color)
    for ax in axes.flat:
        ax.set_facecolor(bg_color)

    # 1. analyze activity (u) - column 3
    activity = x_list_sampled[:, :, 3]  # shape: (n_frames_sampled, n_neurons)
    log_print("--- activity ---")
    log_print(f"  shape: {activity.shape}")
    log_print(f"  range: [{activity.min():.3f}, {activity.max():.3f}]")

    k = min(max_components, min(n_frames_sampled, n_neurons) - 1)

    if use_randomized:
        U_act, S_act, Vt_act = randomized_svd(activity, n_components=k, random_state=42)
    else:
        U_act, S_act, Vt_act = np.linalg.svd(activity, full_matrices=False)
        S_act = S_act[:k]

    # compute cumulative variance
    cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
    rank_90_act = np.searchsorted(cumvar_act, 0.90) + 1
    rank_99_act = np.searchsorted(cumvar_act, 0.99) + 1

    log_print(f"  effective rank (90% var): {rank_90_act}")
    log_print(f"  effective rank (99% var): \033[92m{rank_99_act}\033[0m")

    # compression ratio
    if rank_99_act < k:
        compression_act = (n_frames * n_neurons) / (rank_99_act * (n_frames + n_neurons))
        log_print(f"  compression (rank-{rank_99_act}): {compression_act:.1f}x")
    else:
        log_print("  compression: need more components to reach 99% variance")

    results['activity'] = {
        'singular_values': S_act,
        'cumulative_variance': cumvar_act,
        'rank_90': rank_90_act,
        'rank_99': rank_99_act,
    }

    # plot activity SVD
    ax = axes[0, 0]
    ax.semilogy(S_act, color=mc, lw=1.5)
    ax.set_xlabel('component', fontsize=LABEL_SIZE)
    ax.set_ylabel('singular value', fontsize=LABEL_SIZE)
    ax.set_title('activity: singular values', fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cumvar_act, color=mc, lw=1.5)
    ax.axhline(0.90, color='orange', ls='--', label='90%')
    ax.axhline(0.99, color='green', ls='--', label='99%')
    ax.axvline(rank_90_act, color='orange', ls=':', alpha=0.7)
    ax.axvline(rank_99_act, color='green', ls=':', alpha=0.7)
    ax.set_xlabel('component', fontsize=LABEL_SIZE)
    ax.set_ylabel('cumulative variance', fontsize=LABEL_SIZE)
    ax.set_title(f'activity: rank(90%)={rank_90_act}, rank(99%)={rank_99_act}', fontsize=TITLE_SIZE)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # 2. Analyze external_input / visual stimuli (if present and non-zero) - column 4
    # Determine label based on is_flyvis parameter
    n_input_neurons = None
    if is_flyvis:
        n_input_neurons = getattr(config.simulation, 'n_input_neurons', None) if config else None
    input_label = "visual stimuli" if is_flyvis else "external input"

    if n_features > 4:
        # for visual stimuli, only analyze input neurons (first n_input_neurons)
        if is_flyvis and n_input_neurons is not None and n_input_neurons < n_neurons:
            external_input = x_list_sampled[:, :n_input_neurons, 4]  # shape: (n_frames_sampled, n_input_neurons)
            log_print(f"--- {input_label} (first {n_input_neurons} input neurons) ---")
        else:
            external_input = x_list_sampled[:, :, 4]  # shape: (n_frames_sampled, n_neurons)

        # check if external_input has actual signal
        ext_range = external_input.max() - external_input.min()
        if ext_range > 1e-6:
            if not (is_flyvis and n_input_neurons is not None):
                log_print(f"--- {input_label} ---")
            log_print(f"  shape: {external_input.shape}")
            log_print(f"  range: [{external_input.min():.3f}, {external_input.max():.3f}]")

            if use_randomized:
                U_ext, S_ext, Vt_ext = randomized_svd(external_input, n_components=k, random_state=42)
            else:
                U_ext, S_ext, Vt_ext = np.linalg.svd(external_input, full_matrices=False)
                S_ext = S_ext[:k]

            cumvar_ext = np.cumsum(S_ext**2) / np.sum(S_ext**2)
            rank_90_ext = np.searchsorted(cumvar_ext, 0.90) + 1
            rank_99_ext = np.searchsorted(cumvar_ext, 0.99) + 1

            log_print(f"  effective rank (90% var): {rank_90_ext}")
            log_print(f"  effective rank (99% var): \033[92m{rank_99_ext}\033[0m")

            if rank_99_ext < k:
                compression_ext = (n_frames * n_neurons) / (rank_99_ext * (n_frames + n_neurons))
                log_print(f"  compression (rank-{rank_99_ext}): {compression_ext:.1f}x")
            else:
                log_print("  compression: need more components to reach 99% variance")

            results_key = 'visual_stimuli' if is_flyvis else 'external_input'
            results[results_key] = {
                'singular_values': S_ext,
                'cumulative_variance': cumvar_ext,
                'rank_90': rank_90_ext,
                'rank_99': rank_99_ext,
            }

            # plot external_input / visual stimuli SVD
            ax = axes[1, 0]
            ax.semilogy(S_ext, color=mc, lw=1.5)
            ax.set_xlabel('component', fontsize=LABEL_SIZE)
            ax.set_ylabel('singular value', fontsize=LABEL_SIZE)
            ax.set_title(f'{input_label}: singular values', fontsize=TITLE_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(cumvar_ext, color=mc, lw=1.5)
            ax.axhline(0.90, color='orange', ls='--', label='90%')
            ax.axhline(0.99, color='green', ls='--', label='99%')
            ax.axvline(rank_90_ext, color='orange', ls=':', alpha=0.7)
            ax.axvline(rank_99_ext, color='green', ls=':', alpha=0.7)
            ax.set_xlabel('component', fontsize=LABEL_SIZE)
            ax.set_ylabel('cumulative variance', fontsize=LABEL_SIZE)
            ax.set_title(f'{input_label}: rank(90%)={rank_90_ext}, rank(99%)={rank_99_ext}', fontsize=TITLE_SIZE)
            ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
        else:
            log_print(f"--- {input_label} ---")
            log_print("  no external input found (range < 1e-6)")
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
            results_key = 'visual_stimuli' if is_flyvis else 'external_input'
            results[results_key] = None
    else:
        log_print(f"--- {input_label} ---")
        log_print("  not present in data")
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
        results_key = 'visual_stimuli' if is_flyvis else 'external_input'
        results[results_key] = None

    plt.tight_layout()

    # save plot
    if save_in_subfolder:
        save_folder = os.path.join(output_folder, 'results')
        os.makedirs(save_folder, exist_ok=True)
    else:
        save_folder = output_folder
    save_path = os.path.join(save_folder, 'svd_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close()

    # print SVD results: data info (white) + rank results (green)
    ext_key = 'visual_stimuli' if is_flyvis else 'external_input'
    if results.get(ext_key):
        print(f"{data_info}, \033[92mactivity rank(99%)={results['activity']['rank_99']}, {ext_key} rank(99%)={results[ext_key]['rank_99']}\033[0m")
    else:
        print(f"{data_info}, \033[92mactivity rank(99%)={results['activity']['rank_99']}\033[0m")

    return results


def save_exploration_artifacts(root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                               iter_in_block=1, block_number=1):
    """
    Save exploration artifacts for Claude analysis.

    Args:
        root_dir: Root directory of the project
        exploration_dir: Base directory for exploration artifacts
        config: Configuration object
        config_file_: Config file name (without extension)
        pre_folder: Prefix folder for config
        iteration: Current iteration number
        iter_in_block: Iteration number within current block (1-indexed)
        block_number: Current block number (1-indexed)

    Returns:
        dict with paths to saved directories
    """
    import glob
    import shutil
    import matplotlib.image as mpimg

    config_save_dir = f"{exploration_dir}/config"
    scatter_save_dir = f"{exploration_dir}/connectivity_scatter"
    matrix_save_dir = f"{exploration_dir}/connectivity_matrix"
    activity_save_dir = f"{exploration_dir}/activity"
    mlp_save_dir = f"{exploration_dir}/mlp"
    tree_save_dir = f"{exploration_dir}/exploration_tree"
    protocol_save_dir = f"{exploration_dir}/protocol"

    # create directories at start of experiment
    if iteration == 1:
        # clear and recreate exploration folder
        if os.path.exists(exploration_dir):
            shutil.rmtree(exploration_dir)
        os.makedirs(config_save_dir, exist_ok=True)
        os.makedirs(scatter_save_dir, exist_ok=True)
        os.makedirs(matrix_save_dir, exist_ok=True)
        os.makedirs(activity_save_dir, exist_ok=True)
        os.makedirs(mlp_save_dir, exist_ok=True)
        os.makedirs(tree_save_dir, exist_ok=True)
        os.makedirs(protocol_save_dir, exist_ok=True)

    # determine if this is first iteration of a block
    is_block_start = (iter_in_block == 1)

    # save config file only at first iteration of each block
    if is_block_start:
        src_config = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
        dst_config = f"{config_save_dir}/block_{block_number:03d}.yaml"
        if os.path.exists(src_config):
            shutil.copy2(src_config, dst_config)

    # save connectivity scatterplot (most recent comparison_*.tif from matrix folder)
    matrix_dir = f"{root_dir}/log/{pre_folder}{config_file_}/tmp_training/matrix"
    scatter_files = glob.glob(f"{matrix_dir}/comparison_*.tif")
    if scatter_files:
        # get most recent file
        latest_scatter = max(scatter_files, key=os.path.getmtime)
        dst_scatter = f"{scatter_save_dir}/iter_{iteration:03d}.tif"
        shutil.copy2(latest_scatter, dst_scatter)

    # save connectivity matrix heatmap only at first iteration of each block
    data_folder = f"{root_dir}/graphs_data/{config.dataset}"
    if is_block_start:
        src_matrix = f"{data_folder}/connectivity_matrix.png"
        dst_matrix = f"{matrix_save_dir}/block_{block_number:03d}.png"
        if os.path.exists(src_matrix):
            shutil.copy2(src_matrix, dst_matrix)

    # save activity plot only at first iteration of each block
    activity_path = f"{data_folder}/activity.png"
    if is_block_start:
        dst_activity = f"{activity_save_dir}/block_{block_number:03d}.png"
        if os.path.exists(activity_path):
            shutil.copy2(activity_path, dst_activity)

    # save combined MLP plot (MLP0 + MLP1 side by side) using PNG files from results
    results_dir = f"{root_dir}/log/{pre_folder}{config_file_}/results"
    src_mlp0 = f"{results_dir}/MLP0.png"
    src_mlp1 = f"{results_dir}/MLP1_corrected.png"
    if os.path.exists(src_mlp0) and os.path.exists(src_mlp1):
        try:
            # Load PNG images
            img0 = mpimg.imread(src_mlp0)
            img1 = mpimg.imread(src_mlp1)

            # Create combined figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(img0)
            axes[0].set_title('MLP0 (φ)', fontsize=12)
            axes[0].axis('off')
            axes[1].imshow(img1)
            axes[1].set_title('MLP1 (edge)', fontsize=12)
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(f"{mlp_save_dir}/iter_{iteration:03d}_MLP.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"\033[93mwarning: could not combine MLP plots: {e}\033[0m")

    return {
        'config_save_dir': config_save_dir,
        'scatter_save_dir': scatter_save_dir,
        'matrix_save_dir': matrix_save_dir,
        'activity_save_dir': activity_save_dir,
        'mlp_save_dir': mlp_save_dir,
        'tree_save_dir': tree_save_dir,
        'protocol_save_dir': protocol_save_dir,
        'activity_path': activity_path
    }


class LossRegularizer:
    """
    Handles all regularization terms, coefficient annealing, and history tracking.

    Usage:
        regularizer = LossRegularizer(train_config, model_config, activity_column=6,
                                       plot_frequency=100, n_neurons=1000, trainer_type='signal')

        for epoch in range(n_epochs):
            regularizer.set_epoch(epoch)

            for N in range(Niter):
                regularizer.reset_iteration()

                pred, in_features, msg = model(batch, data_id=data_id, return_all=True)

                regul_loss = regularizer.compute(model, x, in_features, ids, ids_batch, edges, device)
                loss = pred_loss + regul_loss
    """

    # Components tracked in history
    COMPONENTS = [
        'W_L1', 'W_L2', 'W_sign',
        'edge_diff', 'edge_norm', 'edge_weight', 'phi_weight',
        'phi_zero', 'update_diff', 'update_msg_diff', 'update_u_diff', 'update_msg_sign',
        'missing_activity', 'model_a', 'model_b', 'modulation'
    ]

    def __init__(self, train_config, model_config, activity_column: int,
                 plot_frequency: int, n_neurons: int, trainer_type: str = 'signal'):
        """
        Args:
            train_config: TrainingConfig with coeff_* values
            model_config: GraphModelConfig with model settings
            activity_column: Column index for activity (6 for signal, 3 for flyvis)
            plot_frequency: How often to record to history
            n_neurons: Number of neurons for normalization
            trainer_type: 'signal' or 'flyvis' - controls annealing behavior
        """
        self.train_config = train_config
        self.model_config = model_config
        self.activity_column = activity_column
        self.plot_frequency = plot_frequency
        self.n_neurons = n_neurons
        self.trainer_type = trainer_type

        # Current epoch (for annealing)
        self.epoch = 0

        # Iteration counter
        self.iter_count = 0

        # Per-iteration accumulator
        self._iter_total = 0.0
        self._iter_tracker = {}

        # History for plotting
        self._history = {comp: [] for comp in self.COMPONENTS}
        self._history['regul_total'] = []

        # Cache coefficients
        self._coeffs = {}
        self._update_coeffs()

    def _update_coeffs(self):
        """Recompute coefficients based on current epoch (annealing for flyvis only)."""
        tc = self.train_config
        epoch = self.epoch

        # Two-phase training support (like ParticleGraph data_train_synaptic2)
        n_epochs_init = getattr(tc, 'n_epochs_init', 0)
        first_coeff_L1 = getattr(tc, 'first_coeff_L1', tc.coeff_W_L1)

        if self.trainer_type == 'flyvis':
            # Flyvis: annealed coefficients
            self._coeffs['W_L1'] = tc.coeff_W_L1 * (1 - np.exp(-tc.coeff_W_L1_rate * epoch))
            self._coeffs['edge_weight_L1'] = tc.coeff_edge_weight_L1 * (1 - np.exp(-tc.coeff_edge_weight_L1_rate ** epoch))
            self._coeffs['phi_weight_L1'] = tc.coeff_phi_weight_L1 * (1 - np.exp(-tc.coeff_phi_weight_L1_rate * epoch))
        else:
            # Signal: two-phase training if n_epochs_init > 0
            if n_epochs_init > 0 and epoch < n_epochs_init:
                # Phase 1: use first_coeff_L1 (typically 0 or small)
                self._coeffs['W_L1'] = first_coeff_L1
            else:
                # Phase 2: use coeff_W_L1 (target L1)
                self._coeffs['W_L1'] = tc.coeff_W_L1
            self._coeffs['edge_weight_L1'] = tc.coeff_edge_weight_L1
            self._coeffs['phi_weight_L1'] = tc.coeff_phi_weight_L1

        # Non-annealed coefficients (same for both)
        self._coeffs['W_L2'] = tc.coeff_W_L2
        self._coeffs['W_sign'] = tc.coeff_W_sign
        # Two-phase: edge_diff is active in phase 1, disabled in phase 2
        if n_epochs_init > 0 and epoch >= n_epochs_init:
            self._coeffs['edge_diff'] = 0  # Phase 2: no monotonicity constraint
        else:
            self._coeffs['edge_diff'] = tc.coeff_edge_diff
        self._coeffs['edge_norm'] = tc.coeff_edge_norm
        self._coeffs['edge_weight_L2'] = tc.coeff_edge_weight_L2
        self._coeffs['phi_weight_L2'] = tc.coeff_phi_weight_L2
        self._coeffs['phi_zero'] = tc.coeff_lin_phi_zero
        self._coeffs['update_diff'] = tc.coeff_update_diff
        self._coeffs['update_msg_diff'] = tc.coeff_update_msg_diff
        self._coeffs['update_u_diff'] = tc.coeff_update_u_diff
        self._coeffs['update_msg_sign'] = tc.coeff_update_msg_sign
        self._coeffs['missing_activity'] = tc.coeff_missing_activity
        self._coeffs['model_a'] = tc.coeff_model_a
        self._coeffs['model_b'] = tc.coeff_model_b
        self._coeffs['modulation'] = tc.coeff_lin_modulation

    def set_epoch(self, epoch: int, plot_frequency: int = None):
        """Set current epoch and update annealed coefficients."""
        self.epoch = epoch
        self._update_coeffs()
        if plot_frequency is not None:
            self.plot_frequency = plot_frequency
        # Reset iteration counter at epoch start
        self.iter_count = 0

    def reset_iteration(self):
        """Reset per-iteration accumulator."""
        self.iter_count += 1
        self._iter_total = 0.0
        self._iter_tracker = {comp: 0.0 for comp in self.COMPONENTS}
        # Flag to ensure W_L1 is only applied once per iteration (not per batch item)
        self._W_L1_applied_this_iter = False

    def should_record(self) -> bool:
        """Check if we should record to history this iteration."""
        return (self.iter_count % self.plot_frequency == 0) or (self.iter_count == 1)

    def needs_update_regul(self) -> bool:
        """Check if update regularization is needed (update_diff, update_msg_diff, update_u_diff, or update_msg_sign)."""
        return (self._coeffs['update_diff'] > 0 or
                self._coeffs['update_msg_diff'] > 0 or
                self._coeffs['update_u_diff'] > 0 or
                self._coeffs['update_msg_sign'] > 0)

    def _add(self, name: str, term):
        """Internal: accumulate a regularization term."""
        if term is None:
            return
        val = term.item() if hasattr(term, 'item') else float(term)
        self._iter_total += val
        if name in self._iter_tracker:
            self._iter_tracker[name] += val

    def compute(self, model, x, in_features, ids, ids_batch, edges, device,
                xnorm=1.0, index_weight=None):
        """
        Compute all regularization terms internally.

        Args:
            model: The neural network model
            x: Input tensor
            in_features: Features for lin_phi (from model forward pass, can be None)
            ids: Sample indices for regularization
            ids_batch: Batch indices
            edges: Edge tensor
            device: Torch device
            xnorm: Normalization value
            index_weight: Index for W_sign computation (signal only)

        Returns:
            Total regularization loss tensor
        """
        tc = self.train_config
        mc = self.model_config
        n_neurons = self.n_neurons
        total_regul = torch.tensor(0.0, device=device)

        # Get model W (handle multi-run case not working here)
        # For low_rank_factorization, compute W from WL @ WR to allow gradient flow
        
        # --- W regularization ---
        low_rank = getattr(model, 'low_rank_factorization', False)
        if low_rank and hasattr(model, 'WL') and hasattr(model, 'WR'):
            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = (model.WL.norm(1) + model.WR) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True
        else:
            # W_L1: Apply only once per iteration (not per batch item)
            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(1) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True

            if self._coeffs['W_L2'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(2) * self._coeffs['W_L2']
                total_regul = total_regul + regul_term
                self._add('W_L2', regul_term)

        # --- Edge/Phi weight regularization ---
        if (self._coeffs['edge_weight_L1'] + self._coeffs['edge_weight_L2']) > 0:
            for param in model.lin_edge.parameters():
                regul_term = param.norm(1) * self._coeffs['edge_weight_L1'] + param.norm(2) * self._coeffs['edge_weight_L2']
                total_regul = total_regul + regul_term
                self._add('edge_weight', regul_term)

        if (self._coeffs['phi_weight_L1'] + self._coeffs['phi_weight_L2']) > 0:
            for param in model.lin_phi.parameters():
                regul_term = param.norm(1) * self._coeffs['phi_weight_L1'] + param.norm(2) * self._coeffs['phi_weight_L2']
                total_regul = total_regul + regul_term
                self._add('phi_weight', regul_term)

        # --- phi_zero regularization ---
        if self._coeffs['phi_zero'] > 0:
            in_features_phi = get_in_features_update(rr=None, model=model, device=device)
            func_phi = model.lin_phi(in_features_phi[ids].float())
            regul_term = func_phi.norm(2) * self._coeffs['phi_zero']
            total_regul = total_regul + regul_term
            self._add('phi_zero', regul_term)

        # --- Edge diff/norm regularization ---
        if (self._coeffs['edge_diff'] > 0) | (self._coeffs['edge_norm'] > 0):
            in_features_edge, in_features_edge_next = get_in_features_lin_edge(x, model, mc, xnorm, n_neurons, device)

            if self._coeffs['edge_diff'] > 0:
                if mc.lin_edge_positive:
                    msg0 = model.lin_edge(in_features_edge[ids].clone().detach()) ** 2
                    msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach()) ** 2
                else:
                    msg0 = model.lin_edge(in_features_edge[ids].clone().detach())
                    msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach())
                regul_term = torch.relu(msg0 - msg1).norm(2) * self._coeffs['edge_diff']
                total_regul = total_regul + regul_term
                self._add('edge_diff', regul_term)

            if self._coeffs['edge_norm'] > 0:
                in_features_edge_norm = in_features_edge.clone()
                in_features_edge_norm[:, 0] = 2 * xnorm
                if mc.lin_edge_positive:
                    msg_norm = model.lin_edge(in_features_edge_norm[ids].clone().detach()) ** 2
                else:
                    msg_norm = model.lin_edge(in_features_edge_norm[ids].clone().detach())
                # Different normalization target for signal vs flyvis
                if self.trainer_type == 'signal':
                    regul_term = (msg_norm - 1).norm(2) * self._coeffs['edge_norm']
                else:  # flyvis
                    regul_term = (msg_norm - 2 * xnorm).norm(2) * self._coeffs['edge_norm']
                total_regul = total_regul + regul_term
                self._add('edge_norm', regul_term)

        # --- W_sign (Dale's Law) regularization ---
        if self._coeffs['W_sign'] > 0 and self.epoch > 0:
            W_sign_temp = getattr(tc, 'W_sign_temperature', 10.0)

            if self.trainer_type == 'signal' and index_weight is not None:
                # Signal version: uses index_weight
                if self.iter_count % 4 == 0:
                    W_sign = torch.tanh(5 * model_W)
                    loss_contribs = []
                    for i in range(n_neurons):
                        indices = index_weight[int(i)]
                        if indices.numel() > 0:
                            values = W_sign[indices, i]
                            std = torch.std(values, unbiased=False)
                            loss_contribs.append(std)
                    if loss_contribs:
                        regul_term = torch.stack(loss_contribs).norm(2) * self._coeffs['W_sign']
                        total_regul = total_regul + regul_term
                        self._add('W_sign', regul_term)
            else:
                # Flyvis version: uses scatter_add
                weights = model_W.squeeze() if model_W is not None else model.W.squeeze()
                source_neurons = edges[0]

                n_pos = torch.zeros(n_neurons, device=device)
                n_neg = torch.zeros(n_neurons, device=device)
                n_total = torch.zeros(n_neurons, device=device)

                pos_mask = torch.sigmoid(W_sign_temp * weights)
                neg_mask = torch.sigmoid(-W_sign_temp * weights)

                n_pos.scatter_add_(0, source_neurons, pos_mask)
                n_neg.scatter_add_(0, source_neurons, neg_mask)
                n_total.scatter_add_(0, source_neurons, torch.ones_like(weights))

                violation = torch.where(n_total > 0,
                                        (n_pos / n_total) * (n_neg / n_total),
                                        torch.zeros_like(n_total))
                regul_term = violation.sum() * self._coeffs['W_sign']
                total_regul = total_regul + regul_term
                self._add('W_sign', regul_term)

        # Note: Update function regularizations (update_msg_diff, update_u_diff, update_msg_sign)
        # are handled by compute_update_regul() which should be called after the model forward pass.
        # Call finalize_iteration() after all regularizations are computed to record to history.

        return total_regul

    def _record_to_history(self):
        """Append current iteration values to history."""
        n = self.n_neurons
        self._history['regul_total'].append(self._iter_total / n)
        for comp in self.COMPONENTS:
            self._history[comp].append(self._iter_tracker.get(comp, 0) / n)

    def compute_update_regul(self, model, in_features, ids_batch, device,
                              x=None, xnorm=None, ids=None):
        """
        Compute update function regularizations (update_diff, update_msg_diff, update_u_diff, update_msg_sign).

        This method should be called after the model forward pass when in_features is available.

        Args:
            model: The neural network model
            in_features: Features from model forward pass
            ids_batch: Batch indices
            device: Torch device
            x: Input tensor (required for update_diff with 'generic' update_type)
            xnorm: Normalization value (required for update_diff)
            ids: Sample indices (required for update_diff)

        Returns:
            Total update regularization loss tensor
        """
        mc = self.model_config
        embedding_dim = mc.embedding_dim
        n_neurons = self.n_neurons
        total_regul = torch.tensor(0.0, device=device)

        # update_diff: for 'generic' update_type only
        if (self._coeffs['update_diff'] > 0) and (model.update_type == 'generic') and (x is not None):
            in_features_edge, in_features_edge_next = get_in_features_lin_edge(
                x, model, mc, xnorm, n_neurons, device)
            if mc.lin_edge_positive:
                msg0 = model.lin_edge(in_features_edge[ids].clone().detach()) ** 2
                msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach()) ** 2
            else:
                msg0 = model.lin_edge(in_features_edge[ids].clone().detach())
                msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach())
            in_feature_update = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                           model.a[:n_neurons], msg0,
                                           torch.ones((n_neurons, 1), device=device)), dim=1)
            in_feature_update = in_feature_update[ids]
            in_feature_update_next = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                                model.a[:n_neurons], msg1,
                                                torch.ones((n_neurons, 1), device=device)), dim=1)
            in_feature_update_next = in_feature_update_next[ids]
            regul_term = torch.relu(model.lin_phi(in_feature_update) - model.lin_phi(in_feature_update_next)).norm(2) * self._coeffs['update_diff']
            total_regul = total_regul + regul_term
            self._add('update_diff', regul_term)

        if in_features is None:
            return total_regul

        if self._coeffs['update_msg_diff'] > 0:
            pred_msg = model.lin_phi(in_features.clone().detach())
            in_features_msg_next = in_features.clone().detach()
            in_features_msg_next[:, embedding_dim + 1] = in_features_msg_next[:, embedding_dim + 1] * 1.05
            pred_msg_next = model.lin_phi(in_features_msg_next)
            regul_term = torch.relu(pred_msg[ids_batch] - pred_msg_next[ids_batch]).norm(2) * self._coeffs['update_msg_diff']
            total_regul = total_regul + regul_term
            self._add('update_msg_diff', regul_term)

        if self._coeffs['update_u_diff'] > 0:
            pred_u = model.lin_phi(in_features.clone().detach())
            in_features_u_next = in_features.clone().detach()
            in_features_u_next[:, 0] = in_features_u_next[:, 0] * 1.05
            pred_u_next = model.lin_phi(in_features_u_next)
            regul_term = torch.relu(pred_u_next[ids_batch] - pred_u[ids_batch]).norm(2) * self._coeffs['update_u_diff']
            total_regul = total_regul + regul_term
            self._add('update_u_diff', regul_term)

        if self._coeffs['update_msg_sign'] > 0:
            in_features_modified = in_features.clone().detach()
            in_features_modified[:, 0] = 0
            pred_msg = model.lin_phi(in_features_modified)
            msg_col = in_features[:, embedding_dim + 1].clone().detach()
            regul_term = (torch.tanh(pred_msg / 0.1) - torch.tanh(msg_col.unsqueeze(-1) / 0.1)).norm(2) * self._coeffs['update_msg_sign']
            total_regul = total_regul + regul_term
            self._add('update_msg_sign', regul_term)

        return total_regul

    def finalize_iteration(self):
        """
        Finalize the current iteration by recording to history if appropriate.

        This should be called after all regularization computations (compute + compute_update_regul).
        """
        if self.should_record():
            self._record_to_history()

    def get_iteration_total(self) -> float:
        """Get total regularization for current iteration."""
        return self._iter_total

    def get_history(self) -> dict:
        """Get history dictionary for plotting."""
        return self._history
