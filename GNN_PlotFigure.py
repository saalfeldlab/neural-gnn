import os
import umap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FFMpegWriter
from torch_geometric.loader import DataLoader
import torch_geometric.data as data
import imageio.v2 as imageio
from matplotlib import rc
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
import scipy.sparse
from shutil import copyfile
from collections import defaultdict
import scipy
import logging
import re
import matplotlib
import pandas as pd

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# from data_loaders import *


from NeuralGraph.fitting_models import linear_model
from NeuralGraph.sparsify import EmbeddingCluster, sparsify_cluster, clustering_gmm
from NeuralGraph.models.utils import (
    choose_training_model,
    get_in_features,
    get_in_features_update,
    get_index_particles,
    analyze_odor_responses_by_neuron,
    plot_odor_heatmaps,
    analyze_data_svd,
)
from NeuralGraph.models.plot_utils import (
    analyze_mlp_edge_lines,
    analyze_mlp_edge_lines_weighted_with_max,
    analyze_mlp_phi_synaptic,
    find_top_responding_pairs,
    run_neural_architecture_pipeline,
)
from NeuralGraph.utils import (
    to_numpy,
    CustomColorMap,
    sort_key,
    fig_init,
    get_equidistant_points,
    map_matrix,
    create_log_dir,
    find_suffix_pairs_with_index,
    add_pre_folder
)
from NeuralGraph.models.Siren_Network import Siren, Siren_Network
from NeuralGraph.models.graph_trainer import data_test
from NeuralGraph.generators.utils import choose_model
from NeuralGraph.config import NeuralGraphConfig

from NeuralGraph.models.Ising_analysis import analyze_ising_model

from scipy import stats
from io import StringIO
import sys
import warnings
import seaborn as sns
import glob
import numpy as np
import pickle
import json
from tqdm import tqdm, trange
import time
from sklearn import metrics
from tifffile import imread

import matplotlib.ticker as ticker
import shutil

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

# Optional dependency
# try:
#     from pysr import PySRRegressor
# except (ImportError, subprocess.CalledProcessError):
#     PySRRegressor = None


def get_model_W(model):
    """Get the weight matrix from a model, handling low-rank factorization."""
    if hasattr(model, 'W'):
        return model.W
    elif hasattr(model, 'WL') and hasattr(model, 'WR'):
        return model.WL @ model.WR
    else:
        raise AttributeError("Model has neither 'W' nor 'WL'/'WR' attributes")


def get_training_files(log_dir, n_runs):
    files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_*.pt")
    if len(files) == 0:
        return [], np.array([])
    files.sort(key=sort_key)

    # Find the first file with positive sort_key
    file_id = 0
    while file_id < len(files) and sort_key(files[file_id]) <= 0:
        file_id += 1

    # If all files have non-positive sort_key, use all files
    if file_id >= len(files):
        file_id = 0

    files = files[file_id:]

    # Filter based on the Y value (number after "graphs")
    files_with_0 = [file for file in files if int(file.split('_')[-2]) == 0]
    files_without_0 = [file for file in files if int(file.split('_')[-2]) != 0]

    # Generate 50 evenly spaced indices for each list

    # indices_with_0 = np.linspace(0, len(files_with_0) - 1, dtype=int)
    indices_with_0 = np.arange(0, len(files_with_0) - 1, dtype=int)
    indices_without_0 = np.linspace(0, len(files_without_0) - 1, 50, dtype=int)

    # Select the files using the generated indices
    selected_files_with_0 = [files_with_0[i] for i in indices_with_0]
    if len(files_without_0) > 0:
        selected_files_without_0 = [files_without_0[i] for i in indices_without_0]
        selected_files = selected_files_with_0 + selected_files_without_0
    else:
        selected_files = selected_files_with_0

    return selected_files, np.arange(0, len(selected_files), 1)

    # len_files = len(files)
    # print(len_files, len_files//10, len_files//500, len_files//10, len_files, len_files//50)
    # file_id_list0 = np.arange(0, len_files//10, len_files//500)
    # file_id_list1 = np.arange(len_files//10, len_files, len_files//50)
    # file_id_list = np.concatenate((file_id_list0, file_id_list1))
    # # file_id_list = np.arange(0, len(files), (len(files) / 100)).astype(int)
    # return files, file_id_list


def load_training_data(dataset_name, n_runs, log_dir, device):
    x_list = []
    y_list = []
    print('load data ...')
    time.sleep(0.5)
    for run in trange(n_runs, ncols=90):
        # check if path exists
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
            y = torch.tensor(y, dtype=torch.float32, device=device)

        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device).squeeze()
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device).squeeze()
    print("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = []
    y = []

    return x_list, y_list, vnorm, ynorm


def plot_confusion_matrix(index, true_labels, new_labels, n_neuron_types, epoch, it, fig, ax, style):
    # print(f'plot confusion matrix epoch:{epoch} it: {it}')
    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    confusion_matrix = metrics.confusion_matrix(true_labels, new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if n_neuron_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False, colorbar=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d', colorbar=False)
    accuracy = metrics.accuracy_score(true_labels, new_labels)
    plt.title(f'accuracy: {np.round(accuracy, 2)}', fontsize=12)
    # print(f'accuracy: {np.round(accuracy,3)}')
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlabel(r'Predicted label', fontsize=12)
    plt.ylabel(r'True label', fontsize=12)

    return accuracy


# =============================================================================
# Movie generation functions for plot_signal
# =============================================================================

def determine_plot_limits_signal(config, log_dir, n_runs, device, n_neurons, type_list, cmap, connectivity, xnorm, ynorm, n_samples=5, true_model=None, n_neuron_types=1):
    """
    Sample a few models to determine stable xlim/ylim for movies.
    Returns dict with limits for each plot type.

    Args:
        true_model: Ground truth model for computing true MLP limits (optional)
        n_neuron_types: Number of neuron types for true curve computation
    """
    files, file_id_list = get_training_files(log_dir, n_runs)
    if len(file_id_list) == 0:
        return None

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    # Sample indices evenly distributed
    sample_indices = np.linspace(0, len(file_id_list) - 1, min(n_samples, len(file_id_list)), dtype=int)

    # Collect ranges
    weight_min, weight_max = [], []
    embedding_min, embedding_max = [], []
    lin_edge_min, lin_edge_max = [], []
    lin_phi_min, lin_phi_max = [], []

    # Compute true model limits first if available
    rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 100).to(device)
    if true_model is not None:
        with torch.no_grad():
            for n in range(n_neuron_types):
                # True lin_edge (phi) limits
                true_func = true_model.func(rr, n, 'phi')
                lin_edge_min.append(to_numpy(true_func).min())
                lin_edge_max.append(to_numpy(true_func).max())
                # True lin_phi (update) limits
                true_func = true_model.func(rr, n, 'update')
                lin_phi_min.append(to_numpy(true_func).min())
                lin_phi_max.append(to_numpy(true_func).max())

    with torch.no_grad():
        for idx in sample_indices:
            file_id = file_id_list[idx]
            epoch = files[file_id].split('graphs')[1][1:-3]
            net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            # Weight comparison limits
            gt_weight = to_numpy(connectivity).flatten()
            pred_weight = to_numpy(get_model_W(model)).flatten()
            weight_min.extend([gt_weight.min(), pred_weight.min()])
            weight_max.extend([gt_weight.max(), pred_weight.max()])

            # Embedding limits
            amax = torch.max(model.a, dim=0).values
            amin = torch.min(model.a, dim=0).values
            model_a = (model.a - amin) / (amax - amin)
            embedding_min.append(0)
            embedding_max.append(1)

            # Lin_edge limits
            model_name = config.graph_model.signal_model_name
            if model_name in ['PDE_N2', 'PDE_N3', 'PDE_N6']:
                # Simple models: single input
                in_features = rr[:, None]
                func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                lin_edge_min.append(to_numpy(func).min())
                lin_edge_max.append(to_numpy(func).max())
            else:
                # Models with embeddings (PDE_N4, PDE_N5, PDE_N7, PDE_N8, PDE_N11, etc.)
                for n in range(min(10, n_neurons)):
                    embedding_ = model.a[n, :] * torch.ones((100, config.graph_model.embedding_dim), device=device)
                    if model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
                        in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    elif model_name == 'PDE_N5':
                        in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
                    elif model_name == 'PDE_N8':
                        in_features = torch.cat((rr[:, None]*0, rr[:, None], embedding_, embedding_), dim=1)
                    else:
                        in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    func = model.lin_edge(in_features.float())
                    if config.graph_model.lin_edge_positive:
                        func = func ** 2
                    lin_edge_min.append(to_numpy(func).min())
                    lin_edge_max.append(to_numpy(func).max())

            # Lin_phi limits
            for n in range(min(10, n_neurons)):
                embedding_ = model.a[n, :] * torch.ones((100, config.graph_model.embedding_dim), device=device)
                in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                func = model.lin_phi(in_features.float())
                lin_phi_min.append(to_numpy(func).min() * to_numpy(ynorm))
                lin_phi_max.append(to_numpy(func).max() * to_numpy(ynorm))

    # Add margins
    margin = 0.1
    limits = {
        'weight': (min(weight_min) * (1 + margin), max(weight_max) * (1 + margin)),
        'embedding': (-0.1, 1.1),
        'lin_edge': (min(lin_edge_min) * (1 + margin) - 0.1, max(lin_edge_max) * (1 + margin) + 0.1),
        'lin_phi': (min(lin_phi_min) * (1 + margin) - 0.5, max(lin_phi_max) * (1 + margin) + 0.5),
    }

    return limits


def create_signal_weight_subplot(fig, ax, model, connectivity, mc, epoch, iteration, config, limits, second_correction=1.0, apply_weight_correction=False, xnorm=None, device=None):
    """Create weight comparison scatter plot for signal models.

    Args:
        second_correction: Correction factor to apply to predicted weights (default 1.0 = no correction)
        apply_weight_correction: Compute and apply per-neuron correction on the fly (default False)
        xnorm: Normalization value for computing correction (required if apply_weight_correction=True)
        device: Torch device (required if apply_weight_correction=True)
    """
    n_neurons = connectivity.shape[0]
    n_plot = n_neurons

    A = get_model_W(model)[:n_plot, :n_plot].clone().detach()
    A.fill_diagonal_(0)

    # compute and apply per-neuron correction on the fly
    if apply_weight_correction and xnorm is not None and device is not None:
        # compute correction from MLP1 at high x values (same as 'best' option)
        rr = torch.linspace(0, xnorm.squeeze() * 4, 1000).to(device)
        func_list = []
        model_config = config.graph_model
        for n in range(n_plot):
            if model_config.signal_model_name in ['PDE_N4', 'PDE_N5', 'PDE_N7', 'PDE_N11']:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model_config.signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
                    in_features = torch.cat((rr[:, None], embedding_), dim=1)
                elif model_config.signal_model_name == 'PDE_N5':
                    in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
                else:
                    in_features = torch.cat((rr[:, None], embedding_), dim=1)
            else:
                in_features = rr[:, None]
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            if config.graph_model.lin_edge_positive:
                func = func ** 2
            func_list.append(func)
        func_list = torch.stack(func_list).squeeze()
        upper = torch.median(func_list[:, 850:1000], dim=1)[0]
        correction = 1 / (upper + 1E-16)

        # apply correction: transpose, apply row-wise, transpose back
        A_t = A.t()
        correction_np = to_numpy(correction[:n_plot])
        pred_weight = to_numpy(A_t)
        for row in range(n_plot):
            pred_weight[row, :] = pred_weight[row, :] / correction_np[row]
        pred_weight = pred_weight.T.flatten()
    else:
        pred_weight = to_numpy(A).flatten() / second_correction

    gt_weight = to_numpy(connectivity[:n_plot, :n_plot]).flatten()

    # adjust scatter plot parameters based on number of neurons (consistent with 'best' mode)
    if n_neurons < 1000:
        scatter_size = 1
        scatter_alpha = 1.0
    else:
        scatter_size = 0.1
        scatter_alpha = 0.1

    # plot green diagonal line (y=x, perfect correlation) first
    weight_max = np.max(np.abs(gt_weight)) * 1.1
    weight_lim = (-weight_max, weight_max)
    ax.plot(weight_lim, weight_lim, c='g', linewidth=4, zorder=1)

    ax.scatter(gt_weight, pred_weight, s=scatter_size, c=mc, alpha=scatter_alpha, zorder=2)

    # compute R² and slope
    lin_fit, _ = curve_fit(linear_model, gt_weight, pred_weight)
    residuals = pred_weight - linear_model(gt_weight, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((pred_weight - np.mean(pred_weight)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # variable names based on model type
    if 'PDE_N11' in config.graph_model.signal_model_name:
        true_weight_var = '$J_{ij}$'
        learned_weight_var = '$J_{ij}^*$'
    else:
        true_weight_var = '$W_{ij}$'
        learned_weight_var = '$W_{ij}^*$'

    ax.set_xlabel(f'true {true_weight_var}', fontsize=32)
    ax.set_ylabel(f'learned {learned_weight_var}', fontsize=32)
    # ax.set_xlim(weight_lim)
    # ax.set_ylim(weight_lim)
    ax.tick_params(labelsize=16)

    # add R² and slope text
    ax.text(0.05, 0.95, f'$R^2$: {r_squared:.3f}', transform=ax.transAxes,
            fontsize=20, verticalalignment='top')
    ax.text(0.05, 0.87, f'slope: {lin_fit[0]:.2f}', transform=ax.transAxes,
            fontsize=20, verticalalignment='top')
    ax.text(0.05, 0.79, f'epoch: {epoch}', transform=ax.transAxes,
            fontsize=16, verticalalignment='top')

    return r_squared, lin_fit[0]


def create_signal_embedding_subplot(fig, ax, model, type_list, n_neuron_types, cmap, limits):
    """Create embedding scatter plot for signal models."""
    # amax = torch.max(model.a, dim=0).values
    # amin = torch.min(model.a, dim=0).values
    # model_a = (model.a - amin) / (amax - amin + 1E-8)

    for n in range(n_neuron_types - 1, -1, -1):
        pos = torch.argwhere(type_list == n).squeeze()
        if pos.numel() > 0:
            ax.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]),
                      s=30, color=cmap.color(n), alpha=0.8, edgecolors='none')

    ax.set_xlabel(r'$\mathbf{a}_0$', fontsize=32)
    ax.set_ylabel(r'$\mathbf{a}_1$', fontsize=32)
    # ax.set_xlim([-0.2, 1.2])
    # ax.set_ylim([-0.2, 1.2])
    ax.tick_params(labelsize=16)


def create_signal_lin_edge_subplot(fig, ax, model, config, n_neurons, type_list, cmap, device, xnorm, limits, mc, label_style='MLP', true_model=None, n_neuron_types=1, apply_weight_correction=False):
    """Create lin_edge function plot for signal models.

    Args:
        label_style: 'MLP' for MLP_0, MLP_1 labels; 'greek' for phi, f labels
        true_model: Ground truth model for plotting true curves (optional)
        n_neuron_types: Number of neuron types for true curve plotting
        apply_weight_correction: Compute and apply per-neuron correction on the fly (default False)
    """
    rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1000).to(device)

    model_name = config.graph_model.signal_model_name

    # Compute normalization factor from true model at x > 6 (asymptotic region)
    norm_factor = 1.0
    if true_model is not None:
        with torch.no_grad():
            # Evaluate at x > 6 to get true asymptotic value (well into saturation for tanh)
            rr_asymptotic = torch.linspace(6.0, 10.0, 100).to(device)
            true_max = 0.0
            for n in range(n_neuron_types):
                true_func = true_model.func(rr_asymptotic, n, 'phi')
                # Get mean absolute value in asymptotic region
                true_asymptotic = torch.abs(true_func).mean().item()
                true_max = max(true_max, true_asymptotic)
            if true_max > 0:
                norm_factor = true_max

    # Plot true curves first (green, thick) if true_model is provided
    # Plot one curve per neuron, colored by neuron type
    if true_model is not None:
        neuron_types = to_numpy(type_list).astype(int)
        for n in range(n_neurons):
            neuron_type = int(neuron_types[n])
            true_func = true_model.func(rr, neuron_type, 'phi')
            ax.plot(to_numpy(rr), to_numpy(true_func) / norm_factor, color=cmap.color(neuron_type), linewidth=8, alpha=1.0)

    max_radius = config.simulation.max_radius

    if model_name in ['PDE_N2', 'PDE_N3', 'PDE_N6']:
        # Simple models: single line, single input
        in_features = rr[:, None]
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        if config.graph_model.lin_edge_positive:
            func = func ** 2
        ax.plot(to_numpy(rr), to_numpy(func) / norm_factor, color='w', linewidth=4)
    else:
        # Models with embeddings: multiple lines per type (PDE_N4, PDE_N5, PDE_N7, PDE_N8, PDE_N11, etc.)
        if apply_weight_correction:
            # First pass: compute per-neuron correction

            rr = torch.linspace(0 , xnorm.squeeze() * 4 , 1000).to(device)
            func_list = []
            for n in range(0,n_neurons):
                if config.graph_model.signal_model_name in ['PDE_N4', 'PDE_N5', 'PDE_N7', 'PDE_N11']:
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    if config.graph_model.signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
                        in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    elif config.graph_model.signal_model_name == 'PDE_N5':
                        in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
                    else:
                        in_features = get_in_features(rr, embedding_, model, model_config.signal_model_name, max_radius) # noqa: F821
                else:
                    in_features = rr[:,None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                func_list.append(func)
            func_list = torch.stack(func_list).squeeze()

            upper = torch.median(func_list[:,850:1000], dim=1)[0]
            correction = 1 / (upper + 1E-16)

            # Second pass: plot with correction applied
            rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1500).to(device)
            for n in range(0,n_neurons):
                if config.graph_model.signal_model_name in ['PDE_N4', 'PDE_N5', 'PDE_N7', 'PDE_N11']:
                    embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model, config.graph_model.signal_model_name, max_radius)
                else:
                    in_features = rr[:, None]
                with torch.no_grad():
                    if config.graph_model.lin_edge_positive:
                        func = model.lin_edge(in_features.float()) ** 2 * correction[n]
                    else:
                        func = model.lin_edge(in_features.float()) * correction[n]
                plt.plot(to_numpy(rr), to_numpy(func), color='w', linewidth=1, alpha=0.25)
        else:
            # No correction: plot raw functions
            for n in range(n_neurons):
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = get_in_features(rr, embedding_, model, model_name, max_radius)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                ax.plot(to_numpy(rr), to_numpy(func),
                       color='w', linewidth=1, alpha=0.25)

    # variable names
    if 'PDE_N11' in model_name:
        signal_var = '$h_j$'
    else:
        signal_var = '$v_j$'

    # label style
    if label_style == 'MLP':
        ylabel = r'$\mathrm{MLP_1}(\mathbf{a}_j, h_j)$'
    else:
        ylabel = r'$f$'

    ax.set_xlabel(signal_var, fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.set_xlim([-to_numpy(xnorm).item(), to_numpy(xnorm).item()])
    # set ylim to normalized range only when correction is applied
    if apply_weight_correction:
        ax.set_ylim([-1.2, 1.2])
    ax.tick_params(labelsize=16)


def create_signal_lin_phi_subplot(fig, ax, model, config, n_neurons, type_list, cmap, device, xnorm, ynorm, limits, label_style='MLP', true_model=None, n_neuron_types=1):
    """Create lin_phi function plot for signal models.

    Args:
        label_style: 'MLP' for MLP_0, MLP_1 labels; 'greek' for phi, f labels
        true_model: Ground truth model for plotting true curves (optional)
        n_neuron_types: Number of neuron types for true curve plotting
    """
    rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1000).to(device)

    # Plot true curves first (one curve per neuron, colored by neuron type)
    if true_model is not None:
        neuron_types = to_numpy(type_list).astype(int)
        for n in range(n_neurons):
            neuron_type = int(neuron_types[n])
            true_func = true_model.func(rr, neuron_type, 'update')
            ax.plot(to_numpy(rr), to_numpy(true_func), color=cmap.color(neuron_type), linewidth=8, alpha=1.0)

    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = get_in_features_update(rr[:, None], model, embedding_, device)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        func = func[:, 0]
        ax.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
               color='w', linewidth=1, alpha=0.3)

    # Variable names and labels
    if 'PDE_N11' in config.graph_model.signal_model_name:
        signal_var = '$h_i$'
    else:
        signal_var = '$v_i$'

    # Label style
    if label_style == 'MLP':
        if config.training.training_single_type:
            ylabel = rf'$\mathrm{{MLP_0}}({signal_var[1:-1]})$'
        else:
            ylabel = rf'$\mathrm{{MLP_0}}(\mathbf{{a}}_i, {signal_var[1:-1]})$'
    else:
        if config.training.training_single_type:
            ylabel = rf'$\phi({signal_var[1:-1]})$'
        else:
            ylabel = rf'$\phi(\mathbf{{a}}_i, {signal_var[1:-1]})$'

    ax.set_xlabel(signal_var, fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.set_xlim([-to_numpy(xnorm).item(), to_numpy(xnorm).item()])
    ax.set_ylim(limits['lin_phi'])
    ax.tick_params(labelsize=16)


def create_signal_excitation_subplot(fig, ax, model, config, n_frames, mc, device,
                                      connectivity=None, second_correction=1.0):
    """Create excitation function plot for signal models.

    Shows learned oscillation excitation function vs ground truth cosine.

    Args:
        connectivity: Connectivity matrix (used for computing e_i slope correction)
        second_correction: Correction factor from weight comparison (default 1.0)
    """
    simulation_config = config.simulation

    with torch.no_grad():
        kk = torch.arange(0, n_frames, dtype=torch.float32, device=device) / model.NNR_f_T_period
        excitation_field = model.NNR_f(kk[:, None])
        model_a = model.a[-1] * torch.ones((n_frames, 1), device=device)
        in_features = torch.cat([excitation_field, model_a], dim=1)
        msg = model.lin_edge(in_features)

    msg_raw = to_numpy(msg.squeeze())
    frame_ = np.arange(0, len(msg_raw)) / len(msg_raw)
    # Ground truth uses oscillation_max_amplitude from config
    gt_max_amplitude = getattr(simulation_config, 'oscillation_max_amplitude', 1.0)
    gt_excitation = gt_max_amplitude * np.cos((2 * np.pi) * simulation_config.oscillation_frequency * frame_)

    # Compute exc_slope from e_i weights (if connectivity provided)
    exc_slope = 1.0
    if connectivity is not None:
        gt_weight_exc = to_numpy(connectivity[:-1, -1])
        pred_weight_exc = -to_numpy(get_model_W(model)[:-1, -1]) / second_correction
        # Simple linear regression for slope
        x_mean = np.mean(gt_weight_exc)
        y_mean = np.mean(pred_weight_exc)
        denom = np.sum((gt_weight_exc - x_mean) ** 2)
        if denom > 1e-10:
            exc_slope = np.sum((gt_weight_exc - x_mean) * (pred_weight_exc - y_mean)) / denom
        if abs(exc_slope) < 0.01:
            exc_slope = 1.0  # Avoid division by very small number

    # e_i is corrected by: / second_correction / exc_slope
    # f(t) should be corrected by: / exc_slope (to match e_i scale)
    excitation = -msg_raw / exc_slope

    # Plot ground truth (green, thick)
    ax.plot(gt_excitation, c='g', linewidth=8, alpha=0.75, label='ground truth')
    # Plot learned (marker color, thinner)
    ax.plot(excitation, c=mc, linewidth=1, label='learned')

    ax.set_xlabel('$t$', fontsize=32)
    ax.set_ylabel('$\\mathrm{INR}(t)$', fontsize=32)
    ax.tick_params(labelsize=16)
    ax.set_xlim([0, 2000])  # Zoom into first 2000 frames
    # Set ylim based on ground truth amplitude
    ax.set_ylim([-gt_max_amplitude * 1.2, gt_max_amplitude * 1.2])
    ax.legend(fontsize=16, loc='upper right')


# ============================================================================
# Shared Movie Creation Helpers
# ============================================================================

def setup_movies_directory(log_dir):
    """Create and return movies directory path.

    Args:
        log_dir: Log directory path

    Returns:
        movies_dir: Path to movies directory
    """
    movies_dir = f'{log_dir}/results/movies'
    os.makedirs(movies_dir, exist_ok=True)
    return movies_dir


def create_movie(mp4_path, jpg_path, figsize, fps, metadata, file_id_list, frame_callback, dpi=100):
    """Generic movie creation loop.

    Args:
        mp4_path: Path for output MP4 file
        jpg_path: Path for first frame JPG (can be None to skip)
        figsize: Figure size tuple
        fps: Frames per second
        metadata: Metadata dict for FFMpegWriter
        file_id_list: List of file indices to iterate
        frame_callback: Function(fig, file_id_) that creates each frame
        dpi: DPI for movie frames

    Returns:
        Result from frame_callback if any (e.g., r_squared_list)
    """
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=figsize)

    first_frame_saved = False
    results = []

    with writer.saving(fig, mp4_path, dpi=dpi):
        for file_id_ in tqdm(range(len(file_id_list)), desc=os.path.basename(mp4_path).replace('.mp4', ''), ncols=90):
            plt.clf()

            result = frame_callback(fig, file_id_)
            if result is not None:
                results.append(result)

            plt.tight_layout()
            writer.grab_frame()

            if jpg_path and not first_frame_saved:
                plt.savefig(jpg_path, dpi=150, bbox_inches='tight')
                first_frame_saved = True

    plt.close(fig)
    return results if results else None


def load_model_state(model, net_path, device):
    """Load model state from checkpoint.

    Args:
        model: Model instance to load state into
        net_path: Path to checkpoint file
        device: Torch device

    Returns:
        model: Model with loaded state in eval mode
    """
    state_dict = torch.load(net_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    return model


def get_epoch_from_file(file_path, n_runs):
    """Extract epoch number from model file path.

    Args:
        file_path: Path to model file
        n_runs: Number of training runs

    Returns:
        epoch: Epoch string
    """
    return file_path.split('graphs')[1][1:-3]


# ============================================================================


def create_signal_movies(config, log_dir, n_runs, device, n_neurons, n_neuron_types, type_list,
                         cmap, connectivity, xnorm, ynorm, mc, fps=10,
                         true_model=None, apply_weight_correction=False):
    """
    Create movies for signal model training visualization.

    Args:
        config: Configuration object
        log_dir: Log directory path
        n_runs: Number of training runs
        device: Torch device
        n_neurons: Number of neurons
        n_neuron_types: Number of neuron types
        type_list: Tensor of neuron types
        cmap: Color map
        connectivity: Ground truth connectivity matrix
        xnorm, ynorm: Normalization values
        mc: Marker color ('k' or 'w')
        fps: Frames per second for movies
        true_model: Ground truth model for plotting true curves (optional)
        apply_weight_correction: Apply per-neuron correction from correction.pt (default False)
    """
    # Get label_style from config (default to 'MLP')
    label_style = getattr(config.plotting, 'label_style', 'MLP')
    n_frames = config.simulation.n_frames

    # Extract just the base name from dataset (may contain path like 'signal/signal_N11_1')
    dataset_name = os.path.basename(config.dataset)
    movies_dir = setup_movies_directory(log_dir)

    # get training files
    files, file_id_list = get_training_files(log_dir, n_runs)
    if len(file_id_list) == 0:
        print('no training files found for movie creation')
        return

    # determine stable plot limits
    print('determining plot limits...')
    limits = determine_plot_limits_signal(config, log_dir, n_runs, device, n_neurons, type_list,
                                          cmap, connectivity, xnorm, ynorm,
                                          true_model=true_model, n_neuron_types=n_neuron_types)
    if limits is None:
        print('could not determine plot limits')
        return

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    # load second_correction if available (computed in 'best' mode)
    second_correction_path = f'{log_dir}/second_correction.npy'
    if os.path.exists(second_correction_path):
        second_correction = float(np.load(second_correction_path))
    else:
        second_correction = 1.0

    metadata = {'title': f'{dataset_name} training', 'artist': 'NeuralGraph'}

    # movie configurations: (name, figsize, create_function)
    movie_configs = [
        ('weights', (8, 8), 'weight'),
        ('embedding', (8, 8), 'embedding'),
        ('lin_edge', (8, 8), 'lin_edge'),
        ('lin_phi', (8, 8), 'lin_phi'),
    ]

    r_squared_list = []
    slope_list = []

    # add 'without' suffix if correction is not applied
    suffix = '' if apply_weight_correction else '_without'

    for movie_name, figsize, plot_type in movie_configs:
        mp4_path = f'{movies_dir}/{movie_name}_{dataset_name}{suffix}.mp4'
        jpg_path = f'{movies_dir}/{movie_name}_{dataset_name}{suffix}.jpg'

        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=figsize)

        first_frame_saved = False

        with writer.saving(fig, mp4_path, dpi=100):
            for file_id_ in tqdm(range(len(file_id_list)), desc=f'{movie_name}', ncols=90):
                plt.clf()
                ax = fig.add_subplot(111)

                file_id = file_id_list[file_id_]
                epoch = get_epoch_from_file(files[file_id], n_runs)
                net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"
                load_model_state(model, net, device)

                with torch.no_grad():
                    if plot_type == 'weight':
                        r2, slope = create_signal_weight_subplot(fig, ax, model, connectivity, mc,
                                                                  epoch, file_id_, config, limits, second_correction,
                                                                  apply_weight_correction=apply_weight_correction,
                                                                  xnorm=xnorm, device=device)
                        if movie_name == 'weights':
                            r_squared_list.append(r2)
                            slope_list.append(slope)
                    elif plot_type == 'embedding':
                        create_signal_embedding_subplot(fig, ax, model, type_list, n_neuron_types, cmap, limits)
                    elif plot_type == 'lin_edge':
                        create_signal_lin_edge_subplot(fig, ax, model, config, n_neurons, type_list,
                                                       cmap, device, xnorm, limits, mc, label_style,
                                                       true_model=true_model, n_neuron_types=n_neuron_types,
                                                       apply_weight_correction=apply_weight_correction)
                    elif plot_type == 'lin_phi':
                        create_signal_lin_phi_subplot(fig, ax, model, config, n_neurons, type_list,
                                                      cmap, device, xnorm, ynorm, limits, label_style,
                                                      true_model=true_model, n_neuron_types=n_neuron_types)

                plt.tight_layout()
                writer.grab_frame()

                # Save first frame as JPG
                if not first_frame_saved:
                    plt.savefig(jpg_path, dpi=150, bbox_inches='tight')
                    first_frame_saved = True

        plt.close(fig)

    # create combined 2x2 movie
    mp4_path_combined = f'{movies_dir}/combined_{dataset_name}{suffix}.mp4'
    jpg_path_combined = f'{movies_dir}/combined_{dataset_name}{suffix}.jpg'

    # check if training_single_type is enabled (skip embedding panel if so)
    training_single_type = getattr(config.training, 'training_single_type', False) or getattr(config.training, 'init_training_single_type', False)

    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(16, 16))

    first_frame_saved = False

    with writer.saving(fig, mp4_path_combined, dpi=100):
        for file_id_ in tqdm(range(len(file_id_list)), desc='combined', ncols=90):
            plt.clf()

            file_id = file_id_list[file_id_]
            epoch = get_epoch_from_file(files[file_id], n_runs)
            net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"
            load_model_state(model, net, device)

            with torch.no_grad():
                # Weights subplot (top-left)
                ax1 = fig.add_subplot(2, 2, 1)
                create_signal_weight_subplot(fig, ax1, model, connectivity, mc, epoch, file_id_, config, limits, second_correction,
                                             apply_weight_correction=apply_weight_correction,
                                             xnorm=xnorm, device=device)

                # Top-right panel: embedding
                ax2 = fig.add_subplot(2, 2, 2)
                create_signal_embedding_subplot(fig, ax2, model, type_list, n_neuron_types, cmap, limits)

                # Lin_phi subplot (bottom-left) - MLP0
                ax3 = fig.add_subplot(2, 2, 3)
                create_signal_lin_phi_subplot(fig, ax3, model, config, n_neurons, type_list,
                                              cmap, device, xnorm, ynorm, limits, label_style,
                                              true_model=true_model, n_neuron_types=n_neuron_types)

                # Lin_edge subplot (bottom-right) - MLP1
                ax4 = fig.add_subplot(2, 2, 4)
                create_signal_lin_edge_subplot(fig, ax4, model, config, n_neurons, type_list,
                                               cmap, device, xnorm, limits, mc, label_style,
                                               true_model=true_model, n_neuron_types=n_neuron_types,
                                               apply_weight_correction=apply_weight_correction)

            plt.tight_layout()
            writer.grab_frame()

            if not first_frame_saved:
                plt.savefig(jpg_path_combined, dpi=150, bbox_inches='tight')
                first_frame_saved = True

    plt.close(fig)

    # create two-panel connectivity comparison movie (true vs learned heatmaps)
    mp4_path_connectivity = f'{movies_dir}/connectivity_{dataset_name}{suffix}.mp4'
    jpg_path_connectivity = f'{movies_dir}/connectivity_{dataset_name}{suffix}.jpg'

    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(16, 8))

    # prepare true connectivity heatmap data
    n_plot = n_neurons
    true_connectivity_plot = to_numpy(connectivity[:n_plot, :n_plot])

    first_frame_saved = False

    with writer.saving(fig, mp4_path_connectivity, dpi=100):
        for file_id_ in tqdm(range(len(file_id_list)), desc='connectivity', ncols=90):
            plt.clf()

            file_id = file_id_list[file_id_]
            epoch = get_epoch_from_file(files[file_id], n_runs)
            net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"
            load_model_state(model, net, device)

            with torch.no_grad():
                # Get learned weights
                A = get_model_W(model)[:n_plot, :n_plot]
                learned_connectivity_plot = to_numpy(A)

                # Left panel: true connectivity
                ax1 = fig.add_subplot(1, 2, 1)
                sns.heatmap(true_connectivity_plot, center=0, square=True, cmap='bwr',
                           cbar=False, vmin=-0.5, vmax=0.5, ax=ax1)
                ax1.set_title('true connectivity', fontsize=20)
                ax1.set_xticks([])
                ax1.set_yticks([])

                # Right panel: learned connectivity (raw)
                ax2 = fig.add_subplot(1, 2, 2)
                sns.heatmap(learned_connectivity_plot, center=0, square=True, cmap='bwr',
                           cbar=False, vmin=-0.5, vmax=0.5, ax=ax2)
                ax2.set_title('learned connectivity', fontsize=20)
                ax2.set_xticks([])
                ax2.set_yticks([])

            plt.tight_layout()
            writer.grab_frame()

            if not first_frame_saved:
                plt.savefig(jpg_path_connectivity, dpi=150, bbox_inches='tight')
                first_frame_saved = True

    plt.close(fig)

    return r_squared_list, slope_list




def plot_signal(config, epoch_list, log_dir, logger, cc, style, extended, device, apply_weight_correction=False, log_file=None):

    dataset_name = config.dataset
    model_config = config.graph_model
    simulation_config = config.simulation

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_neuron_types = config.simulation.n_neuron_types
    delta_t = config.simulation.delta_t
    p = config.simulation.params
    omega = model_config.omega
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    embedding_cluster = EmbeddingCluster(config)
    external_input_type = simulation_config.external_input_type
    if (external_input_type != '') & (external_input_type!='none'):
        n_input_neurons = config.simulation.n_input_neurons
        n_input_neurons_per_axis = int(np.sqrt(n_input_neurons))
        has_external_input = True
    else:
        has_external_input = False

    x_list = []
    y_list = []
    run =0
    if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
        x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        x = to_numpy(torch.stack(x))
        y = to_numpy(torch.stack(y))

    else:
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        if os.path.exists(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy'):
            raw_x = np.load(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy')
    x_list.append(x)
    y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    if os.path.exists(os.path.join(log_dir, 'xnorm.pt')):
        xnorm = torch.load(os.path.join(log_dir, 'xnorm.pt'))
    else:
        xnorm = torch.tensor([5], device=device)
    print(f'xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    # SVD analysis of activity and external_input (skip if already exists)
    svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.pdf')
    if not os.path.exists(svd_plot_path):
        svd_style = 'dark_background' if 'black' in style else None
        try:
            analyze_data_svd(x_list[0], log_dir, config=config, logger=logger, style=svd_style)
        except Exception as e:
            import traceback
            print(f'SVD analysis failed: {e}')
            traceback.print_exc()


    x = x_list[0][n_frames - 5]
    n_neurons = x.shape[0]

    print(f'n neurons: {n_neurons}')
    logger.info(f'N neurons: {n_neurons}')
    config.simulation.n_neurons = n_neurons
    type_list = torch.tensor(x[:, 6:7], device=device)  # neuron_type at index 6

    activity = torch.tensor(x_list[0][:, :, 3:4],device=device)
    activity = activity.squeeze()
    to_numpy(activity.flatten())
    activity = activity.t()

    if os.path.exists(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy'):
        raw_activity = torch.tensor(raw_x[:, :, 3:4], device=device)
        raw_activity = raw_activity.squeeze()
        raw_activity = raw_activity.t()

    xc, yc = get_equidistant_points(n_points=n_neurons)
    X1_first = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(X1_first.size(0), device=device)
    X1_first = X1_first[perm]
    torch.save(X1_first, f'./graphs_data/{dataset_name}/X1.pt')
    xc, yc = get_equidistant_points(n_points=n_neurons ** 2)
    X_msg = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(X_msg.size(0), device=device)
    X_msg = X_msg[perm]
    torch.save(X_msg, f'./graphs_data/{dataset_name}/X_msg.pt')

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    # Get label_style from config (default to 'MLP')
    label_style = getattr(config.plotting, 'label_style', 'MLP')

    if has_external_input:
        if ('short_term_plasticity' in external_input_type) | ('modulation_permutation' in external_input_type):
            model_f = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                            hidden_features=model_config.hidden_dim_nnr,
                            hidden_layers=model_config.n_layers_nnr, first_omega_0=omega, hidden_omega_0=omega,
                            outermost_linear=model_config.outermost_linear_nnr)
        else:
            model_f = Siren_Network(image_width=n_input_neurons_per_axis, in_features=model_config.input_size_nnr_f, out_features=model_config.output_size_nnr_f, hidden_features=model_config.hidden_dim_nnr_f,
                                            hidden_layers=model_config.n_layers_nnr_f, outermost_linear=True, device=device, first_omega_0=model_config.omega_f, hidden_omega_0=model_config.omega_f)
        model_f.to(device=device)
        model_f.train()

    if epoch_list[0] == 'all':
        files = glob.glob(f"{log_dir}/models/*.pt")
        files.sort(key=os.path.getmtime)

        model, bc_pos, bc_dpos = choose_training_model(config, device)

        # plt.rcParams['text.usetex'] = False
        # plt.rc('font', family='sans-serif')
        # plt.rc('text', usetex=False)
        # matplotlib.rcParams['savefig.pad_inches'] = 0

        files, file_id_list = get_training_files(log_dir, n_runs)

        # Load connectivity for movie generation and create true_model for plotting true curves
        connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)

        true_model, _, _ = choose_model(config=config, W=connectivity, device=device)

        # create movies with stable axes
        create_signal_movies(
            config=config,
            log_dir=log_dir,
            n_runs=n_runs,
            device=device,
            n_neurons=n_neurons,
            n_neuron_types=n_neuron_types,
            type_list=type_list,
            cmap=cmap,
            connectivity=connectivity,
            xnorm=xnorm,
            ynorm=ynorm,
            mc=mc,
            fps=10,
            true_model=true_model,
            apply_weight_correction=apply_weight_correction
        )

        # Continue with existing PNG generation for backwards compatibility
        r_squared_list = []
        slope_list = []
        it = -1
        with torch.no_grad():
            for file_id_ in trange(0,len(file_id_list), ncols=90):
                it = it + 1
                num = str(it).zfill(4)
                file_id = file_id_list[file_id_]
                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"

                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                if has_external_input:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                amax = torch.max(model.a, dim=0).values
                amin = torch.min(model.a, dim=0).values
                model_a = (model.a - amin) / (amax - amin)

                fig, ax = fig_init()
                for n in range(n_neuron_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n).squeeze()
                    plt.scatter(to_numpy(model_a[pos, 0]), to_numpy(model_a[pos, 1]), s=50, color=cmap.color(n), alpha=1.0, edgecolors='none')   # cmap.color(n)
                plt.xlabel(r'$\mathbf{a}_0$', fontsize=68)
                plt.ylabel(r'$\mathbf{a}_1$', fontsize=68)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{num}.png", dpi=80)
                plt.close()

                # Load correction factors if apply_weight_correction is enabled
                if apply_weight_correction and os.path.exists(f'{log_dir}/correction.pt'):
                    correction = torch.load(f'{log_dir}/correction.pt', map_location=device)
                    second_correction = np.load(f'{log_dir}/second_correction.npy') if os.path.exists(f'{log_dir}/second_correction.npy') else 1.0
                else:
                    correction = torch.ones(n_neurons, device=device)
                    second_correction = 1.0

                A = get_model_W(model).clone().detach()
                A.fill_diagonal_(0)
                A = A.t()

                # Apply per-neuron correction if enabled
                if apply_weight_correction:
                    A_corrected = A.clone()
                    for row in range(n_neurons):
                        A_corrected[row, :] = A[row, :] * correction[row]
                    A_plot = to_numpy(A_corrected) / second_correction
                else:
                    A_plot = to_numpy(A)

                fig, ax = fig_init()
                ax = sns.heatmap(A_plot, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1, vmax=0.1)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=48)
                plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.subplot(2, 2, 1)
                ax = sns.heatmap(A_plot[0:20, 0:20], cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/W_{num}.png", dpi=80)
                plt.close()


                rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1000).to(device)
                if model_config.signal_model_name == 'PDE_N5':
                    fig, ax = fig_init()
                    plt.axis('off')
                    for k in range(n_neuron_types):
                        ax = fig.add_subplot(2, 2, k + 1)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(cmap.color(k))
                            spine.set_linewidth(3)
                        if k==0:
                            plt.ylabel(r'learned $\mathrm{MLP_1}( a_i, a_j, v_j)$', fontsize=32)
                        for n in range(n_neuron_types):
                            for m in range(250):
                                pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                                pos1 = to_numpy(torch.argwhere(type_list == n).squeeze())
                                n0 = np.random.randint(len(pos0))
                                n0 = pos0[n0, 0]
                                n1 = np.random.randint(len(pos1))
                                n1 = pos1[n1, 0]
                                embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                                embedding1 = model.a[n1, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                                in_features = torch.cat((rr[:, None], embedding0, embedding1), dim=1)
                                func = model.lin_edge(in_features.float())
                                if apply_weight_correction:
                                    func = func * correction[n0]
                                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(n), linewidth=3, alpha=0.25)
                        plt.ylim([-1.6, 1.6])
                        plt.xlim([-5, 5])
                        plt.xticks([])
                        plt.yticks([])
                    plt.xlabel(r'$x_j$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N4'):
                    fig, ax = fig_init()
                    for k in range(n_neuron_types):
                        for m in range(250):
                            pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                            n0 = np.random.randint(len(pos0))
                            n0 = pos0[n0, 0]
                            embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            in_features = get_in_features(rr, embedding0, model, model_config.signal_model_name, max_radius)
                            if config.graph_model.lin_edge_positive:
                                func = model.lin_edge(in_features.float()) ** 2
                            else:
                                func = model.lin_edge(in_features.float())
                            if apply_weight_correction:
                                func = func * correction[n0]
                            plt.plot(to_numpy(rr), to_numpy(func), color=cmap.color(k), linewidth=2, alpha=0.25)
                    plt.xlabel(r'$x_j$', fontsize=68)
                    plt.ylabel(r'learned $\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=68)
                    plt.ylim([-1.6, 1.6])
                    plt.xlim([-to_numpy(xnorm)//2, to_numpy(xnorm)//2])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N8'):
                    rr = torch.linspace(0, 10, 1000).to(device)
                    fig, ax = fig_init()
                    for idx, k in enumerate(np.linspace(4, 10, 13)):  # Corrected step size to generate 13 evenly spaced values

                        for n in range(0,n_neurons,4):
                            embedding_i = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            embedding_j = model.a[np.random.randint(n_neurons), :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            if model.embedding_trial:
                                in_features = torch.cat((torch.ones_like(rr[:, None]) * k, rr[:, None], embedding_i, embedding_j, model.b[0].repeat(1000, 1)), dim=1)
                            else:
                                in_features = torch.cat((rr[:, None], torch.ones_like(rr[:, None])*k, embedding_i, embedding_j), dim=1)
                            with torch.no_grad():
                                func = model.lin_edge(in_features.float())
                            if config.graph_model.lin_edge_positive:
                                func = func ** 2
                            plt.plot(to_numpy(rr-k), to_numpy(func), 2, color=cmap.color(idx), linewidth=2, alpha=0.25)
                    plt.xlabel(r'$x_i-x_j$', fontsize=68)
                    # plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                    plt.ylabel(r'$\mathrm{MLP_1}(\mathbf{a}_i, a_j, v_i, v_j)$', fontsize=68)
                    plt.ylim([0,4])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N11'):
                    fig, ax = fig_init()
                    # Compute normalization factor from true model at x > 6 (asymptotic region)
                    norm_factor = 1.0
                    if true_model is not None:
                        rr_asymptotic = torch.linspace(6.0, 10.0, 100).to(device)
                        true_max = 0.0
                        for n_type in range(n_neuron_types):
                            true_func = true_model.func(rr_asymptotic, n_type, 'phi')
                            true_asymptotic = torch.abs(true_func).mean().item()
                            true_max = max(true_max, true_asymptotic)
                        if true_max > 0:
                            norm_factor = true_max
                    # Plot true curves first (green, thick)
                    if true_model is not None:
                        for n_type in range(n_neuron_types):
                            true_func = true_model.func(rr, n_type, 'phi')
                            plt.plot(to_numpy(rr), to_numpy(true_func) / norm_factor, c='g', linewidth=8)
                    # Plot learned curves
                    for k in range(n_neuron_types):
                        pos0 = torch.argwhere(type_list == k).squeeze()
                        if pos0.numel() == 0:
                            continue
                        pos0_np = to_numpy(pos0)
                        if pos0_np.ndim == 0:
                            pos0_np = np.array([[pos0_np.item()]])
                        elif pos0_np.ndim == 1:
                            pos0_np = pos0_np[:, None]
                        for m in range(250):
                            n0 = np.random.randint(len(pos0_np))
                            n0 = pos0_np[n0, 0]
                            embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            in_features = torch.cat((rr[:, None], embedding0), dim=1)
                            with torch.no_grad():
                                func = model.lin_edge(in_features.float())
                            if apply_weight_correction:
                                func = func * correction[n0]
                            plt.plot(to_numpy(rr), to_numpy(func) / norm_factor, color=cmap.color(k), linewidth=2, alpha=0.25)
                    plt.xlabel(r'$v_j$', fontsize=68)
                    if label_style == 'MLP':
                        plt.ylabel(r'$\mathrm{MLP_1}$', fontsize=68)
                    else:
                        plt.ylabel(r'$f$', fontsize=68)
                    plt.ylim([-1.2, 1.2])
                    plt.xlim([-to_numpy(xnorm).item(), to_numpy(xnorm).item()])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()

                fig, ax = fig_init()
                func_list = []
                for n in range(n_neurons):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25)
                plt.ylim([-4, 4])
                plt.xlabel(r'$v_i$', fontsize=68)
                plt.ylabel(r'learned $\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=68)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP0_{num}.png", dpi=80)
                plt.close()

                A = get_model_W(model).clone().detach()
                A.fill_diagonal_(0)

                # apply per-neuron correction if enabled
                if apply_weight_correction:
                    correction_np = to_numpy(correction)
                    pred_weight = to_numpy(A)
                    for row in range(n_neurons):
                        pred_weight[row, :] = pred_weight[row, :] / correction_np[row]
                    pred_weight = pred_weight / second_correction
                else:
                    pred_weight = to_numpy(A)

                fig, ax = fig_init()
                # use true_model.W for ground truth (like 'best' option)
                if hasattr(true_model, 'W') or (hasattr(true_model, 'WL') and hasattr(true_model, 'WR')):
                    gt_weight = to_numpy(get_model_W(true_model))
                else:
                    gt_weight = to_numpy(connectivity)
                plt.scatter(gt_weight, pred_weight, s=0.1, c=mc, alpha=0.1)
                plt.xlabel(r'true $W_{ij}$', fontsize=68)
                plt.ylabel(r'learned $W_{ij}$', fontsize=68)
                if n_neurons == 8000:
                    plt.xlim([-0.05, 0.05])
                    plt.ylim([-0.05, 0.05])
                else:
                    # plt.xlim([-0.2, 0.2])
                    # plt.ylim([-0.2, 0.2])
                    plt.xlim([-0.15, 0.15])
                    plt.ylim([-0.15, 0.15])

                x_data = np.reshape(gt_weight, (n_neurons * n_neurons))
                y_data = np.reshape(pred_weight, (n_neurons * n_neurons))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                r_squared_list.append(r_squared)
                slope_list.append(lin_fit[0])

                if n_neurons == 8000:
                    plt.text(-0.042, 0.042, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.042, 0.036, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                else:
                    # plt.text(-0.17, 0.15, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    # plt.text(-0.17, 0.12, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                    plt.text(-0.13, 0.13, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.13, 0.11, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/comparison_{num}.png", dpi=80)
                plt.close()

                if has_external_input:

                    if 'short_term_plasticity' in external_input_type:
                        fig, ax = fig_init()
                        t = torch.linspace(1, 100000, 1, dtype=torch.float32, device=device).unsqueeze(1)
                        prediction = model_f(t) ** 2
                        prediction = prediction.t()
                        plt.imshow(to_numpy(prediction), aspect='auto', cmap='gray')
                        plt.title(r'learned $FMLP(t)_i$', fontsize=68)
                        plt.xlabel(r'$t$', fontsize=68)
                        plt.ylabel(r'$i$', fontsize=68)
                        plt.xticks([10000,100000], [10000, 100000], fontsize=48)
                        plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/yi_{num}.png", dpi=80)
                        plt.close()

                        prediction = prediction * torch.tensor(second_correction,device=device) / 10

                        fig, ax = fig_init()
                        ids = np.arange(0,100000,100).astype(int)
                        plt.scatter(to_numpy(modulation[:,ids]), to_numpy(prediction[:,ids]), s=1, color=mc, alpha=0.05) # noqa F821
                        # plt.xlim([0,0.5])
                        # plt.ylim([0,2])
                        # plt.xticks([0,0.5], [0,0.5], fontsize=48)
                        # plt.yticks([0,1,2], [0,1,2], fontsize=48)
                        x_data = to_numpy(modulation[:,ids]).flatten() # noqa F821
                        y_data = to_numpy(prediction[:,ids]).flatten()
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        plt.xlabel(r'true $y_i(t)$', fontsize=68)
                        plt.ylabel(r'learned $y_i(t)$', fontsize=68)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/comparison_yi_{num}.png", dpi=80)
                        plt.close()

                    else:

                        fig, ax = fig_init()
                        pred = model_f(time=file_id_ / len(file_id_list), enlarge=True) ** 2
                        # pred = torch.reshape(pred, (n_input_neurons_per_axis, n_input_neurons_per_axis))
                        pred = torch.reshape(pred, (640, 640))
                        pred = to_numpy(torch.sqrt(pred))
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        plt.imshow(pred, cmap='grey')
                        plt.ylabel(r'learned $FMLP(s_i, t)$', fontsize=68)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/field_{num}.png", dpi=80)
                        plt.close()

                if 'derivative' in external_input_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 400)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid) / 40
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig, ax = fig_init()
                    plt.title(r'learned $MLP_2(x_i, y_i)$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/all/derivative_yi_{num}.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(to_numpy(true_derivative.flatten()), to_numpy(pred_modulation.flatten()), s=5, color=mc, alpha=0.1)
                    x_data = to_numpy(true_derivative.flatten())
                    y_data = to_numpy(pred_modulation.flatten())
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.xlabel(r'true $\dot{y_i}(t)$', fontsize=68)
                    plt.ylabel(r'learned $\dot{y_i}(t)$', fontsize=68)

                    # plt.xticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    # plt.yticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    # plt.xlim([-0.2,0.025])
                    # plt.ylim([-0.2,0.025])

                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/comparison_derivative_yi_{num}.png", dpi=80)
                    plt.close()

                if (model.update_type == 'generic') & (model_config.signal_model_name == 'PDE_N5'):

                    k = np.random.randint(n_frames - 50)
                    x = torch.tensor(x_list[0][k], device=device)

                    fig, ax = fig_init()
                    msg_list = []
                    u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
                    for sample in range(n_neurons):
                        id0 = np.random.randint(0, n_neurons)
                        id1 = np.random.randint(0, n_neurons)
                        f = x[id0, 8:9]
                        embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim),
                                                                  device=device)
                        embedding1 = model.a[id1, :] * torch.ones((400, config.graph_model.embedding_dim),
                                                                  device=device)
                        in_features = torch.cat((u[:, None], embedding0, embedding1), dim=1)
                        msg = model.lin_edge(in_features.float()) ** 2 * correction
                        in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msg,
                                                 f * torch.ones((400, 1), device=device)), dim=1)
                        plt.plot(to_numpy(u), to_numpy(msg), c=cmap.color(to_numpy(x[id0, 5]).astype(int)), linewidth=2, alpha=0.25)
                        # plt.scatter(to_numpy(u), to_numpy(model.lin_phi(in_features)), s=5, c='r', alpha=0.15)
                        # plt.scatter(to_numpy(u), to_numpy(f*msg), s=1, c='w', alpha=0.1)
                        msg_list.append(msg)
                    plt.tight_layout()
                    msg_list = torch.stack(msg_list).squeeze()
                    y_min, y_max = msg_list.min().item(), msg_list.max().item()
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'learned MLPs', fontsize=68)
                    plt.ylim([y_min - y_max/2, y_max * 1.5])
                    plt.tight_layout()
                    plt.savefig(f'./{log_dir}/results/all/MLP1_{num}.png', dpi=80)
                    plt.close()

                im0 = imageio.imread(f"./{log_dir}/results/all/comparison_{num}.png")
                im1 = imageio.imread(f"./{log_dir}/results/all/embedding_{num}.png")
                im2 = imageio.imread(f"./{log_dir}/results/all/MLP0_{num}.png")
                im3 = imageio.imread(f"./{log_dir}/results/all/MLP1_{num}.png")

                fig = plt.figure(figsize=(16, 16))
                plt.axis('off')
                plt.subplot(2, 2, 1)
                plt.axis('off')
                plt.imshow(im0)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 2)
                plt.axis('off')
                plt.imshow(im1)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.subplot(2, 2, 3)
                plt.axis('off')
                plt.imshow(im2)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 4)
                plt.axis('off')
                plt.imshow(im3)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()

                plt.savefig(f"./{log_dir}/results/training/fig_{num}.png", dpi=80)
                plt.close()

    else:

        files = glob.glob(f'./{log_dir}/results/*.png')
        for f in files:
            if 'svd' not in f:
                os.remove(f)

        connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)

        # Create true_model for computing plot limits with true MLP values
        true_model_for_limits, _, _ = choose_model(config=config, W=connectivity, device=device)

        # Compute plot limits for consistent axes (includes true MLP values)
        limits = determine_plot_limits_signal(
            config=config,
            log_dir=log_dir,
            n_runs=n_runs,
            device=device,
            n_neurons=n_neurons,
            type_list=type_list,
            cmap=cmap,
            connectivity=connectivity,
            xnorm=xnorm,
            ynorm=ynorm,
            n_samples=5,
            true_model=true_model_for_limits,
            n_neuron_types=n_neuron_types
        )

        # Determine variable name based on model type
        model_name = config.graph_model.signal_model_name
        if 'PDE_N11' in model_name:
            weight_var = '$J_{ij}$'
        else:
            weight_var = '$W_{ij}$'

        adjacency = connectivity.t().clone().detach()
        adj_t = torch.abs(adjacency) > 0
        edge_index = adj_t.nonzero().t().contiguous()

        # Compute weight limits from actual data
        n_plot = n_neurons
        gt_weight_flat = to_numpy(connectivity[:n_plot, :n_plot]).flatten()
        weight_max = np.max(np.abs(gt_weight_flat)) * 1.1
        weight_lim = (-weight_max, weight_max)

        plt.figure(figsize=(10, 10))
        connectivity_plot = adjacency[:n_plot, :n_plot]
        ax = sns.heatmap(to_numpy(connectivity_plot), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=weight_lim[0], vmax=weight_lim[1])
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_plot - 1], [1, n_plot], fontsize=48)
        plt.yticks([0, n_plot - 1], [1, n_plot], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity_plot[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=weight_lim[0], vmax=weight_lim[1])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/connectivity_true.tif', dpi=300)
        plt.close()

        # Fig2a: Kinograph (activity heatmap - all neurons × all frames)
        # Use imshow instead of sns.heatmap for large data (much faster)
        kinograph_path = f'./{log_dir}/results/kinograph.png'
        # Always regenerate kinograph
        plt.figure(figsize=(15, 10))
        # Use LaTeX fonts with Palatino (same as other plots in codebase)
        # plt.rcParams['text.usetex'] = True
        # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        activity_np = to_numpy(activity)
        vmax = np.abs(activity_np).max()
        # origin='lower' so neuron 1 at bottom, n_neurons at top (matching reference)
        im = plt.imshow(activity_np, aspect='auto', cmap='viridis', vmin=-vmax, vmax=vmax, origin='lower')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        # Set colorbar ticks explicitly (rounded to nearest 10)
        vmax_rounded = int(np.ceil(vmax / 10) * 10)
        cbar_ticks = np.arange(-vmax_rounded, vmax_rounded + 1, 10)
        cbar_ticks = cbar_ticks[(cbar_ticks >= -vmax) & (cbar_ticks <= vmax)]
        cbar.set_ticks(cbar_ticks)
        cbar.ax.tick_params(labelsize=32)
        plt.ylabel('neurons', fontsize=64)
        plt.xlabel('time', fontsize=64)
        plt.xticks([0, n_frames - 1], [0, n_frames], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.tight_layout()
        plt.savefig(kinograph_path, dpi=300)
        plt.close()

        # Fig2b: Sample 100 traces if n_neurons > 200
        if n_neurons >= 200:
            sampled_indices = np.random.choice(n_neurons, 100, replace=False)
            sampled_indices = np.sort(sampled_indices)
            activity_plot = to_numpy(activity[sampled_indices])
            n_plot = 100
        else:
            sampled_indices = np.arange(n_neurons)
            activity_plot = to_numpy(activity[:n_neurons])
            n_plot = n_neurons

        activity_plot = activity_plot - 10 * np.arange(n_plot)[:, None] + 200

        plt.figure(figsize=(18, 12))
        plt.rcParams['text.usetex'] = False
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
        plt.plot(activity_plot.T, linewidth=1)
        for i in range(0, n_plot, 5):
            plt.text(-200, +200 -10 * i, str(sampled_indices[i]), fontsize=24, va='center', ha='right')
        ax = plt.gca()
        ax.text(-1200, activity_plot.mean(), 'neuron index', fontsize=32, va='center', ha='center', rotation=90)
        plt.xlabel("frame", fontsize=32)
        plt.xticks(fontsize=24)
        plt.yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlim([0, n_frames])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/activity_gt.pdf', dpi=300)
        plt.close()

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt'
            model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edge_index

            A = get_model_W(model).clone().detach()
            A.fill_diagonal_(0)
            A = A.t()
            A = A[:n_neurons,:n_neurons]

            if has_external_input:
                net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])

            # print learnable parameters table
            mlp0_params = sum(p.numel() for p in model.lin_phi.parameters())
            mlp1_params = sum(p.numel() for p in model.lin_edge.parameters())
            a_params = model.a.numel()
            w_params = get_model_W(model).numel()
            print('learnable parameters:')
            print(f'  MLP0 (lin_phi): {mlp0_params:,}')
            print(f'  MLP1 (lin_edge): {mlp1_params:,}')
            print(f'  a (embeddings): {a_params:,}')
            print(f'  W (connectivity): {w_params:,}')
            total_params = mlp0_params + mlp1_params + a_params + w_params
            if has_external_input:
                siren_params = sum(p.numel() for p in model_f.parameters())
                print(f'  INR (model_f): {siren_params:,}')
                total_params += siren_params
            if hasattr(model, 'NNR_f') and model.NNR_f is not None:
                nnr_f_params = sum(p.numel() for p in model.NNR_f.parameters())
                print(f'  INR (NNR_f): {nnr_f_params:,}')
                total_params += nnr_f_params
            print(f'  total: {total_params:,}')

            if has_external_input:
                if 'short_term_plasticity' in external_input_type:

                    fig, ax = fig_init()
                    t = torch.linspace(1, 1000, 1, dtype=torch.float32, device=device).unsqueeze(1)
                    prediction = model_f(t) ** 2
                    prediction = prediction.t()
                    plt.imshow(to_numpy(prediction), aspect='auto')
                    plt.title(r'learned $MLP_2(i,t)$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    # plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    # plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/learned_plasticity.png", dpi=80)
                    plt.close()

                    modulation_short = modulation[:, np.linspace(0, 100000, 1000).astype(int)] # noqa F821
                    activity_short = activity[:, np.linspace(0, 100000, 1000).astype(int)]

                    fig, ax = fig_init()
                    plt.scatter(to_numpy(modulation_short), to_numpy(prediction), s=1, color=mc, alpha=0.1)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/short_comparison.png", dpi=80)
                    plt.close()

                    time_step = 32
                    start = 400
                    end = 600
                    derivative_prediction = prediction[:, time_step:] - prediction[:, :-time_step]
                    derivative_prediction = derivative_prediction * 1000
                    x_ = activity_short[:, start:end].flatten()
                    y_ = modulation_short[:, start:end].flatten()
                    derivative_ = derivative_prediction[:, start-time_step//2:end-time_step//2].flatten()
                    fig, ax = fig_init()
                    plt.scatter(to_numpy(x_), to_numpy(y_), s=1, c=to_numpy(derivative_),
                                alpha=0.1, vmin=-100,vmax=100, cmap='viridis')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/plasticity_map.png", dpi=80)
                    plt.close()

                    model_pysrr = PySRRegressor( #noqa: F821
                        niterations=30,  # < Increase me for better results
                        binary_operators=["+", "-", "*", "/"],
                        random_state=0,
                        temp_equation_file=False
                    )
                    rr = torch.concatenate((y_[:, None], x_[:, None]), dim=1)
                    model_pysrr.fit(to_numpy(rr), to_numpy(derivative_[:, None]))

                    tau = 100
                    alpha = 0.02
                    true_derivative_ = (1 - y_) / tau - alpha * y_ * torch.abs(x_)
                    fig, ax = fig_init()
                    plt.scatter(to_numpy(x_), to_numpy(y_), s=10, c=to_numpy(true_derivative_),
                                alpha=1, cmap='viridis')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_plasticity_map.png", dpi=80)
                    plt.close()

            fig, ax = fig_init()
            for n in range(n_neuron_types,-1,-1):
                pos = torch.argwhere(type_list == n).squeeze()
                plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=200, color=cmap.color(n), alpha=0.25, edgecolors='none')
            if 'latex' in style:
                plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
                plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
            else:
                plt.xlabel(r'$a_{0}$', fontsize=68)
                plt.ylabel(r'$a_{1}$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/embedding.tif", dpi=170.7)
            plt.close()


            # Sample v in range [0.8*xnorm, xnorm] to get the plateau value within visible plot range
            rr = torch.linspace(0.8 * xnorm.squeeze(), xnorm.squeeze(), 1000).to(device)
            func_list = []
            for n in trange(0,n_neurons, ncols=90):
                if model_config.signal_model_name in ['PDE_N4', 'PDE_N5', 'PDE_N7', 'PDE_N11']:
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    if model_config.signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
                        in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    elif model_config.signal_model_name == 'PDE_N5':
                        in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
                    else:
                        in_features = get_in_features(rr, embedding_, model, model_config.signal_model_name, max_radius)
                else:
                    in_features = rr[:,None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                func_list.append(func)
            func_list = torch.stack(func_list).squeeze()

            # Calculate normalization - use mean of plateau region
            upper = func_list.mean(dim=1)

            print(f'normalization (v range: {0.8 * xnorm.squeeze():.2f} to {xnorm.squeeze():.2f}):')
            print(f'  mean:   min={upper.min():.4f}, max={upper.max():.4f}, mean={upper.mean():.4f}')
            print(f'  correction will be: {(1/upper.mean()):.4f}')

            correction = 1 / (upper + 1E-16)
            torch.save(correction, f'{log_dir}/correction.pt')

            print('plot functions ...')


            rr = torch.linspace(-xnorm.squeeze()  , xnorm.squeeze() , 1000).to(device)
            # MLP1 raw (without correction)
            fig, ax = fig_init()
            func_vals = []
            for n in range(n_neurons):
                if model_config.signal_model_name in ['PDE_N4', 'PDE_N5', 'PDE_N7', 'PDE_N11']:
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model, model_config.signal_model_name, max_radius)
                else:
                    in_features = rr[:, None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float()) ** 2 if config.graph_model.lin_edge_positive else model.lin_edge(in_features.float())
                func_vals.append(func)
                plt.plot(to_numpy(rr), to_numpy(func), color=mc, linewidth=4, alpha=0.25)
            func_vals = torch.stack(func_vals)
            ymin, ymax = func_vals.min().item(), func_vals.max().item()
            margin = (ymax - ymin) * 0.05
            plt.xlabel(r'$v_i$', fontsize=68)
            plt.ylabel(r'$\mathrm{MLP_1}$ (raw)', fontsize=68)
            plt.xlim([-to_numpy(xnorm), to_numpy(xnorm)])
            # plt.ylim([ymin - margin, ymax + margin])
            # Create yticks based on actual data range
            yticks = np.linspace(ymin, ymax, 11)
            ax.set_yticks(yticks)
            ax.tick_params(axis='y', labelsize=24)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/MLP1_raw.png", dpi=170.7)
            plt.close()

            fig, ax = fig_init()
            rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1500).to(device)

            # Style-dependent plotting: dark background uses green+white, white background uses gray+colored
            if 'black' in style:
                gt_color = 'g'
                line_color = 'w'
                line_alpha = 0.25
            else:
                gt_color = 'gray'
                line_color = None  # Will use cmap.color(type) instead
                line_alpha = 0.25

            # Plot ground truth curves first (behind learned curves)
            if (model_config.signal_model_name == 'PDE_N4'):
                for n in range(n_neuron_types):
                    true_func = true_model.func(rr, n, 'phi')
                    plt.plot(to_numpy(rr), to_numpy(true_func), c=gt_color, linewidth=16, label='original')
            else:
                true_func = true_model.func(rr, 0, 'phi')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=gt_color, linewidth=16, label='original')

            for n in trange(0,n_neurons, ncols=90):
                if model_config.signal_model_name in ['PDE_N4', 'PDE_N5', 'PDE_N7', 'PDE_N11']:
                    embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model, model_config.signal_model_name, max_radius)
                else:
                    in_features = rr[:, None]
                with torch.no_grad():
                    if config.graph_model.lin_edge_positive:
                        func = model.lin_edge(in_features.float()) ** 2 * correction[n]
                    else:
                        func = model.lin_edge(in_features.float()) * correction[n]
                if line_color is None:
                    # White background: use neuron type color
                    neuron_type = int(to_numpy(type_list[n]).item())
                    plt.plot(to_numpy(rr), to_numpy(func), color=cmap.color(neuron_type), linewidth=4, alpha=line_alpha)
                else:
                    plt.plot(to_numpy(rr), to_numpy(func), color=line_color, linewidth=4, alpha=line_alpha)
            plt.xlabel(r'$x_i$', fontsize=68)
            if label_style == 'MLP':
                plt.ylabel(r'$\mathrm{MLP_1}$', fontsize=68)
            else:
                if (model_config.signal_model_name == 'PDE_N4'):
                    plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, x_i)$', fontsize=68)
                elif model_config.signal_model_name == 'PDE_N5':
                    plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, a_j, x_i)$', fontsize=68)
                else:
                    plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
            plt.xlim([-to_numpy(xnorm), to_numpy(xnorm)])
            plt.ylim([-1.1, 1.1])
            ax.set_yticks([-1.0, 0.0, 1.0])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/MLP1_corrected.tif", dpi=170.7)
            plt.close()



            fig, ax = fig_init()

            # Style-dependent plotting: dark background uses green+white, white background uses gray+colored
            if 'black' in style:
                gt_color = 'g'
                line_color = 'w'
                line_alpha = 0.25
            else:
                gt_color = 'gray'
                line_color = None
                line_alpha = 0.25

            # Plot ground truth curves first (behind learned curves)
            for n in trange(n_neuron_types, ncols=90):
                true_func = true_model.func(rr, n, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=gt_color, linewidth=16, label='original')

            phi_list = []
            for n in trange(n_neurons, ncols=90):
                embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                func = func[:, 0]
                phi_list.append(func)
                if line_color is None:
                    # White background: use neuron type color
                    neuron_type = int(to_numpy(type_list[n]).item())
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(neuron_type), linewidth=4, alpha=line_alpha)
                else:
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                             color=line_color, linewidth=4, alpha=line_alpha)
            phi_list = torch.stack(phi_list)
            func_list_ = to_numpy(phi_list)
            phi_y_min = (phi_list * ynorm).min().item()
            phi_y_max = (phi_list * ynorm).max().item()
            phi_y_range = phi_y_max - phi_y_min
            phi_ylim = [phi_y_min - phi_y_range * 0.1, phi_y_max + phi_y_range * 0.1]
            plt.xlabel(r'$x_i$', fontsize=68)
            if label_style == 'MLP':
                plt.ylabel(r'$\mathrm{MLP_0}$', fontsize=68)
            else:
                plt.ylabel(r'learned $\phi^*(\mathbf{a}_i, x_i)$', fontsize=68)
            plt.tight_layout()
            # plt.xlim([-to_numpy(xnorm), to_numpy(xnorm)])
            plt.ylim(phi_ylim)
            plt.savefig(f'./{log_dir}/results/MLP0.tif', dpi=300)
            plt.close()



            print('UMAP reduction ...')

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0,
                                  random_state=config.training.seed).fit(func_list_)
                proj_interaction = trans.transform(func_list_)

            proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
            fig, ax = fig_init()
            for n in trange(n_neuron_types, ncols=90):
                pos = torch.argwhere(type_list == n)
                pos = to_numpy(pos)
                if len(pos) > 0:
                    plt.scatter(proj_interaction[pos, 0],
                                proj_interaction[pos, 1], s=200, alpha=0.1)
            plt.xlabel(r'UMAP 0', fontsize=68)
            plt.ylabel(r'UMAP 1', fontsize=68)
            plt.xlim([-0.2, 1.2])
            plt.ylim([-0.2, 1.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/UMAP.pdf", dpi=170.7)
            plt.close()

            config.training.cluster_distance_threshold = 0.1
            config.training.cluster_method = 'distance_embedding'
            embedding = to_numpy(model.a.squeeze())
            labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
                                                              config.training.cluster_distance_threshold, type_list,
                                                              n_neuron_types, embedding_cluster)
            accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels[:n_neurons])
            print(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}  ')
            logger.info(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method} ')

            dot_size = 400 if n_neurons < 1000 else 150

            # Combined plot: true types (left) and learned types (right) with accuracy
            fig, axes = plt.subplots(1, 2, figsize=(20, 12))

            # Left panel: true types
            axes[0].scatter(to_numpy(X1_first[:n_neurons, 0]), to_numpy(X1_first[:n_neurons, 1]), s=dot_size, color=cmap.color(to_numpy(type_list).astype(int)))
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            axes[0].axis('off')
            axes[0].set_aspect('equal')
            axes[0].set_title('true types', fontsize=24, color=mc)

            # Right panel: learned types
            axes[1].scatter(to_numpy(X1_first[:n_neurons, 0]), to_numpy(X1_first[:n_neurons, 1]), s=dot_size, color=cmap.color(new_labels[:n_neurons].astype(int)))
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].axis('off')
            axes[1].set_aspect('equal')
            axes[1].set_title('learned types', fontsize=24, color=mc)

            # Add accuracy and cluster info as suptitle
            fig.suptitle(f'accuracy: {accuracy:.3f}   n_clusters: {n_clusters}', fontsize=28, color=mc)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/types_comparison.pdf", dpi=170.7)
            plt.close()


            print('plot weights ...')

            fig, ax = fig_init()
            n_plot = n_neurons
            if hasattr(true_model, 'W') or (hasattr(true_model, 'WL') and hasattr(true_model, 'WR')):
                gt_weight = to_numpy(get_model_W(true_model)[:n_plot, :n_plot])
            else:
                gt_weight = to_numpy(connectivity[:n_plot, :n_plot])
            pred_weight = to_numpy(A[:n_plot, :n_plot])
            if n_plot < 1000:
                scatter_size = 1
                scatter_alpha = 1.0
            else:
                scatter_size = 0.1
                scatter_alpha = 0.1
            # Raw weights: compute R² and slope
            plt.scatter(gt_weight, pred_weight, s=scatter_size, c=mc, alpha=scatter_alpha)
            x_data_raw = np.reshape(gt_weight, (n_plot * n_plot))
            y_data_raw = np.reshape(pred_weight, (n_plot * n_plot))
            lin_fit_raw, _ = curve_fit(linear_model, x_data_raw, y_data_raw)
            residuals_raw = y_data_raw - linear_model(x_data_raw, *lin_fit_raw)
            ss_res_raw = np.sum(residuals_raw ** 2)
            ss_tot_raw = np.sum((y_data_raw - np.mean(y_data_raw)) ** 2)
            r_squared_raw = 1 - (ss_res_raw / ss_tot_raw)
            # Add R² and slope text to plot using axes transform for consistent positioning
            ax.text(0.05, 0.95, f'$R^2$: {np.round(r_squared_raw, 3)}', transform=ax.transAxes,
                    fontsize=34, verticalalignment='top')
            ax.text(0.05, 0.85, f'slope: {np.round(lin_fit_raw[0], 2)}', transform=ax.transAxes,
                    fontsize=34, verticalalignment='top')
            plt.xlabel(f'true {weight_var}', fontsize=68)
            plt.ylabel(f'learned {weight_var}', fontsize=68)
            plt.xlim(weight_lim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/weights_comparison_raw.png", dpi=87)
            plt.close()
            print(f'R² (raw): {r_squared_raw:.3f}  slope: {np.round(lin_fit_raw[0], 4)}')
            logger.info(f'R² (raw): {np.round(r_squared_raw, 4)}  slope: {np.round(lin_fit_raw[0], 4)}')







            # Corrected weights: multiply each row by correction factor
            correction_np = to_numpy(correction)
            pred_weight_corrected = pred_weight.copy()
            for i in range(n_plot):
                pred_weight_corrected[i, :] = pred_weight[i, :] / correction_np[i]

            fig, ax = fig_init()
            plt.scatter(gt_weight, pred_weight_corrected, s=scatter_size, c=mc, alpha=scatter_alpha)
            x_data = np.reshape(gt_weight, (n_plot * n_plot))
            y_data = np.reshape(pred_weight_corrected, (n_plot * n_plot))
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            # Add R² and slope text to plot using axes transform for consistent positioning
            ax.text(0.05, 0.95, f'$R^2$: {np.round(r_squared, 3)}', transform=ax.transAxes,
                    fontsize=34, verticalalignment='top')
            ax.text(0.05, 0.85, f'slope: {np.round(lin_fit[0], 2)}', transform=ax.transAxes,
                    fontsize=34, verticalalignment='top')
            plt.xlabel(f'true {weight_var}', fontsize=68)
            plt.ylabel(f'learned {weight_var}', fontsize=68)
            plt.xlim(weight_lim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/weights_comparison_corrected.tif", dpi=87)
            plt.close()
            if r_squared > 0.9:
                r2_color = '\033[92m'  # green
            elif r_squared > 0.7:
                r2_color = '\033[93m'  # yellow
            elif r_squared > 0.3:
                r2_color = '\033[38;5;208m'  # orange
            else:
                r2_color = '\033[91m'  # red
            print(f'R² (corrected): {r2_color}{r_squared:.3f}\033[0m  slope: {np.round(lin_fit[0], 4)}')
            logger.info(f'R² (corrected): {np.round(r_squared, 4)}  slope: {np.round(lin_fit[0], 4)}')
            if log_file:
                log_file.write(f"connectivity_R2: {r_squared:.4f}\n")

            # Connectivity heatmap corrected by slope (for comparison with true)
            # Same format as connectivity_true.png with zoom-in subplot
            connectivity_corrected = pred_weight_corrected / lin_fit[0]
            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(connectivity_corrected, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=weight_lim[0], vmax=weight_lim[1])
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=32)
            plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.xticks(rotation=0)
            plt.subplot(2, 2, 1)
            ax = sns.heatmap(connectivity_corrected[0:20, 0:20], cbar=False, center=0, square=True, cmap='bwr', vmin=weight_lim[0], vmax=weight_lim[1])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/connectivity_learned.tif', dpi=300)
            plt.close()

            # eigenvalue and eigenvector analysis
            print('plot eigenvalue spectrum and eigenvector comparison ...')

            # compute eigenvalues and eigenvectors
            eig_true, V_true = np.linalg.eig(gt_weight)
            eig_learned, V_learned = np.linalg.eig(pred_weight_corrected / lin_fit[0])

            # sort by eigenvalue magnitude
            idx_true = np.argsort(-np.abs(eig_true))
            idx_learned = np.argsort(-np.abs(eig_learned))
            eig_true_sorted = eig_true[idx_true]
            eig_learned_sorted = eig_learned[idx_learned]
            V_true = V_true[:, idx_true]
            V_learned = V_learned[:, idx_learned]

            # left eigenvectors (right eigenvectors of transpose)
            _, L_true = np.linalg.eig(gt_weight.T)
            _, L_learned = np.linalg.eig((pred_weight_corrected / lin_fit[0]).T)
            L_true = L_true[:, idx_true]
            L_learned = L_learned[:, idx_learned]

            # compute alignment matrices
            alignment_R = np.abs(V_true.conj().T @ V_learned)
            alignment_L = np.abs(L_true.conj().T @ L_learned)

            # best alignment score for each true eigenvector
            best_alignment_R = np.max(alignment_R, axis=1)
            best_alignment_L = np.max(alignment_L, axis=1)

            # create 2x3 figure
            fig, axes = plt.subplots(2, 3, figsize=(30, 20))

            # Row 1: Eigenvalues
            # (0,0) complex plane scatter
            axes[0, 0].scatter(eig_true.real, eig_true.imag, s=100, c='green', alpha=0.7, label='true')
            axes[0, 0].scatter(eig_learned.real, eig_learned.imag, s=100, c=mc, alpha=0.7, label='learned')
            axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            axes[0, 0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            axes[0, 0].set_xlabel('real', fontsize=32)
            axes[0, 0].set_ylabel('imag', fontsize=32)
            axes[0, 0].legend(fontsize=20)
            axes[0, 0].tick_params(labelsize=20)
            axes[0, 0].set_title('eigenvalues in complex plane', fontsize=28)

            # (0,1) eigenvalue magnitude comparison (sorted)
            axes[0, 1].scatter(np.abs(eig_true_sorted), np.abs(eig_learned_sorted), s=20, c=mc, alpha=0.7)
            max_val = max(np.abs(eig_true_sorted).max(), np.abs(eig_learned_sorted).max())
            axes[0, 1].plot([0, max_val], [0, max_val], 'g--', linewidth=2)
            axes[0, 1].set_xlabel('true |eigenvalue|', fontsize=32)
            axes[0, 1].set_ylabel('learned |eigenvalue|', fontsize=32)
            axes[0, 1].tick_params(labelsize=20)
            axes[0, 1].set_title('eigenvalue magnitude comparison', fontsize=28)

            # (0,2) singular value spectrum (log scale)
            axes[0, 2].plot(np.abs(eig_true_sorted), color='green', linewidth=2, label='true')
            axes[0, 2].plot(np.abs(eig_learned_sorted), color=mc, linewidth=2, label='learned')
            axes[0, 2].set_xlabel('index', fontsize=32)
            axes[0, 2].set_ylabel('|eigenvalue|', fontsize=32)
            axes[0, 2].set_yscale('log')
            axes[0, 2].legend(fontsize=20)
            axes[0, 2].tick_params(labelsize=20)
            axes[0, 2].set_title('eigenvalue spectrum (log scale)', fontsize=28)

            # Row 2: Eigenvectors
            # (1,0) right eigenvector alignment matrix
            im = axes[1, 0].imshow(alignment_R, cmap='hot', vmin=0, vmax=1)
            axes[1, 0].set_xlabel('learned eigenvector index', fontsize=28)
            axes[1, 0].set_ylabel('true eigenvector index', fontsize=28)
            axes[1, 0].set_title('right eigenvector alignment', fontsize=28)
            axes[1, 0].tick_params(labelsize=16)
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

            # (1,1) left eigenvector alignment matrix
            im_L = axes[1, 1].imshow(alignment_L, cmap='hot', vmin=0, vmax=1)
            axes[1, 1].set_xlabel('learned eigenvector index', fontsize=28)
            axes[1, 1].set_ylabel('true eigenvector index', fontsize=28)
            axes[1, 1].set_title('left eigenvector alignment', fontsize=28)
            axes[1, 1].tick_params(labelsize=16)
            plt.colorbar(im_L, ax=axes[1, 1], fraction=0.046)

            # (1,2) best alignment scores
            axes[1, 2].scatter(range(len(best_alignment_R)), best_alignment_R, s=50, c='b', alpha=0.7, label=f'right (mean={np.mean(best_alignment_R):.2f})')
            axes[1, 2].scatter(range(len(best_alignment_L)), best_alignment_L, s=50, c='r', alpha=0.7, label=f'left (mean={np.mean(best_alignment_L):.2f})')
            axes[1, 2].axhline(y=1/np.sqrt(n_plot), color='gray', linestyle='--', linewidth=2, label=f'random ({1/np.sqrt(n_plot):.2f})')
            axes[1, 2].set_xlabel('eigenvector index (sorted by |eigenvalue|)', fontsize=28)
            axes[1, 2].set_ylabel('best alignment score', fontsize=28)
            axes[1, 2].set_title('best alignment per eigenvector', fontsize=28)
            axes[1, 2].set_ylim([0, 1.05])
            axes[1, 2].legend(fontsize=20)
            axes[1, 2].tick_params(labelsize=16)

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/eigen_comparison.pdf", dpi=87)
            plt.close()

            # spectral radius comparison
            true_spectral_radius = np.max(np.abs(eig_true))
            learned_spectral_radius = np.max(np.abs(eig_learned))
            print(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
            logger.info(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')

            # summary statistics
            mean_align_R = np.mean(best_alignment_R)
            mean_align_L = np.mean(best_alignment_L)
            print(f'eigenvector alignment - right: {mean_align_R:.3f}  left: {mean_align_L:.3f}')
            logger.info(f'eigenvector alignment - right: {mean_align_R:.3f}  left: {mean_align_L:.3f}')

            if has_external_input:

                print('plot field ...')

                # Load second_correction if available
                second_correction_path = f'{log_dir}/second_correction.npy'
                if os.path.exists(second_correction_path):
                    second_correction = float(np.load(second_correction_path))
                else:
                    second_correction = 1.0

                if 'derivative' in external_input_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 400)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid)
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig, ax = fig_init()
                    plt.title(r'$\dot{y_i}$', fontsize=68)

                    plt.imshow(to_numpy(true_derivative))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_field_derivative.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.title(r'learned $\dot{y_i}$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/field_derivative.png", dpi=80)
                    plt.close()


                if ('short_term_plasticity' in external_input_type) | ('modulation_permutation' in external_input_type):

                    for frame in trange(0, n_frames, n_frames // 100, ncols=90):
                        t = torch.tensor([frame/ n_frames], dtype=torch.float32, device=device)
                        if (model_config.update_type == '2steps'):
                                m_ = model_f(t) ** 2
                                m_ = m_[:,None]
                                in_features= torch.cat((torch.zeros_like(m_), torch.ones_like(m_)*xnorm, m_), dim=1)
                                m = model.lin_phi2(in_features)
                        else:
                            m = model_f(t) ** 2

                        if 'permutation' in external_input_type:
                            inverse_permutation_indices = torch.load(f'./graphs_data/{dataset_name}/inverse_permutation_indices.pt', map_location=device)
                            modulation_ = m[inverse_permutation_indices]
                        else:
                            modulation_ = m
                        modulation_ = torch.reshape(modulation_, (32, 32)) * torch.tensor(second_correction, device=device) / 10

                        fig = plt.figure(figsize=(10, 10.5))
                        plt.axis('off')
                        plt.xticks([])
                        plt.xticks([])
                        im_ = to_numpy(modulation_)
                        im_ = np.rot90(im_, k=-1)
                        im_ = np.flipud(im_)
                        im_ = np.fliplr(im_)
                        plt.imshow(im_, cmap='gray')
                        # plt.title(r'neuromodulation $b_i$', fontsize=48)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/xi_{frame}.png", dpi=80)
                        plt.close()

                    fig, ax = fig_init()
                    t = torch.linspace(0, 1, 100000, dtype=torch.float32, device=device).unsqueeze(1)

                    prediction = model_f(t) ** 2
                    prediction = prediction.t()
                    plt.imshow(to_numpy(prediction), aspect='auto')
                    plt.title(r'learned $MLP_2(i,t)$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/learned_plasticity.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.imshow(to_numpy(modulation), aspect='auto') # noqa F821
                    plt.title(r'$y_i$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_plasticity.png", dpi=80)
                    plt.close()

                    prediction = prediction * torch.tensor(second_correction, device=device) / 10

                    fig, ax = fig_init()
                    ids = np.arange(0, 100000, 100).astype(int)
                    plt.scatter(to_numpy(modulation[:, ids]), to_numpy(prediction[:, ids]), s=0.1, color=mc, alpha=0.05) # noqa F821
                    x_data = to_numpy(modulation[:, ids]).flatten() # noqa F821
                    y_data = to_numpy(prediction[:, ids]).flatten()
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.xlabel(r'true $y_i(t)$', fontsize=68)
                    plt.ylabel(r'learned $y_i(t)$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/comparison_yi.png", dpi=80)
                    plt.close()

                else:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])
                    im = imread(f"graphs_data/{config.simulation.node_value_map}")

                    x = x_list[0][0]

                    slope_list = list([])
                    im_list = list([])
                    pred_list = list([])

                    # Ensure field directory exists
                    os.makedirs(f"./{log_dir}/results/field", exist_ok=True)

                    # Compute scale factor from first frame
                    im_first = im[0].squeeze()
                    im_first = np.sqrt(im_first)
                    pred_first = model_f(time=0, enlarge=False) ** 2
                    pred_first = torch.reshape(pred_first, (n_input_neurons_per_axis, n_input_neurons_per_axis))
                    pred_first = to_numpy(torch.sqrt(pred_first))
                    scale_factor = im_first.max() / (pred_first.max() + 1e-8)
                    print(f'[DEBUG field] scale_factor={scale_factor:.4f} mc={mc} n_input_neurons_per_axis={n_input_neurons_per_axis}')

                    for frame in trange(0, n_frames, n_frames // 100, ncols=90):

                        # true field - match training: sqrt + rot90
                        fig, ax = fig_init()
                        im_ = np.zeros((n_input_neurons_per_axis, n_input_neurons_per_axis))
                        if (frame >= 0) & (frame < n_frames):
                            im_ = im[int(frame / n_frames * 256)].squeeze()
                        im_ = np.sqrt(im_)
                        # im_ = np.rot90(im_, k=1)
                        plt.imshow(im_, cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/true_field{epoch}_{frame}.png", dpi=80)
                        plt.close()

                        # reconstructed LR - match training: model_f**2, then sqrt, then rot90, then scale
                        pred = model_f(time=frame / n_frames, enlarge=False) ** 2
                        pred = torch.reshape(pred, (n_input_neurons_per_axis, n_input_neurons_per_axis))
                        pred = to_numpy(torch.sqrt(pred)) * scale_factor
                        if frame == 0:
                            print(f'[DEBUG LR] pred min={pred.min():.4f} max={pred.max():.4f} im_ min={im_.min():.4f} max={im_.max():.4f}')
                        pred = np.rot90(pred, k=1)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_LR {epoch}_{frame}.png", dpi=80)
                        plt.close()

                        x_data = np.reshape(im_, (n_input_neurons_per_axis * n_input_neurons_per_axis))
                        y_data = np.reshape(pred, (n_input_neurons_per_axis * n_input_neurons_per_axis))
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        # print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
                        slope_list.append(lin_fit[0])

                        if frame == 0:
                            print(f'[DEBUG scatter] im_ shape={im_.shape} min={im_.min():.4f} max={im_.max():.4f}')
                            print(f'[DEBUG scatter] pred shape={pred.shape} min={pred.min():.4f} max={pred.max():.4f}')
                            print(f'[DEBUG scatter] x_data range=[{x_data.min():.4f}, {x_data.max():.4f}] y_data range=[{y_data.min():.4f}, {y_data.max():.4f}]')
                            print(f'[DEBUG scatter] R^2={r_squared:.4f} slope={lin_fit[0]:.4f}')
                        fig, ax = fig_init()
                        plt.scatter(im_, pred, s=10, c=mc)
                        plt.xlabel(r'true neuromodulation', fontsize=48)
                        plt.ylabel(r'learned neuromodulation', fontsize=48)
                        ax.text(0.05, 0.95, f'$R^2$: {r_squared:0.2f}  slope: {np.round(lin_fit[0], 2)}',
                                transform=ax.transAxes, fontsize=42, verticalalignment='top')
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/comparison {epoch}_{frame}.png", dpi=80)
                        plt.close()
                        im_list.append(im_)
                        pred_list.append(pred)

                        # reconstructed HR - same processing with scale
                        pred = model_f(time=frame / n_frames, enlarge=True) ** 2
                        pred = torch.reshape(pred, (640, 640))
                        pred = to_numpy(torch.sqrt(pred)) * scale_factor
                        if frame == 0:
                            print(f'[DEBUG HR] pred min={pred.min():.4f} max={pred.max():.4f}')
                        pred = np.rot90(pred, k=1)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_HR {epoch}_{frame}.png", dpi=80)
                        plt.close()

                    im_list = np.array(np.array(im_list))
                    pred_list = np.array(np.array(pred_list))

                    im_list_ = np.reshape(im_list,(100,1024))
                    pred_list_ = np.reshape(pred_list,(100,1024))
                    im_list_ = np.rot90(im_list_)
                    pred_list_ = np.rot90(pred_list_)
                    im_list_ = scipy.ndimage.zoom(im_list_, (1024 / im_list_.shape[0], 1024 / im_list_.shape[1]))
                    pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))

                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)
                    plt.title('true field')
                    plt.imshow(im_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(1, 2, 2)
                    plt.title('reconstructed field')
                    plt.imshow(pred_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(im_list, pred_list, s=1, c=mc, alpha=0.1)
                    plt.xlim([0.3, 1.6])
                    plt.ylim([0.3, 1.6])
                    plt.xlabel(r'true $\Omega_i$', fontsize=68)
                    plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all_comparison {epoch}.png", dpi=80)
                    plt.close()

                    x_data = np.reshape(im_list, (100 * n_input_neurons_per_axis * n_input_neurons_per_axis))
                    y_data = np.reshape(pred_list, (100 * n_input_neurons_per_axis * n_input_neurons_per_axis))
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    print(f'external_input R²: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
                    if log_file is not None:
                        log_file.write(f'external_input_R2: {r_squared:0.4f}\n')
                        log_file.write(f'external_input_slope: {np.round(lin_fit[0], 4)}\n')






            if ('PDE_N6' in model_config.signal_model_name):

                modulation = torch.tensor(x_list[0], device=device)
                modulation = modulation[:, :, 8:9].squeeze()
                modulation = modulation.t()
                modulation = modulation.clone().detach()
                modulation = to_numpy(modulation)

                modulation = scipy.ndimage.zoom(modulation, (1024 / modulation.shape[0], 1024 / modulation.shape[1]))
                pred_list_ = to_numpy(model.b**2)
                pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))

                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.title('true field')
                plt.imshow(modulation, cmap='grey')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(1, 2, 2)
                plt.title('reconstructed field')
                plt.imshow(pred_list_, cmap='grey')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.png", dpi=80)
                plt.close()

                for frame in trange(0, modulation.shape[1], modulation.shape[1] // 257, ncols=90):
                    im = modulation[:, frame]
                    im = np.reshape(im, (32, 32))
                    plt.figure(figsize=(8, 8))
                    plt.axis('off')
                    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/true_field_{frame}.png", dpi=80)
                    plt.close()

            if ('PDE_N5' in model_config.signal_model_name) & (model.update_type == 'generic') & False:

                k = np.random.randint(n_frames - 50)
                x = torch.tensor(x_list[0][k], device=device)
                if has_external_input:
                    if 'visual' in external_input_type:
                        x[:n_input_neurons, 4:5] = model_f(time=k / n_frames) ** 2
                        x[n_input_neurons:n_neurons, 4:5] = 1
                    elif 'learnable_short_term_plasticity' in external_input_type:
                        alpha = (k % model.embedding_step) / model.embedding_step
                        x[:, 4] = alpha * model.b[:, k // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,
                                                                                                         k // model.embedding_step] ** 2
                    elif ('short_term_plasticity' in external_input_type) | ('modulation_permutation' in external_input_type):
                        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                        x[:, 4] = model_f(t) ** 2
                    else:
                        x[:, 4:5] = model_f(time=k / n_frames) ** 2
                else:
                    x[:, 4:5] = torch.ones_like(x[:, 0:1])
                dataset = data.Data(x=x, edge_index=edge_index)
                data_id = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                pred, in_features_ = model(data=dataset, data_id=data_id, return_all=True)
                feature_list = ['u', 'embedding0', 'embedding1', 'msg', 'field']
                for n in range(in_features_.shape[1]):
                    print(f'feature {feature_list[n]}: {to_numpy(torch.mean(in_features_[:, n])):0.4f}  std: {to_numpy(torch.std(in_features_[:, n])):0.4f}')

                fig, ax = fig_init()
                plt.hist(to_numpy(in_features_[:, -1]), 150)
                plt.tight_layout()
                plt.close()

                fig, ax = fig_init()
                f = torch.reshape(x[:n_input_neurons, 4:5], (n_input_neurons_per_axis, n_input_neurons_per_axis))
                f = to_numpy(torch.sqrt(f))
                f = np.rot90(f, k=1)
                plt.imshow(f, cmap='grey')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/field/field_{epoch}.png", dpi=80)
                plt.close()


                fig, ax = fig_init()
                msg_list = []
                u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
                for sample in range(n_neurons):
                    id0 = np.random.randint(0, n_neurons)
                    id1 = np.random.randint(0, n_neurons)
                    f = x[id0, 8:9]
                    embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
                    embedding1 = model.a[id1, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((u[:, None], embedding0, embedding1), dim=1)
                    msg = model.lin_edge(in_features.float()) ** 2 * correction
                    in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msg,
                                             f * torch.ones((400, 1), device=device)), dim=1)
                    plt.plot(to_numpy(u), to_numpy(msg), c=cmap.color(to_numpy(x[id0, 5]).astype(int)), linewidth=2, alpha=0.15)
                    # plt.scatter(to_numpy(u), to_numpy(model.lin_phi(in_features)), s=5, c='r', alpha=0.15)
                    # plt.scatter(to_numpy(u), to_numpy(f*msg), s=1, c='w', alpha=0.1)
                    msg_list.append(msg)
                plt.tight_layout()
                msg_list = torch.stack(msg_list).squeeze()
                y_min, y_max = msg_list.min().item(), msg_list.max().item()
                plt.xlabel(r'$v_i$', fontsize=68)
                plt.ylabel(r'learned $\mathrm{MLP_0}$', fontsize=68)
                plt.ylim([y_min - y_max / 2, y_max * 1.5])
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/learned_multiple_psi_{epoch}.png', dpi=300)
                plt.close()

                fig, ax = fig_init()
                u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
                for n in range(n_neuron_types):
                    for m in range(n_neuron_types):
                        true_func = true_model.func(u, n, m, 'phi')
                        plt.plot(to_numpy(u), to_numpy(true_func), c=cmap.color(n), linewidth=3)
                plt.xlabel(r'$v_i$', fontsize=68)
                plt.ylabel(r'true functions', fontsize=68)
                plt.ylim([y_min - y_max / 2, y_max * 1.5])
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/true_multiple_psi.png', dpi=300)
                plt.close()

                msg_start = torch.mean(in_features_[:, 3]) - torch.std(in_features_[:, 3])
                msg_end = torch.mean(in_features_[:, 3]) + torch.std(in_features_[:, 3])
                msgs = torch.linspace(msg_start, msg_end, 400).to(device)
                fig, ax = fig_init()
                func_list = []
                rr_list = []
                for sample in range(n_neurons):
                    id0 = np.random.randint(0, n_neurons)
                    embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msgs[:,None], torch.ones((400, 1), device=device)), dim=1)
                    pred = model.lin_phi(in_features)
                    plt.plot(to_numpy(msgs), to_numpy(pred), c=cmap.color(to_numpy(x[id0, 5]).astype(int)),  linewidth=2, alpha=0.25)
                    func_list.append(pred)
                    rr_list.append(msgs)
                plt.xlabel(r'$sum_i$', fontsize=68)
                plt.ylabel(r'$\mathrm{MLP_0}(\mathbf{a}_i, x_i=0, sum_i, g_i=1)$', fontsize=48)
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/learned_multivariate_phi_{epoch}.png', dpi=300)
                plt.close()


                print('symbolic regression ...')

                text_trap = StringIO()
                sys.stdout = text_trap

                model_pysrr = PySRRegressor( #noqa: F821
                    niterations=30,  # < Increase me for better results
                    binary_operators=["+", "*"],
                    unary_operators=[
                        "cos",
                        "exp",
                        "sin",
                        "tanh"
                    ],
                    random_state=0,
                    temp_equation_file=False
                )

                # rr_ = torch.rand((4000, 2), device=device)
                # func_ = rr_[:,0] * rr_[:,1]
                # model_pysrr.fit(to_numpy(rr_), to_numpy(func_))
                # model_pysrr.sympy

                func_list = torch.stack(func_list).squeeze()
                rr_list = torch.stack(rr_list).squeeze()
                func = torch.reshape(func_list, (func_list.shape[0] * func_list.shape[1], 1))
                rr = torch.reshape(rr_list, (func_list.shape[0] * func_list.shape[1], 1))
                idx = torch.randperm(len(rr))[:5000]

                model_pysrr.fit(to_numpy(rr[idx]), to_numpy(func[idx]))

                sys.stdout = sys.__stdout__

            if False:
                print ('symbolic regression ...')

                def get_pyssr_function(model_pysrr, rr, func):

                    text_trap = StringIO()
                    sys.stdout = text_trap

                    model_pysrr.fit(to_numpy(rr[:, None]), to_numpy(func[:, None]))

                    sys.stdout = sys.__stdout__

                    return model_pysrr.sympy

                model_pysrr = PySRRegressor( # noqa: F821
                    niterations=30,  # < Increase me for better results
                    binary_operators=["+", "*"],
                    unary_operators=[
                        "cos",
                        "exp",
                        "sin",
                        "tanh"
                    ],
                    random_state=0,
                    temp_equation_file=False
                )

                match model_config.signal_model_name:

                    case 'PDE_N2':

                        func = torch.mean(psi_list, dim=0).squeeze() # noqa: F821

                        symbolic = get_pyssr_function(model_pysrr, rr, func)

                        for n in range(0,7):
                            print(symbolic(n))
                            logger.info(symbolic(n))

                    case 'PDE_N4':

                        for k in range(n_neuron_types):
                            print('  ')
                            print('  ')
                            print('  ')
                            print(f'psi{k} ................')
                            logger.info(f'psi{k} ................')

                            pos = np.argwhere(labels == k)
                            pos = pos.squeeze()

                            func = psi_list[pos] # noqa: F821
                            func = torch.mean(psi_list[pos], dim=0) # noqa: F821

                            symbolic = get_pyssr_function(model_pysrr, rr, func)

                            # for n in range(0, 5):
                            #     print(symbolic(n))
                            #     logger.info(symbolic(n))

                    case 'PDE_N5':

                        for k in range(4**2):

                            print('  ')
                            print('  ')
                            print('  ')
                            print(f'psi {k//4} {k%4}................')
                            logger.info(f'psi {k//4} {k%4} ................')

                            pos =np.arange(k*250,(k+1)*250)
                            func = psi_list[pos] # noqa: F821
                            func = torch.mean(psi_list[pos], dim=0) # noqa: F821

                            symbolic = get_pyssr_function(model_pysrr, rr, func)

                            # for n in range(0, 7):
                            #     print(symbolic(n))
                            #     logger.info(symbolic(n))

                for k in range(n_neuron_types):
                    print('  ')
                    print('  ')
                    print('  ')
                    print(f'phi{k} ................')
                    logger.info(f'phi{k} ................')

                    pos = np.argwhere(labels == k)
                    pos = pos.squeeze()

                    func = phi_list[pos]
                    func = torch.mean(phi_list[pos], dim=0)

                    symbolic = get_pyssr_function(model_pysrr, rr, func)

                    # for n in range(4, 7):
                    #     print(symbolic(n))
                    #     logger.info(symbolic(n))



def analyze_neuron_type_reconstruction(config, model, edges, true_weights, gt_taus, gt_V_Rest,
                                       learned_weights, learned_tau, learned_V_rest, type_list, n_frames, dimension,
                                       n_neuron_types, device, log_dir, dataset_name, logger, index_to_name):

    print('stratified analysis by neuron type...')

    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    rmse_weights = []
    rmse_taus = []
    rmse_vrests = []
    n_connections = []

    for neuron_type in range(n_neuron_types):

        type_indices = np.where(type_list[edges[1,:]] == neuron_type)[0]
        gt_w_type = true_weights[type_indices]
        learned_w_type = learned_weights[type_indices]
        n_conn = len(type_indices)

        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        gt_vrest_type = gt_V_Rest[type_indices]

        learned_tau_type = learned_tau[type_indices]
        learned_vrest_type = learned_V_rest[type_indices]

        rmse_w = np.sqrt(np.mean((gt_w_type - learned_w_type)** 2))
        rmse_tau = np.sqrt(np.mean((gt_tau_type - learned_tau_type)** 2))
        rmse_vrest = np.sqrt(np.mean((gt_vrest_type - learned_vrest_type)** 2))

        rmse_weights.append(rmse_w)
        rmse_taus.append(rmse_tau)
        rmse_vrests.append(rmse_vrest)
        n_connections.append(n_conn)

    n_neurons = len(type_list)

    # Per-neuron RMSE for tau
    rmse_tau_per_neuron = np.abs(learned_tau - gt_taus)
    # Per-neuron RMSE for V_rest
    rmse_vrest_per_neuron = np.abs(learned_V_rest - gt_V_Rest)
    # Per-neuron RMSE for weights (incoming connections)
    rmse_weights_per_neuron = np.zeros(n_neurons)
    for neuron_idx in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == neuron_idx)[0]
        if len(incoming_edges) > 0:
            true_w = true_weights[incoming_edges]
            learned_w = learned_weights[incoming_edges]
            rmse_weights_per_neuron[neuron_idx] = np.sqrt(np.mean((learned_w - true_w)**2))

    # Convert to arrays
    rmse_weights = np.array(rmse_weights)
    rmse_taus = np.array(rmse_taus)
    rmse_vrests = np.array(rmse_vrests)

    unique_types_in_order = []
    seen_types = set()
    for i in range(len(type_list)):
        neuron_type_id = type_list[i].item() if hasattr(type_list[i], 'item') else int(type_list[i])
        if neuron_type_id not in seen_types:
            unique_types_in_order.append(neuron_type_id)
            seen_types.add(neuron_type_id)

    # Create neuron type names in the same order as they appear in data
    sorted_neuron_type_names = [index_to_name.get(type_id, f'Type{type_id}') for type_id in unique_types_in_order]
    unique_types_in_order = np.array(unique_types_in_order)
    sort_indices = unique_types_in_order.astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    x_pos = np.arange(len(sort_indices))

    # Plot weights RMSE
    ax1 = axes[0]
    ax1.bar(x_pos, rmse_weights[sort_indices], color='skyblue', alpha=0.7)
    ax1.set_ylabel('RMSE weights', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 2.5])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax1.grid(False)
    ax1.tick_params(axis='y', labelsize=12)

    for i, (tick, rmse_w) in enumerate(zip(ax1.get_xticklabels(), rmse_weights[sort_indices])):
        if rmse_w > 0.5:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 2 (tau)
    ax2 = axes[1]
    ax2.bar(x_pos, rmse_taus[sort_indices], color='lightcoral', alpha=0.7)
    ax2.set_ylabel(r'RMSE $\tau$', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.3])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax2.grid(False)
    ax2.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth taus per neuron type
    mean_gt_taus = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        mean_gt_taus.append(np.mean(np.abs(gt_tau_type)))

    mean_gt_taus = np.array(mean_gt_taus)

    for i, (tick, rmse_tau) in enumerate(zip(ax2.get_xticklabels(), rmse_taus[sort_indices])):
        if rmse_tau > 0.03:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 3 (V_rest)
    ax3 = axes[2]
    ax3.bar(x_pos, rmse_vrests[sort_indices], color='lightgreen', alpha=0.7)
    ax3.set_ylabel(r'RMSE $V_{rest}$', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.8])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax3.grid(False)
    ax3.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth V_rest per neuron type
    mean_gt_vrests = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_vrest_type = gt_V_Rest[type_indices]
        mean_gt_vrests.append(np.mean(np.abs(gt_vrest_type)))

    mean_gt_vrests = np.array(mean_gt_vrests)
    for i, (tick, rmse_vrest) in enumerate(zip(ax3.get_xticklabels(), rmse_vrests[sort_indices])):
        if rmse_vrest > 0.08:
            tick.set_color('red')
            tick.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(f'./{log_dir}/results/neuron_type_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Log summary statistics
    logger.info(f"mean weights RMSE: {np.mean(rmse_weights):.3f} ± {np.std(rmse_weights):.3f}")
    logger.info(f"mean tau RMSE: {np.mean(rmse_taus):.3f} ± {np.std(rmse_taus):.3f}")
    logger.info(f"mean V_rest RMSE: {np.mean(rmse_vrests):.3f} ± {np.std(rmse_vrests):.3f}")

    # Return per-neuron results (NEW)
    return {
        'rmse_weights_per_neuron': rmse_weights_per_neuron,
        'rmse_tau_per_neuron': rmse_tau_per_neuron,
        'rmse_vrest_per_neuron': rmse_vrest_per_neuron,
        'rmse_weights_per_type': rmse_weights,
        'rmse_tau_per_type': rmse_taus,
        'rmse_vrest_per_type': rmse_vrests
    }


def plot_ising_comparison_from_saved(config_list, labels=None, output_path='fig/ising_noise_comparison.png'):
    valid_configs = []

    for config_file_ in config_list:
        try:
            config_file, pre_folder = add_pre_folder(config_file_)
            data_path = f'./log/{config_file}/results/info_ratio_results.npz'
            if not os.path.exists(data_path):
                print(f"Warning: {data_path} not found")
                continue
            valid_configs.append(config_file)
        except Exception as e:
            print(f"Error processing {config_file_}: {e}")
            continue

    if len(valid_configs) != 3:
        raise ValueError(f"Need exactly 3 valid configs, got {len(valid_configs)}")

    # Default sigma labels with lowercase noise descriptions
    if labels is None:
        labels = [r'$\sigma=10^{-6}$ (low noise)',
                 r'$\sigma=0.25$ (moderate noise)',
                 r'$\sigma=2.5$ (large noise)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.4)

    for col, (config_file, label) in enumerate(zip(valid_configs, labels)):
        data_path = f'./log/{config_file}/results/info_ratio_results.npz'
        data = np.load(data_path)

        obs_rates = data['observed_rates'].flatten()
        pair_rates = data['predicted_rates_pairwise'].flatten()
        indep_rates = data['predicted_rates_independent'].flatten()

        # Rate scatter plots
        ax = axes[col]

        # Plot data
        ax.loglog(obs_rates, pair_rates, 'r.', alpha=0.1, markersize=1.5)
        ax.loglog(obs_rates, indep_rates, 'g.', alpha=0.1, markersize=1.5)
        ax.plot([1e-4, 1e1], [1e-4, 1e1], 'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel(r'observed rate (s$^{-1}$)', fontsize=18)
        ax.set_ylabel(r'predicted rate (s$^{-1}$)' if col == 0 else '', fontsize=18)
        ax.set_title(f'{label}\n$I_N$={data["I_N_median"]:.2f} bits', fontsize=16)
        ax.set_xlim(1e-4, 1e1)
        ax.set_ylim(1e-4, 1e1)
        ax.tick_params(labelsize=14)

        # Legend only on the first panel - top left
        if col == 0:
            red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=1.0)
            green_patch = plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=1.0)
            ax.legend([red_patch, green_patch], ['pairwise ($P_2$)', 'independent ($P_1$)'],
                     loc='upper left', fontsize=14, frameon=False)

    # Add panel labels - moved upwards
    for i, ax in enumerate(axes):
        ax.text(-0.15, 1.15, f'{chr(97+i)})', transform=ax.transAxes, fontsize=20, va='top', ha='left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.show()





def compare_gnn_results(config_list, varied_parameter):
    """
    Compare GNN experiments by reading config files and results.log files
    Focuses on: weights/tau/V_rest R², clustering accuracy, loss curves
    """

    # Global style
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['text.color'] = 'white'

    plt.style.use('default')

    results = []

    # Read & parse per-config files
    for config_file_ in config_list:
        try:
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')

            # Resolve varied parameter value
            if varied_parameter is None:
                parts = config_file_.split('_')
                if len(parts) >= 2:
                    param_value = parts[-1]
                else:
                    print(f"warning: cannot extract indices from config name '{config_file_}'")
                    continue
            else:
                if '.' not in varied_parameter:
                    raise ValueError("parameter must be in 'section.parameter' format")
                section_name, param_name = varied_parameter.split('.', 1)
                section = getattr(config, section_name, None)
                if section is None:
                    raise ValueError(f"config section '{section_name}' not found")
                param_value = getattr(section, param_name, None)
                if param_value is None:
                    print(f"warning: parameter '{param_name}' not found")
                    continue

            # Parse results.log
            results_log_path = os.path.join('./log', config_file, 'results.log')
            if not os.path.exists(results_log_path):
                print(f"warning: {results_log_path} not found")
                continue

            with open(results_log_path, 'r') as f:
                content = f.read()

            # Parse standard metrics
            r2_match = re.search(r'second weights fit\s+R²:\s*([\d.-]+)', content)
            r2 = float(r2_match.group(1)) if r2_match else None

            tau_r2_match = re.search(r'tau reconstruction R²:\s*([\d.-]+)', content)
            tau_r2 = float(tau_r2_match.group(1)) if tau_r2_match else None

            vrest_r2_match = re.search(r'V_rest reconstruction R²:\s*([\d.-]+)', content)
            vrest_r2 = float(vrest_r2_match.group(1)) if vrest_r2_match else None

            acc_match = re.search(r'accuracy=([\d.-]+)', content)
            best_clustering_acc = float(acc_match.group(1)) if acc_match else None

            results.append({
                'config': config_file_,
                'param_value': param_value,
                'r2': r2,
                'tau_r2': tau_r2,
                'vrest_r2': vrest_r2,
                'best_clustering_acc': best_clustering_acc,
            })

        except Exception as e:
            print(f"error processing {config_file_}: {e}")

    # Group by parameter value
    grouped_results = defaultdict(list)
    for r in results:
        if all(r[k] is not None for k in ['r2', 'tau_r2', 'vrest_r2', 'best_clustering_acc']):
            grouped_results[r['param_value']].append(r)

    # Aggregate into summary
    summary_results = []
    for param_val, group in grouped_results.items():
        r2_values = [r['r2'] for r in group]
        tau_r2_values = [r['tau_r2'] for r in group]
        vrest_r2_values = [r['vrest_r2'] for r in group]
        acc_values = [r['best_clustering_acc'] for r in group]

        summary_results.append({
            'param_value': param_val,
            'r2_mean': np.mean(r2_values),
            'tau_r2_mean': np.mean(tau_r2_values),
            'vrest_r2_mean': np.mean(vrest_r2_values),
            'acc_mean': np.mean(acc_values),
            'r2_std': np.std(r2_values) if len(r2_values) > 1 else 0,
            'tau_r2_std': np.std(tau_r2_values) if len(tau_r2_values) > 1 else 0,
            'vrest_r2_std': np.std(vrest_r2_values) if len(vrest_r2_values) > 1 else 0,
            'acc_std': np.std(acc_values) if len(acc_values) > 1 else 0,
            'n_configs': len(group)
        })

    # Sort results
    summary_results.sort(key=lambda x: x['param_value'])

    # Display parameter name
    if varied_parameter is None:
        param_display_name = "config_indices"
    else:
        param_display_name = varied_parameter.split('.')[1]

    # Print summary table
    print("-" * 75)
    print(f"{'parameter':<15} {'weights R²':<15} {'tau R²':<15} {'V_rest R²':<15} {'clustering':<15}")
    print("-" * 75)

    for r in summary_results:
        r2_str = f"{r['r2_mean']:.3f}±{r['r2_std']:.3f}" if r['r2_std'] > 0 else f"{r['r2_mean']:.3f}"
        tau_str = f"{r['tau_r2_mean']:.3f}±{r['tau_r2_std']:.3f}" if r['tau_r2_std'] > 0 else f"{r['tau_r2_mean']:.3f}"
        vrest_str = f"{r['vrest_r2_mean']:.3f}±{r['vrest_r2_std']:.3f}" if r['vrest_r2_std'] > 0 else f"{r['vrest_r2_mean']:.3f}"
        acc_str = f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}" if r['acc_std'] > 0 else f"{r['acc_mean']:.3f}"

        print(f"{str(r['param_value']):<15} {r2_str:<15} {tau_str:<15} {vrest_str:<15} {acc_str:<15}")


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.10, wspace=0.32, hspace=0.45)
    ax1.text(-0.08, 1.05, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')
    ax2.text(-0.08, 1.05, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')
    ax3.text(-0.08, 1.05, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')
    ax4.text(-0.08, 1.05, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

    param_values = [r['param_value'] for r in summary_results]
    # param_display_name = 'noise level'

    x_values = [float(p) for p in param_values]
    use_log = min(x_values) > 0 and max(x_values)/min(x_values) > 10 and varied_parameter != None

    for ax, ydata, label, color in [
        (ax1, [r['r2_mean'] for r in summary_results], 'weights R²', 'blue'),
        (ax2, [r['tau_r2_mean'] for r in summary_results], 'tau R²', 'green'),
        (ax3, [r['vrest_r2_mean'] for r in summary_results], 'V_rest R²', 'orange'),
        (ax4, [r['acc_mean'] for r in summary_results], 'clustering accuracy', 'red'),
    ]:

        if use_log:
            plot_fn = ax.semilogx
            plot_fn(x_values, ydata, 'o', color=color, linewidth=2, markersize=8)
            ax.set_xlim(left=0, right=5)
        else:
            plot_fn = ax.plot
            plot_fn(x_values, ydata, 'o', color=color, linewidth=2, markersize=8)

        ax.set_xlabel(param_display_name, fontsize=18)

        if label == 'clustering accuracy':
            ax.set_ylabel('classification accuracy', fontsize=18)
        elif label == 'weights R²':
            ax.set_ylabel(r'learned $W_{ij}\quad R²$', fontsize=18)
        elif label == 'tau R²':
            ax.set_ylabel(r'learned $\tau_i \quad R^2$', fontsize=18)
        elif label == 'V_rest R²':
            ax.set_ylabel(r'learned $V^{rest}_i\quad R^2$', fontsize=18)
        ax.set_ylim(0, 1.1)
        if use_log:
            ax.set_xscale('log')
        # ax.set_xlim(left=1e-6, right=1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    sigma_labels = [r"$\sigma=1E-6$", r"$\sigma=0.25$", r"$\sigma=2.5$"]
    if not use_log and len(x_values) == 3:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks(x_values)
            ax.set_xticklabels(sigma_labels, fontsize=18)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.tick_params(axis='both', which='major', labelsize=10)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)

    plt.tight_layout()
    plt.savefig(f'fig/gnn_comparison_{param_display_name}.png', dpi=400, bbox_inches='tight')
    plt.show()

    return summary_results


def collect_gnn_results_multimodel(config_list, varied_parameter=None):
    """
    Collect per-model GNN experiment metrics (parameter reconstruction +
    rollout quality) for runs launched via GNN_Main_multimodel.py.

    This version does NOT plot anything; it just returns a tidy pandas
    DataFrame where each row corresponds to one (config, model_id).

    Parameters
    ----------
    config_list : list[str]
        Base config names, e.g. ['fly_N9_22_10', 'fly_N9_44_24'].
        These are the same strings you pass to GNN_Main_multimodel.py.
    varied_parameter : str or None
        Optional 'section.parameter' name inside NeuralGraphConfig to treat
        as the varied hyperparameter (e.g. 'training.noise_model_level').
        If None, falls back to using the trailing token in the config name.
    log_root : str
        Root directory containing log folders (default: './log').

    Returns
    -------
    pandas.DataFrame
        Columns include:
        - 'base_config', 'config_file', 'pre_folder', 'model_id'
        - 'param_value'
        - 'weights_r2', 'tau_r2', 'vrest_r2', 'clustering_acc'
        - rollout metrics:
          'rollout_rmse_mean', 'rollout_rmse_std', 'rollout_rmse_min', 'rollout_rmse_max'
          'rollout_pearson_mean', 'rollout_pearson_std', 'rollout_pearson_min', 'rollout_pearson_max'
          'rollout_r2_mean', 'rollout_r2_std', 'rollout_r2_min', 'rollout_r2_max'
          'rollout_feve_mean', 'rollout_feve_std', 'rollout_feve_min', 'rollout_feve_max'
          'rollout_n_neurons', 'rollout_start_frame', 'rollout_end_frame'
    """
    records = []

    this_file_parent = os.path.dirname(__file__)
    log_root = os.path.join(this_file_parent, "log")
    config_root = os.path.join(this_file_parent, "config")

    for base_config in config_list:
        try:
            config_file, pre_folder = add_pre_folder(base_config)
            config = NeuralGraphConfig.from_yaml(f'{config_root}/{config_file}.yaml')
        except Exception as e:
            print(f"error loading base config '{base_config}': {e}")
            continue

        # Determine parameter value for this base config
        if varied_parameter is None:
            parts = base_config.split('_')
            if len(parts) >= 2:
                param_value = parts[-1]
            else:
                print(f"warning: cannot extract indices from config name '{base_config}'")
                param_value = None
        else:
            if '.' not in varied_parameter:
                raise ValueError("parameter must be in 'section.parameter' format")
            section_name, param_name = varied_parameter.split('.', 1)
            section = getattr(config, section_name, None)
            if section is None:
                print(f"warning: config section '{section_name}' not found in '{base_config}'")
                param_value = None
            else:
                param_value = getattr(section, param_name, None)
                if param_value is None:
                    print(f"warning: parameter '{param_name}' not found in section '{section_name}' for '{base_config}'")

        # Discover per-model log directories created by GNN_Main_multimodel
        pre_log_root = os.path.join(log_root, pre_folder)
        if not os.path.isdir(pre_log_root):
            print(f"warning: log root '{pre_log_root}' not found for base config '{base_config}'")
            continue

        for entry in os.listdir(pre_log_root):
            # Expect entries like 'fly_N9_22_10__mid_000'
            if not entry.startswith(base_config + "__mid_"):
                continue

            model_id_str = entry.split("__mid_")[-1]
            try:
                model_id = int(model_id_str)
            except ValueError:
                model_id = None

            config_file_with_mid = os.path.join(pre_folder, entry)
            log_dir = os.path.join(log_root, pre_folder, entry)

            results_log_path = os.path.join(log_dir, "results.log")
            rollout_log_path = os.path.join(log_dir, "results_rollout.log")

            if not os.path.exists(results_log_path):
                print(f"warning: {results_log_path} not found, skipping")
                continue
            if not os.path.exists(rollout_log_path):
                print(f"warning: {rollout_log_path} not found, skipping")
                continue

            # --- Parse results.log (parameter reconstructions, clustering) ---
            try:
                with open(results_log_path, "r") as f:
                    content = f.read()
            except Exception as e:
                print(f"error reading {results_log_path}: {e}")
                continue

            r2_match = re.search(r'second weights fit\s+R²:\s*([\d.eE+-]+)', content)
            tau_r2_match = re.search(r'tau reconstruction R²:\s*([\d.eE+-]+)', content)
            vrest_r2_match = re.search(r'V_rest reconstruction R²:\s*([\d.eE+-]+)', content)
            acc_match = re.search(r'accuracy=([\d.eE+-]+)', content)

            weights_r2 = float(r2_match.group(1)) if r2_match else None
            tau_r2 = float(tau_r2_match.group(1)) if tau_r2_match else None
            vrest_r2 = float(vrest_r2_match.group(1)) if vrest_r2_match else None
            clustering_acc = float(acc_match.group(1)) if acc_match else None

            # --- Parse results_rollout.log (rollout metrics) ---
            try:
                with open(rollout_log_path, "r") as f:
                    r_content = f.read()
            except Exception as e:
                print(f"error reading {rollout_log_path}: {e}")
                continue

            def _parse_metric(line_prefix, text):
                # Example line:
                # RMSE: 0.0066 ± 0.0052 [0.0002, 0.0478]
                pat = rf'{line_prefix}:\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)\s*\[([\d.eE+-]+),\s*([\d.eE+-]+)\]'
                m = re.search(pat, text)
                if not m:
                    return (None, None, None, None)
                return tuple(float(g) for g in m.groups())

            rmse_mean, rmse_std, rmse_min, rmse_max = _parse_metric("RMSE", r_content)
            pearson_mean, pearson_std, pearson_min, pearson_max = _parse_metric("Pearson r", r_content)
            r2_mean, r2_std, r2_min, r2_max = _parse_metric("R²", r_content)
            feve_mean, feve_std, feve_min, feve_max = _parse_metric("FEVE", r_content)

            n_neurons = None
            start_frame = None
            end_frame = None

            m_neurons = re.search(r"Number of neurons evaluated:\s*(\d+)", r_content)
            if m_neurons:
                n_neurons = int(m_neurons.group(1))

            m_frames = re.search(r"Frames evaluated:\s*(\d+)\s*to\s*(\d+)", r_content)
            if m_frames:
                start_frame = int(m_frames.group(1))
                end_frame = int(m_frames.group(2))

            # --- Compute per-neuron RMSEs and correlations from saved rollout traces ---
            rollout_rmse_per_neuron = None
            rollout_rmse_median = None
            rollout_corr_per_neuron = None
            rollout_corr_median = None
            try:
                results_dir = os.path.join(log_dir, "results")
                # Prefer modified activity arrays; fall back to standard if present
                act_mod = os.path.join(results_dir, "activity_modified.npy")
                act_mod_pred = os.path.join(results_dir, "activity_modified_pred.npy")
                act_true = os.path.join(results_dir, "activity_true.npy")
                act_pred = os.path.join(results_dir, "activity_pred.npy")

                if os.path.exists(act_mod) and os.path.exists(act_mod_pred):
                    act_path_true, act_path_pred = act_mod, act_mod_pred
                elif os.path.exists(act_true) and os.path.exists(act_pred):
                    act_path_true, act_path_pred = act_true, act_pred
                else:
                    act_path_true = act_path_pred = None

                if act_path_true is not None:
                    a_true = np.load(act_path_true)
                    a_pred = np.load(act_path_pred)
                    if a_true.shape == a_pred.shape and a_true.ndim == 2:
                        # Expect shape (n_neurons, n_frames)
                        diff = a_true - a_pred
                        mse_per_neuron = np.mean(diff ** 2, axis=1)
                        rmse_per_neuron = np.sqrt(mse_per_neuron)
                        rollout_rmse_per_neuron = rmse_per_neuron.tolist()
                        rollout_rmse_median = float(np.median(rmse_per_neuron))

                        # Vectorized per-neuron Pearson correlation over time
                        # Center each neuron's trace
                        x = a_true
                        y = a_pred
                        x_mean = x.mean(axis=1, keepdims=True)
                        y_mean = y.mean(axis=1, keepdims=True)
                        x_centered = x - x_mean
                        y_centered = y - y_mean

                        # Standard deviations per neuron
                        x_std = x_centered.std(axis=1, ddof=0)
                        y_std = y_centered.std(axis=1, ddof=0)

                        # Mask neurons with zero variance in either trace
                        valid = (x_std > 0) & (y_std > 0)
                        corr = np.full(x_std.shape, np.nan, dtype=float)
                        if np.any(valid):
                            x_norm = x_centered[valid] / x_std[valid, None]
                            y_norm = y_centered[valid] / y_std[valid, None]
                            corr_valid = np.mean(x_norm * y_norm, axis=1)
                            corr[valid] = corr_valid

                        rollout_corr_per_neuron = corr.tolist()
                        rollout_corr_median = float(np.nanmedian(corr))
            except Exception as e:
                print(f"warning: could not compute per-neuron RMSE for '{log_dir}': {e}")

            records.append(
                {
                    "base_config": base_config,
                    "config_file": config_file_with_mid,
                    "pre_folder": pre_folder,
                    "model_id": model_id,
                    "param_value": param_value,
                    # Parameter reconstruction metrics
                    "weights_r2": weights_r2,
                    "tau_r2": tau_r2,
                    "vrest_r2": vrest_r2,
                    "clustering_acc": clustering_acc,
                    # Rollout metrics
                    "rollout_rmse_mean": rmse_mean,
                    "rollout_rmse_std": rmse_std,
                    "rollout_rmse_min": rmse_min,
                    "rollout_rmse_max": rmse_max,
                    "rollout_pearson_mean": pearson_mean,
                    "rollout_pearson_std": pearson_std,
                    "rollout_pearson_min": pearson_min,
                    "rollout_pearson_max": pearson_max,
                    "rollout_r2_mean": r2_mean,
                    "rollout_r2_std": r2_std,
                    "rollout_r2_min": r2_min,
                    "rollout_r2_max": r2_max,
                    "rollout_feve_mean": feve_mean,
                    "rollout_feve_std": feve_std,
                    "rollout_feve_min": feve_min,
                    "rollout_feve_max": feve_max,
                    "rollout_n_neurons": n_neurons,
                    "rollout_start_frame": start_frame,
                    "rollout_end_frame": end_frame,
                    "rollout_rmse_per_neuron": rollout_rmse_per_neuron,
                    "rollout_rmse_median": rollout_rmse_median,
                    "rollout_corr_per_neuron": rollout_corr_per_neuron,
                    "rollout_corr_median": rollout_corr_median,
                }
            )

    if len(records) == 0:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    return df


def compare_ising_results(config_list, varied_parameter):
    """
    Compare Ising/information theory metrics across experiments
    Focuses on: I_N, I_2, I_2/I_N ratio, higher-order correlations
    """

    from NeuralGraph.config import NeuralGraphConfig
    from NeuralGraph.models.utils import add_pre_folder

    def parse_ising_metrics(content):
        """Extract Ising metrics from results.log content"""
        metrics = {}

        # Parse I_N
        match = re.search(r'I_N:\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['I_N_median'] = float(match.group(1))
            metrics['I_N_q1'] = float(match.group(2))
            metrics['I_N_q3'] = float(match.group(3))

        # Parse I2
        match = re.search(r'I2:\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['I2_median'] = float(match.group(1))
            metrics['I2_q1'] = float(match.group(2))
            metrics['I2_q3'] = float(match.group(3))

        # Parse I_HOC
        match = re.search(r'I_HOC:\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['I_HOC_median'] = float(match.group(1))
            metrics['I_HOC_q1'] = float(match.group(2))
            metrics['I_HOC_q3'] = float(match.group(3))

        # Parse ratio
        match = re.search(r'(?:ratio|I_2/I_N):\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['ratio_median'] = float(match.group(1))
            metrics['ratio_q1'] = float(match.group(2))
            metrics['ratio_q3'] = float(match.group(3))

        # Parse C_3 (connected triplets)
        match = re.search(r'C_3:\s*(?:mean=)?([0-9eE+\-\.]+)\s*±\s*([0-9eE+\-\.]+)', content)
        if match:
            metrics['C3_mean'] = float(match.group(1))
            metrics['C3_std'] = float(match.group(2))

        return metrics

    results = []

    # Parse each config
    for config_file_ in config_list:
        try:
            config_file, _ = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')

            # Get parameter value
            if varied_parameter is None:
                parts = config_file_.split('_')
                param_value = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else config_file_
            else:
                section_name, param_name = varied_parameter.split('.', 1)
                section = getattr(config, section_name, None)
                param_value = getattr(section, param_name, config_file_) if section else config_file_

            # Read results.log
            results_log_path = os.path.join('./log', config_file, 'results.log')
            if not os.path.exists(results_log_path):
                continue

            with open(results_log_path, 'r') as f:
                content = f.read()

            # Parse Ising metrics
            metrics = parse_ising_metrics(content)
            if metrics:
                metrics['param_value'] = param_value
                metrics['config'] = config_file_
                results.append(metrics)

        except Exception as e:
            print(f"Error processing {config_file_}: {e}")

    if not results:
        # print("No Ising metrics found")
        return None

    # Sort by parameter value
    try:
        results.sort(key=lambda x: float(x['param_value']))
    except:
        results.sort(key=lambda x: str(x['param_value']))

    # Display name
    if varied_parameter is None:
        param_display_name = "config"
    else:
        param_display_name = varied_parameter.split('.')[1]

    # Print summary table
    print(f"\n=== Ising Analysis Comparison: {param_display_name} ===")
    print(f"{'Parameter':<12} {'I_N (bits)':<15} {'I_2 (bits)':<15} {'I_HOC (bits)':<15} {'I_2/I_N':<12} {'C_3':<15}")
    print("-" * 85)

    for r in results:
        i_n = f"{r.get('I_N_median', 0):.3f}" if 'I_N_median' in r else "N/A"
        i_2 = f"{r.get('I2_median', 0):.3f}" if 'I2_median' in r else "N/A"
        i_hoc = f"{r.get('I_HOC_median', 0):.3f}" if 'I_HOC_median' in r else "N/A"
        ratio = f"{r.get('ratio_median', 0):.3f}" if 'ratio_median' in r else "N/A"
        c3 = f"{r.get('C3_mean', 0):.4f}±{r.get('C3_std', 0):.4f}" if 'C3_mean' in r else "N/A"

        print(f"{str(r['param_value']):<12} {i_n:<15} {i_2:<15} {i_hoc:<15} {ratio:<12} {c3:<15}")

    # Create plots
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Add panel labels a), b), c), d)
    axes[0, 0].text(-0.08, 1.05, 'a)', transform=axes[0, 0].transAxes, fontsize=18, va='top', ha='right')
    axes[0, 1].text(-0.08, 1.05, 'b)', transform=axes[0, 1].transAxes, fontsize=18, va='top', ha='right')
    axes[1, 0].text(-0.08, 1.05, 'c)', transform=axes[1, 0].transAxes, fontsize=18, va='top', ha='right')
    # axes[1, 1].text(-0.15, 1.08, 'd)', transform=axes[1, 1].transAxes, fontsize=18, va='top', ha='right')

    param_values = [r['param_value'] for r in results]
    param_display_name = 'noise level'

    # Try to convert to numeric for better plotting
    try:
        x_values = [float(p) for p in param_values]
        use_log = min(x_values) > 0 and max(x_values)/min(x_values) > 10
    except:
        x_values = range(len(param_values))
        use_log = False

    # Panel 1: I_N and I_2
    ax = axes[0, 0]
    if 'I_N_median' in results[0]:
        i_n_vals = [r.get('I_N_median', 0) for r in results]
        i_2_vals = [r.get('I2_median', 0) for r in results]

        plot_fn = ax.semilogx if use_log else ax.plot
        plot_fn(x_values, i_n_vals, 'o', label='$I_N$', linewidth=2, markersize=8)
        plot_fn(x_values, i_2_vals, 's', label='$I^{(2)}$', linewidth=2, markersize=8)
        # Add shaded region for sigma in [0.25, right edge]
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)
        ax.set_xlabel(param_display_name, fontsize=18)
        ax.set_ylabel('information (bits)', fontsize=18)
        ax.legend(fontsize=18)
        ax.set_xlim(left=0, right=5)
        # ax.grid(True, alpha=0.3)

    # Panel 2: I_2/I_N ratio
    ax = axes[0, 1]
    if 'ratio_median' in results[0]:
        ratio_vals = [r.get('ratio_median', 0) for r in results]
        plot_fn = ax.semilogx if use_log else ax.plot
        plot_fn(x_values, ratio_vals, 'o', color='green', linewidth=2, markersize=8)
        # Add shaded region for sigma in [0.25, right edge]
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)
        ax.set_xlabel(param_display_name, fontsize=18)
        ax.set_ylabel('$I^{(2)}/I_N$', fontsize=18)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(left=0, right=5)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        # ax.grid(True, alpha=0.3)

    # Panel 3: Higher-order correlation
    ax = axes[1, 0]
    if 'I_HOC_median' in results[0]:
        i_hoc_vals = [r.get('I_HOC_median', 0) for r in results]
        plot_fn = ax.semilogx if use_log else ax.plot
        plot_fn(x_values, i_hoc_vals, 'o', color='purple', linewidth=2, markersize=8)
        # Add shaded region for sigma in [0.25, right edge]
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)
        ax.set_xlabel(param_display_name, fontsize=18)
        ax.set_ylabel('$I_{HOC}$ (bits)', fontsize=18)
        ax.set_xlim(left=0, right=5)
        # ax.grid(True, alpha=0.3)

    # Panel 4: Connected triplets C_3
    ax = axes[1, 1]
    # if 'C3_mean' in results[0]:
    #     c3_vals = [r.get('C3_mean', 0) for r in results]
    #     c3_err = [r.get('C3_std', 0) for r in results]

    #     if use_log:
    #         ax.errorbar(x_values, c3_vals, yerr=c3_err, fmt='o', color='teal',
    #                    linewidth=2, markersize=8, capsize=5)
    #         ax.set_xscale('log')
    #     else:
    #         ax.errorbar(x_values, c3_vals, yerr=c3_err, fmt='o', color='teal',
    #                    linewidth=2, markersize=8, capsize=5)
    #     ax.set_xlabel(param_display_name, fontsize=18)
    #     ax.set_ylabel('$C_3$ (triplet correlation)', fontsize=18)
    #     # ax.grid(True, alpha=0.3)

    ax.axis('off')

    # Set x-axis labels if not numeric
    if not isinstance(x_values[0], (int, float)):
        for ax in axes.flat:
            ax.set_xticks(x_values)
            ax.set_xticklabels(param_values, rotation=45, ha='right')

    # plt.suptitle(f'Information Structure vs {param_display_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'fig/ising_comparison_{param_display_name}.png', dpi=400, bbox_inches='tight')
    plt.show()

    return results


def compare_experiments(config_list, varied_parameter=None):
    """
    Run both GNN and Ising comparisons

    Args:
        config_list: list of config file names
        varied_parameter: 'section.parameter' format or None for config indices
    """

    # print("\n" + "="*80)
    # print("EXPERIMENT COMPARISON")
    # print("="*80)

    # Run GNN comparison
    gnn_results = compare_gnn_results(config_list, varied_parameter)

    # Run Ising comparison
    ising_results = compare_ising_results(config_list, varied_parameter)

    # Optional: Create combined summary plot if both analyses succeeded
    if gnn_results and ising_results:
        create_combined_summary_plot(gnn_results, ising_results, varied_parameter)

    return {'gnn': gnn_results, 'ising': ising_results}


def create_combined_summary_plot(gnn_results, ising_results, varied_parameter):
    """Create a combined plot showing both GNN performance and information metrics"""

    # This is where you could create your custom combined visualization
    # For example, showing GNN accuracy alongside I_HOC to demonstrate
    # the relationship between higher-order correlations and performance

    pass  # Implement as needed





def plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, delta_t, output_path):

   # Calculate mean and std for each neuron
   mu_activity = torch.mean(activity, dim=1)
   sigma_activity = torch.std(activity, dim=1)

   # Create the plot (keeping original visualization)
   plt.figure(figsize=(16, 8))
   plt.errorbar(np.arange(n_neurons), to_numpy(mu_activity), yerr=to_numpy(sigma_activity),
                fmt='o', ecolor='lightgray', alpha=0.6, elinewidth=1, capsize=0,
                markersize=3, color='red')

   # Group neurons by type and add labels at type boundaries (similar to plot_ground_truth_distributions)
   type_boundaries = {}
   current_type = None
   for i in range(n_neurons):
       neuron_type_id = to_numpy(type_list[i]).item()
       if neuron_type_id != current_type:
           if current_type is not None:
               type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
           type_boundaries[neuron_type_id] = (i, i)
           current_type = neuron_type_id

   # Close the last type boundary
   if current_type is not None:
       type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

   # Add vertical lines and x-tick labels for each neuron type
   tick_positions = []
   tick_labels = []

   for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
       center_pos = (start_idx + end_idx) / 2
       neuron_type_name = index_to_name.get(neuron_type_id, f'Type{neuron_type_id}')

       tick_positions.append(center_pos)
       tick_labels.append(neuron_type_name)

       # Add vertical line at type boundary
       if start_idx > 0:
           plt.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.3)

   # Set x-ticks with neuron type names rotated 90 degrees
   plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)
   plt.ylabel(r'neuron voltage $v_i(t)\quad\mu_i \pm \sigma_i$', fontsize=16)
   plt.yticks(fontsize=18)

   plt.tight_layout()
   plt.savefig(f'./{output_path}/activity_mu_sigma.png', dpi=300, bbox_inches='tight')
   plt.close()

   # Return per-neuron statistics (NEW)
   return {
       'mu_activity': to_numpy(mu_activity),
       'sigma_activity': to_numpy(sigma_activity)
   }


def plot_ground_truth_distributions(edges, true_weights, gt_taus, gt_V_Rest, type_list, n_neuron_types,
                                    sorted_neuron_type_names, output_path):
    """
    Create a 4-panel vertical figure showing ground truth parameter distributions per neuron type
    with neuron type names as x-axis labels
    """

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Get type boundaries for labels
    type_boundaries = {}
    current_type = None
    n_neurons = len(type_list)

    for i in range(n_neurons):
        neuron_type_id = int(type_list[i])
        if neuron_type_id != current_type:
            if current_type is not None:
                type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
            type_boundaries[neuron_type_id] = (i, i)
            current_type = neuron_type_id

    # Close the last type boundary
    if current_type is not None:
        type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

    def add_type_labels_and_setup_axes(ax, y_values, title):
        # Add mean line for each type and collect type positions
        type_positions = []
        type_names = []

        for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
            center_pos = (start_idx + end_idx) / 2
            type_positions.append(center_pos)
            neuron_type_name = sorted_neuron_type_names[int(neuron_type_id)] if int(neuron_type_id) < len(
                sorted_neuron_type_names) else f'Type{neuron_type_id}'
            type_names.append(neuron_type_name)

            # Add mean line for this type
            type_mean = np.mean(y_values[start_idx:end_idx + 1])
            ax.hlines(type_mean, start_idx, end_idx, colors='red', linewidth=3)

        # Set x-ticks to neuron type names
        ax.set_xticks(type_positions)
        ax.set_xticklabels(type_names, rotation=90, fontsize=8)
        ax.tick_params(axis='y', labelsize=16)

    # Panel 1: Scatter plot of true weights per connection with neuron index
    ax1 = axes[0]
    connection_targets = edges[1, :]
    connection_weights = true_weights

    ax1.scatter(connection_targets, connection_weights, c='white', s=0.1)
    ax1.set_ylabel('true weights', fontsize=16)

    # For weights, compute means per target neuron
    weight_means_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == i)[0]
        if len(incoming_edges) > 0:
            weight_means_per_neuron[i] = np.mean(true_weights[incoming_edges])

    add_type_labels_and_setup_axes(ax1, weight_means_per_neuron, 'distribution of true weights by neuron type')

    # Panel 2: Number of connections per neuron
    ax2 = axes[1]
    n_connections_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        n_connections_per_neuron[i] = np.sum(edges[1, :] == i)

    ax2.scatter(np.arange(n_neurons), n_connections_per_neuron, c='white', s=0.1)
    ax2.set_ylabel('number of connections', fontsize=16)
    add_type_labels_and_setup_axes(ax2, n_connections_per_neuron, 'number of incoming connections by neuron type')

    # Panel 3: Scatter plot of true tau values per neuron
    ax3 = axes[2]
    ax3.scatter(np.arange(n_neurons), gt_taus * 1000, c='white', s=0.1)
    ax3.set_ylabel(r'true $\tau$ values [ms]', fontsize=16)
    add_type_labels_and_setup_axes(ax3, gt_taus * 1000, r'distribution of true $\tau$ by neuron type')

    # Panel 4: Scatter plot of true V_rest values per neuron
    ax4 = axes[3]
    ax4.scatter(np.arange(n_neurons), gt_V_Rest, c='white', s=0.1)
    ax4.set_ylabel(r'true $v_{rest}$ values [a.u.]', fontsize=16)
    add_type_labels_and_setup_axes(ax4, gt_V_Rest, r'distribution of true $v_{rest}$ by neuron type')

    plt.tight_layout()
    plt.savefig(f'{output_path}/ground_truth_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    return fig


def plot_loss_curves(log_dir, ylim=None):
    """
    Iterates through all folders in the specified directory, loads 'loss.pt' files,
    and plots all loss curves on a single plot.

    Parameters:
    - log_dir (str): Path to the directory containing subfolders with 'loss.pt' files.
    - output_file (str): Path to save the resulting plot.
    - ylim (tuple, optional): Y-axis limits for the plot (e.g., (0, 0.0075)).
    """
    loss_data = {}

    # Iterate through all folders in the specified directory
    for folder in os.listdir(log_dir):
        folder_path = os.path.join(log_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a directory
            loss_file = os.path.join(folder_path, 'loss.pt')
            if os.path.exists(loss_file):  # Check if 'loss.pt' exists
                try:
                    # Load the loss values
                    loss_values = torch.load(loss_file)
                    loss_data[folder] = loss_values
                except Exception as e:
                    print(f"Error loading {loss_file}: {e}")

    # Plot all loss lists on a single plot
    plt.figure(figsize=(10, 6))
    for folder, loss_values in loss_data.items():
        plt.plot(loss_values, label=folder)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Curves', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(log_dir+'/loss_curves.png', dpi=150)
    plt.close()




def data_plot(config, config_file, epoch_list, style, extended, device, apply_weight_correction=False, log_file=None):

    # plt.rcParams['text.usetex'] = False  # LaTeX disabled
    # rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    if 'black' in style:
        plt.style.use('dark_background')
        mc ='w'
    else:
        plt.style.use('default')
        mc = 'k'

    if 'latex' in style:
        plt.rcParams['text.usetex'] = False  # LaTeX disabled
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})

    matplotlib.rcParams['savefig.pad_inches'] = 0

    log_dir, logger = create_log_dir(config=config, erase=False)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)

    if epoch_list==['best']:
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch_list=[filename]
        print(f'best model: {epoch_list}')
        logger.info(f'best model: {epoch_list}')

    if os.path.exists(f'{log_dir}/loss.pt'):
        loss = torch.load(f'{log_dir}/loss.pt')
        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(loss, color=mc, linewidth=4)
        plt.xlim([0, 20])
        plt.ylabel('loss', fontsize=68)
        plt.xlabel('epochs', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/loss.png", dpi=170.7)
        plt.close()
        # Log final loss to analysis.log
        if log_file and len(loss) > 0:
            log_file.write(f"final_loss: {loss[-1]:.4e}\n")



    if ('PDE_N3' in config.graph_model.signal_model_name):
        plot_synaptic3(config, epoch_list, log_dir, logger, 'viridis', style, extended, device)
    else:
        plot_signal(config, epoch_list, log_dir, logger, 'viridis', style, extended, device, apply_weight_correction, log_file)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def plot_results_figure(config_file_, config_indices, panel_suffix='domain'):
    """
    Generate a 2x3 figure panel for a given configuration.

    Args:
        config_file_: Config file name (e.g., 'fly_N9_44_24')
        config_indices: Index string for specific plots (e.g., '44_6')
        panel_suffix: Suffix for edge_functions panel ('domain' or 'all')
    """
    # Setup config
    config_file, pre_folder = add_pre_folder(config_file_)
    config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_file_
    print(f'figure results {config_file_}...')

    # Generate plots
    # data_plot(config=config, config_file=config_file, epoch_list=['best'],
    #           style='white color', extended='plots', device=device)

    # Setup paths
    log_dir = f'log/fly/{config_file_}'
    panels = {
        'a': f"{log_dir}/results/weights_comparison_corrected.png",
        'b': f"{log_dir}/results/embedding_{config_indices}.png",
        'c': f"{log_dir}/results/MLP1_{config_indices}.png",
        'd': f"{log_dir}/results/MLP0_{config_indices}_domain.png",
        'e': f"{log_dir}/results/tau_comparison_{config_indices}.png",
        'f': f"{log_dir}/results/V_rest_comparison_{config_indices}.png"
    }

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    for idx, (label, path) in enumerate(panels.items()):
        ax = fig.add_subplot(2, 3, idx+1)
        img = imageio.imread(path)
        plt.imshow(img)
        plt.axis('off')
        ax.text(0.1, 1.01, f'{label})', transform=ax.transAxes,
                fontsize=24, va='bottom', ha='right')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                        wspace=0.02, hspace=0.04)
    plt.savefig(f"./fig_paper/results_{config_file_.split('_')[-2]}_{config_file_.split('_')[-1]}.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(f"./fig_paper/results_{config_file_.split('_')[-2]}_{config_file_.split('_')[-1]}.pdf",
                dpi=300, bbox_inches='tight')
    plt.close()


def get_figures(index):

    plt.style.use('default')

    match index:

        case 'results_44_6':
             plot_results_figure('fly_N9_44_24', '44_6', 'domain')
        case 'results_51_2':
             plot_results_figure('fly_N9_51_2', '37_2', 'domain')
        case 'results_22_10':
             plot_results_figure('fly_N9_22_10', '18_4_0', 'domain')

        case 'extra_edges':
            config_list = ['fly_N9_51_9', 'fly_N9_51_10', 'fly_N9_51_11', 'fly_N9_51_12']

            for config_file_ in config_list:
                config_file, pre_folder = add_pre_folder(config_file_)
                config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                config.dataset = pre_folder + config.dataset
                config.config_file = pre_folder + config_file_
                logdir = f'log/fly/{config_file_}'
                data_test(
                    config,
                    visualize=True,
                    style="white color name",
                    verbose=False,
                    best_model='best',
                    run=0,
                    test_mode="full",
                    sample_embedding=False,
                    step=25000,
                    device=device,
                    particle_of_interest=0,
                )

        case 'figure_1_cosyne_2026':

            # config_file_ = 'fly_N9_22_10'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_22_10'
            # config.simulation.visual_input_type = "DAVIS"

            # data_test(
            #     config,
            #     visualize=False,
            #     style="white color name",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full",
            #     sample_embedding=False,
            #     step=25000,
            #     device=device,
            #     particle_of_interest=0,
            # )

            # config_file_ = 'fly_N9_22_10'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_22_10'
            # config.simulation.visual_input_type = "DAVIS"

            # data_test(
            #     config,
            #     visualize=False,
            #     style="white color name",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full test_ablation_50",
            #     sample_embedding=False,
            #     step=25000,
            #     device=device,
            #     particle_of_interest=0,
            # )

            # config_file_ = 'fly_N9_44_6'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_44_6'
            # config.simulation.visual_input_type = "DAVIS"

            # data_test(
            #     config,
            #     visualize=False,
            #     style="white color name",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full",
            #     sample_embedding=False,
            #     step=50,
            #     device=device,
            #     particle_of_interest=0,
            # )

            # config_file_ = 'fly_N9_51_2'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_51_2'

            # data_test(
            #     config,
            #     visualize=False,
            #     style="black color name true_only",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full",
            #     sample_embedding=False,
            #     step=50,
            #     device=device,
            #     particle_of_interest=0,
            # )


            index_to_name = {0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2',
                7: 'L3', 8: 'L4', 9: 'L5', 10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10',
                14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14', 18: 'Mi15', 19: 'Mi2',
                20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
                27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a',
                34: 'T3', 35: 'T4a', 36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b',
                41: 'T5c', 42: 'T5d', 43: 'Tm1', 44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28',
                48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y', 52: 'Tm5a', 53: 'Tm5b',
                54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14', 59: 'TmY15',
                60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'}

            print('plot figure 1...')
            x = np.load('graphs_data/fly/fly_N9_18_4_0/x_list_0.npy')
            type_list = x[-1,:, 6].astype(int)
            len(type_list)

            selected_types = [5, 12, 19, 23, 31, 35, 39, 43, 50, 55]
            neuron_indices = []
            for stype in selected_types:
                indices = np.where(type_list == stype)[0]
                if len(indices) == 0:
                    print(f"Type {stype} ({index_to_name[stype]}) not found in type_list")
                else:
                    neuron_indices.append(indices[0])
                    print(f"Type {stype} ({index_to_name[stype]}): neuron {indices[0]}")

            print(f"Found {len(neuron_indices)} neurons out of {len(selected_types)}")
            print(f"Unique types in type_list: {np.unique(type_list)}")

            logdirs = {'a': 'log/fly/fly_N9_22_10',
                    'b': 'log/fly/fly_N9_22_10',
                    'c': 'log/fly/fly_N9_44_6'}

            start_frame = 88000
            end_frame = 88500

            plt.style.use('default')
            fig, axes = plt.subplots(1, 3, figsize=(20, 12))

            for idx, (key, log_dir) in enumerate(logdirs.items()):
                print(f'processing {key}) {log_dir}...')
                ax = axes[idx]

                if idx==1:
                    true = np.load(f"./{log_dir}/results/activity_modified.npy")
                    pred = np.load(f"./{log_dir}/results/activity_modified_pred.npy")
                else:
                    true = np.load(f"./{log_dir}/results/activity_true.npy")
                    pred = np.load(f"./{log_dir}/results/activity_pred.npy")

                true_slice = true[neuron_indices, start_frame:end_frame].copy()
                pred_slice = pred[neuron_indices, start_frame:end_frame].copy()
                del true, pred

                step_v = 1.5

                for i in range(10):
                    baseline = np.mean(true_slice[i]) if idx==1 else np.mean(pred_slice[i])
                    lw = 6 if idx==2 else 10
                    ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw, c='green', alpha=0.5,
                            label='true' if i == 0 and idx==2 else None)

                for i in range(10):
                    baseline = np.mean(true_slice[i]) if idx==1 else np.mean(pred_slice[i])
                    ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=2, c='black',
                            label='pred' if i == 0 and idx==2 else None)

                if idx == 0:
                    for i in range(10):
                        ax.text(-100, i * step_v, index_to_name[selected_types[i]],
                                fontsize=24, va='center')

                ax.set_ylim([-step_v, 10 * step_v])
                ax.set_yticks([])

                # Panel labels
                ax.text(-0.12 if idx==0 else -0.05, 1.02, f'{chr(97+idx)})', transform=ax.transAxes,
                        fontsize=24, va='top')

                # Spine removal
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if idx > 0:
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                # X-axis only for panel a
                if idx == 0:
                    ax.set_xticks([0, end_frame - start_frame])
                    ax.set_xticklabels([start_frame, end_frame], fontsize=20)
                    ax.set_xlabel('frame', fontsize=24)
                else:
                    ax.set_xticks([])

            plt.tight_layout()
            plt.subplots_adjust(left=0.05)
            plt.savefig('./fig_paper/Fig1.pdf', dpi=400, bbox_inches='tight')
            plt.savefig('./fig_paper/Fig1.png', dpi=400, bbox_inches='tight')
            plt.close()

        case 'N9_44_6':
            config_file_ = 'fly_N9_44_6'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            logdir = 'log/fly/fly_N9_44_6'

            # config.training.noise_model_level = 0.0
            config.simulation.visual_input_type = "DAVIS"

            data_test(
                config,
                visualize=True,
                style="white color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="full",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_wo_noise.png')


            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS_true_wo_noise.png')

            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS_pred_wo_noise.png')

            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS_pred_abalation_50_wo_noise.png')


            os.remove(f'./{logdir}/results/activity_8x8_panel_comparison.png')

        case 'N9_51_2':
            config_file_ = 'fly_N9_51_2'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            logdir = 'log/fly/fly_N9_51_2'

            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_true_signal.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_abalation_50.png')
            copyfile('overlay_all_W.png', f'./{logdir}/results/overlay_all_W.png')
            os.remove('overlay_all_W.png')

            config.simulation.visual_input_type = ""
            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_optical_flow.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_optical_flow.png')
            os.remove(f'./{logdir}/results/activity_8x8_panel_comparison.png')

        case 'N9_22_10':
            config_file_ = 'fly_N9_22_10'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            logdir = 'log/fly/fly_N9_22_10'
            config.simulation.visual_input_type = "DAVIS"
            data_test(
                config,
                visualize=True,
                style="white color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="full",
                sample_embedding=False,
                step=25000,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_DAVIS.png')
            copyfile(f'./{logdir}/results/activity_5x4_panel_comparison.png',
                     f'./{logdir}/results/activity_5x4_panel_comparison_DAVIS.png')
            data_test(
                config,
                visualize=True,
                style="white color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="full test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_DAVIS_ablation_50.png')
            copyfile(f'./{logdir}/results/activity_5x4_panel_comparison.png',
                     f'./{logdir}/results/activity_5x4_panel_comparison_DAVIS_ablation_50.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_abalation_50.png')

            config.simulation.visual_input_type = "DAVIS"
            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_DAVIS.png')

            os.remove(f'./{logdir}/results/activity_8x8_panel_comparison.png')



        case 'new_network_1':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
                new_params = [0.25, 10, 20, 30, 60],
            )

        case 'new_network_2':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
                new_params = [0.5, 60, 40, 0, 0],
            )

        case 'ablation_weights':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_90",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )

        case 'ablation_cells':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_inactivity_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_inactivity_25",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_inactivity_50",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )

        case 'permutation_types':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_permutation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_permutation_50",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_permutation_90",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )

        case 'weight_vs_noise':

            config_list = ['fly_N9_44_15', 'fly_N9_44_16', 'fly_N9_44_17', 'fly_N9_44_18', 'fly_N9_44_19', 'fly_N9_44_20', 'fly_N9_44_21', 'fly_N9_44_22', 'fly_N9_44_23', 'fly_N9_44_24', 'fly_N9_44_25', 'fly_N9_44_26']
            compare_experiments(config_list,'training.noise_model_level')

            # copy file ising_comparison_noise level.png
            shutil.copy('fig/ising_comparison_noise level.png', 'fig_paper/ising_comparison_noise_level.png')
            shutil.copy('fig/gnn_comparison_noise_model_level.png', 'fig_paper/gnn_comparison_noise_model_level.png')

        case 'correction_weight':

            config_file_ = 'fly_N9_22_10'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            print('figure correction_weight...')

            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='white color', extended='plots', device=device)

            log_dir = 'log/fly/fly_N9_22_10'
            config_indices = '18_4_0'

            fig = plt.figure(figsize=(12, 10))

            ax1 = fig.add_subplot(3, 3, 1)
            panel_pic_path =f"./{log_dir}/results/MLP1_{config_indices}.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax1.text(0.1, 1.01, 'a)', transform=ax1.transAxes, fontsize=18, va='bottom', ha='right')

            ax2 = fig.add_subplot(3, 3, 2)
            panel_pic_path =f"./{log_dir}/results/MLP1_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax2.text(0.1, 1.01, 'b)', transform=ax2.transAxes, fontsize=18, va='bottom', ha='right')

            ax3 = fig.add_subplot(3, 3, 3)
            panel_pic_path = f"./{log_dir}/results/MLP1_slope_{config_indices}.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax3.text(0.1, 1.01, 'c)', transform=ax3.transAxes, fontsize=18, va='bottom', ha='right')

            # Second row
            ax4 = fig.add_subplot(3, 3, 4)
            panel_pic_path = f"./{log_dir}/results/MLP0_{config_indices}.png"

            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax4.text(0.1, 1.01, 'd)', transform=ax4.transAxes, fontsize=18, va='bottom', ha='right')

            ax5 = fig.add_subplot(3, 3, 5)
            panel_pic_path = f"./{log_dir}/results/MLP0_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax5.text(0.1, 1.01, 'e)', transform=ax5.transAxes, fontsize=18, va='bottom', ha='right')

            ax6 = fig.add_subplot(3, 3, 6)
            panel_pic_path = f"./{log_dir}/results/MLP0_{config_indices}_params.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax6.text(0.1, 1.01, 'f)', transform=ax6.transAxes, fontsize=18, va='bottom', ha='right')

            # Third row
            ax7 = fig.add_subplot(3, 3, 7)
            panel_pic_path = f"./{log_dir}/results/weights_comparison_raw.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Neuron\nEmbedding', ha='center', va='center', fontsize=16, transform=ax7.transAxes)
            plt.axis('off')
            ax7.text(0.1, 1.01, 'g)', transform=ax7.transAxes, fontsize=18, va='bottom', ha='right')

            ax8 = fig.add_subplot(3, 3, 8)
            panel_pic_path = f"./{log_dir}/results/weights_comparison_rj.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Type\nReconstruction', ha='center', va='center', fontsize=16, transform=ax8.transAxes)
            plt.axis('off')
            ax8.text(0.1, 1.01, 'h)', transform=ax8.transAxes, fontsize=18, va='bottom', ha='right')

            ax9 = fig.add_subplot(3, 3, 9)
            panel_pic_path = f"./{log_dir}/results/weights_comparison_corrected.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Reconstruction\nCorrelations', ha='center', va='center', fontsize=16, transform=ax9.transAxes)
            plt.axis('off')
            ax9.text(0.1, 1.01, 'i)', transform=ax9.transAxes, fontsize=18, va='bottom', ha='right')

            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.02, hspace=0.04)
            plt.savefig("./fig_paper/figure_correction_weight.png", dpi=300, bbox_inches='tight')
            plt.close()

        case 'correction_weight_noise':

            config_file_ = 'fly_N9_44_6'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            print('figure correction_weight...')

            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device)

            log_dir = 'log/fly/fly_N9_44_6'
            config_indices = '44_6'

            fig = plt.figure(figsize=(10, 9))

            ax1 = fig.add_subplot(3, 3, 1)
            panel_pic_path =f"./{log_dir}/results/MLP1_{config_indices}.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax1.text(0.1, 1.01, 'a)', transform=ax1.transAxes, fontsize=18, va='bottom', ha='right')

            ax2 = fig.add_subplot(3, 3, 2)
            panel_pic_path =f"./{log_dir}/results/MLP1_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax2.text(0.1, 1.01, 'b)', transform=ax2.transAxes, fontsize=18, va='bottom', ha='right')

            ax3 = fig.add_subplot(3, 3, 3)
            panel_pic_path = f"./{log_dir}/results/MLP1_slope_{config_indices}.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax3.text(0.1, 1.01, 'c)', transform=ax3.transAxes, fontsize=18, va='bottom', ha='right')

            # Second row
            ax4 = fig.add_subplot(3, 3, 4)
            panel_pic_path = f"./{log_dir}/results/MLP0_{config_indices}.png"

            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax4.text(0.1, 1.01, 'd)', transform=ax4.transAxes, fontsize=18, va='bottom', ha='right')

            ax5 = fig.add_subplot(3, 3, 5)
            panel_pic_path = f"./{log_dir}/results/MLP0_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax5.text(0.1, 1.01, 'e)', transform=ax5.transAxes, fontsize=18, va='bottom', ha='right')

            ax6 = fig.add_subplot(3, 3, 6)
            panel_pic_path = f"./{log_dir}/results/MLP0_{config_indices}_params.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax6.text(0.1, 1.01, 'f)', transform=ax6.transAxes, fontsize=18, va='bottom', ha='right')

            # Third row
            ax7 = fig.add_subplot(3, 3, 7)
            panel_pic_path = f"./{log_dir}/results/weights_comparison_raw.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Neuron\nEmbedding', ha='center', va='center', fontsize=16, transform=ax7.transAxes)
            plt.axis('off')
            ax7.text(0.1, 1.01, 'g)', transform=ax7.transAxes, fontsize=18, va='bottom', ha='right')

            ax8 = fig.add_subplot(3, 3, 8)
            panel_pic_path = f"./{log_dir}/results/weights_comparison_rj.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Type\nReconstruction', ha='center', va='center', fontsize=16, transform=ax8.transAxes)
            plt.axis('off')
            ax8.text(0.1, 1.01, 'h)', transform=ax8.transAxes, fontsize=18, va='bottom', ha='right')

            ax9 = fig.add_subplot(3, 3, 9)
            panel_pic_path = f"./{log_dir}/results/weights_comparison_corrected.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Reconstruction\nCorrelations', ha='center', va='center', fontsize=16, transform=ax9.transAxes)
            plt.axis('off')
            ax9.text(0.1, 1.01, 'i)', transform=ax9.transAxes, fontsize=18, va='bottom', ha='right')

            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.02, hspace=0.04)
            plt.savefig("./fig_paper/figure_correction_weight_noise.png", dpi=300, bbox_inches='tight')
            plt.close()










if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    print(' ')
    print(f'device {device}')

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass




    config_list = ['signal_Claude']


    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        print(f'\033[94mconfig_file  {config.config_file}\033[0m')
        folder_name = './log/' + pre_folder + '/tmp_results/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)
        # data_plot(config=config, config_file=config_file, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=False)
        # data_plot(config=config, config_file=config_file, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=True)


    # compare_experiments(config_list, 'training.batch_size')

    # get_figures('weight_vs_noise')
    # get_figures('correction_weight')
    # get_figures('correction_weight_noise')

    # get_figures('ablation_weights')]
    # get_figures('permutation_types')
    # get_figures('new_network_1')
    # get_figures('new_network_2')

    # get_figures('extra_edges')
    # get_figures('N9_22_10')
    # get_figures('results_22_10')
    # get_figures('results_44_24')
    # get_figures('N9_44_6')
    # get_figures('N9_51_2')

    # get_figures('results_51_2')
    # get_figures('figure_1_cosyne_2026')



    print("analysis completed")



