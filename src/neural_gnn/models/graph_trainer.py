import os
import time
import glob
import warnings
import logging

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import tifffile
import numpy as np

from neural_gnn.models.utils import (
    choose_training_model,
    choose_inr_model,
    increasing_batch_size,
    constant_batch_size,
    set_trainable_parameters,
    get_in_features_update,
    analyze_edge_function,
    plot_training_signal,
    plot_training_signal_visual_input,
    plot_training_signal_missing_activity,
    plot_weight_comparison,
    get_index_particles,
    analyze_data_svd,
)
from neural_gnn.utils import (
    to_numpy,
    CustomColorMap,
    create_log_dir,
    check_and_clear_memory,
    sort_key,
    fig_init,
    get_equidistant_points,
    open_gcs_zarr,
    compute_trace_metrics,
    get_datavis_root_dir,
    LossRegularizer,
)
from neural_gnn.models.Siren_Network import Siren, Siren_Network
from neural_gnn.models.LowRank_INR import LowRankINR
from neural_gnn.models.Neural_ode_wrapper_Signal import integrate_neural_ode_Signal, neural_ode_loss_Signal

from neural_gnn.models.HashEncoding_Network import HashEncodingMLP

from neural_gnn.sparsify import EmbeddingCluster, sparsify_cluster, clustering_evaluation
from neural_gnn.fitting_models import linear_model

from scipy.optimize import curve_fit

from torch_geometric.data import Data as pyg_Data
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
import seaborn as sns
# denoise_data import not needed - removed star import
from tifffile import imread
from matplotlib.colors import LinearSegmentedColormap
from neural_gnn.generators.utils import choose_model, plot_signal_loss, generate_compressed_video_mp4, init_connectivity
from neural_gnn.generators.graph_data_generator import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)
import pandas as pd
import napari
from collections import deque
from tqdm import tqdm, trange
from prettytable import PrettyTable
import imageio




def data_train(config=None, erase=False, best_model=None, style=None, device=None, log_file=None):
    # plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
    # rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    dataset_name = config.dataset
    print(f"\033[94mdataset_name: {dataset_name}\033[0m")
    print(f"\033[92m{config.description}\033[0m")

    data_train_signal(config, erase, best_model, style, device, log_file)

    print("training completed.")



def data_train_signal(config, erase, best_model, style, device, log_file=None):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    n_epochs = train_config.n_epochs
    n_runs = train_config.n_runs
    n_neuron_types = simulation_config.n_neuron_types

    dataset_name = config.dataset
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    recurrent_training = train_config.recurrent_training
    noise_recurrent_level = train_config.noise_recurrent_level
    recurrent_parameters = train_config.recurrent_parameters.copy()
    neural_ODE_training = train_config.neural_ODE_training
    ode_method = train_config.ode_method
    ode_rtol = train_config.ode_rtol
    ode_atol = train_config.ode_atol
    ode_adjoint = train_config.ode_adjoint
    target_batch_size = train_config.batch_size
    delta_t = simulation_config.delta_t
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    batch_ratio = train_config.batch_ratio
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq

    # external input configuration (hierarchy: visual > signal > none)
    external_input_type = simulation_config.external_input_type
    inr_type = model_config.inr_type

    embedding_cluster = EmbeddingCluster(config)

    time_step = train_config.time_step
    has_missing_activity = train_config.has_missing_activity
    multi_connectivity = config.training.multi_connectivity
    baseline_value = simulation_config.baseline_value
    cmap = CustomColorMap(config=config)

    if "black" in style:
        plt.style.use("dark_background")
        mc = 'white'
    else:
        plt.style.use("default")
        mc = 'black'

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)

    log_dir, logger = create_log_dir(config, erase)
    print('loading data...')

    x_list = []
    y_list = []
    for run in trange(0,n_runs, ncols=80):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)

    run = 0
    x = x_list[0][n_frames - 10]
    n_neurons = x.shape[0]
    config.simulation.n_neurons =n_neurons
    type_list = torch.tensor(x[:, 6:7], device=device)  # neuron_type is at column 6

    activity = torch.tensor(x_list[0][:, :, 3], device=device)  # signal state is at column 3
    distrib = activity.flatten()
    distrib = distrib[~torch.isnan(distrib)]
    if len(distrib) > 0:
        xnorm = torch.round(1.5 * torch.std(distrib))
    else:
        print('no valid distribution found, setting xnorm to 1.0')
        xnorm = torch.tensor(1.0, device=device)
    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)

    # SVD analysis of activity and external_input (skip if already exists)
    # svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    # if not os.path.exists(svd_plot_path):
    #     analyze_data_svd(x_list[0], log_dir, config=config, logger=logger)
    # else:
    #     print(f'svd analysis already exists: {svd_plot_path}')

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(model_config=config, device=device)
    model.to(device)  # Ensure all model parameters are on the correct device
    model.train()

    if has_missing_activity:
        assert batch_ratio == 1, f"batch_ratio must be 1, got {batch_ratio}"
        model_missing_activity = nn.ModuleList([
            Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                  hidden_features=model_config.hidden_dim_nnr,
                  hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                  hidden_omega_0=model_config.omega,
                  outermost_linear=model_config.outermost_linear_nnr)
            for n in range(n_runs)
        ])
        model_missing_activity.to(device=device)
        optimizer_missing_activity = torch.optim.Adam(lr=train_config.learning_rate_missing_activity,
                                                      params=model_missing_activity.parameters())
        model_missing_activity.train()


    # external input model (model_f) for learning external_input reconstruction
    optimizer_f = None
    model_f = None
    if external_input_type != 'none':
        model_f = choose_inr_model(config=config, n_neurons=n_neurons, n_frames=n_frames, x_list=x_list, device=device)
        # Separate omega parameters from other parameters for different learning rates
        omega_params = [(name, p) for name, p in model_f.named_parameters() if 'omega' in name]
        other_params = [p for name, p in model_f.named_parameters() if 'omega' not in name]
        if omega_params:
            print(f"model_f omega parameters found: {[name for name, p in omega_params]}")
            optimizer_f = torch.optim.Adam([
                {'params': other_params, 'lr': train_config.learning_rate_NNR_f},
                {'params': [p for name, p in omega_params], 'lr': train_config.learning_rate_omega_f}
            ])
        else:
            print("model_f: no omega parameters found (omega_f_learning=False or non-SIREN model)")
            optimizer_f = torch.optim.Adam(lr=train_config.learning_rate_NNR_f, params=model_f.parameters())
        model_f.train()
        # Print initial omega values if learnable
        if hasattr(model_f, 'get_omegas'):
            omegas = model_f.get_omegas()
            if omegas:
                print(f"model_f initial omegas: {omegas}")
                print(f"model_f omega LR: {train_config.learning_rate_omega_f}, L2 coeff: {train_config.coeff_omega_f_L2}")

    if (best_model != None) & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'load {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
        if model_f is not None:
            net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{best_model}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])
        list_loss = torch.load(os.path.join(log_dir, 'loss.pt'))
    else:
        start_epoch = 0
        list_loss = []

    loss_components = {'loss': []}  # regularizer handles other components

    print('set optimizer ...')
    lr = train_config.learning_rate_start
    lr_update = lr
    if train_config.init_training_single_type:
        lr_embedding = 1.0E-16
    else:
        lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start
    lr_modulation = train_config.learning_rate_modulation_start

    learning_rate_NNR = train_config.learning_rate_NNR
    learning_rate_NNR_f = train_config.learning_rate_NNR_f

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
    model.train()

    print(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}')
    logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}')

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    logger.info(f'network: {net}  N epochs: {n_epochs}  initial batch_size: {batch_size}')

    print('training setup ...')
    connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)

    if train_config.with_connectivity_mask:
        model.mask = (connectivity > 0) * 1.0
        adj_t = model.mask.float() * 1
        adj_t = adj_t.t()
        edges = adj_t.nonzero().t().contiguous()
        edges_all = edges.clone().detach()

        with torch.no_grad():
            if multi_connectivity:
                for run_ in range(n_runs):
                    model.W[run_].copy_(model.W[run_] * model.mask)
            else:
                model.W.copy_(model.W * model.mask)
    else:

        edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
        edges_all = edges.clone().detach()

    if train_config.coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            index_weight.append(torch.argwhere(model.mask[:, i] > 0).squeeze())

    print(f'n neurons: {n_neurons}, edges:{edges.shape[1]}, xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'n neurons: {n_neurons}, edges:{edges.shape[1]}, xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')


     # PDE_N3 is special, embedding changes over time
    if 'PDE_N3' in model_config.signal_model_name:
        ind_a = torch.tensor(np.arange(1, n_neurons * 100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]

    # initialize regularizer
    regularizer = LossRegularizer(
        train_config=train_config,
        model_config=model_config,
        activity_column=3,  # signal uses column 6
        plot_frequency=1,   # will be updated per epoch
        n_neurons=n_neurons,
        trainer_type='signal'
    )

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []
    time.sleep(1.0)

    training_start_time = time.time()

    for epoch in range(start_epoch, n_epochs):

        if (epoch == train_config.epoch_reset):
            with torch.no_grad():
                model.W.copy_(model.W * 0)
                model.a.copy_(model.a * 0)
            logger.info(f'reset W model.a at epoch : {epoch}')
            print(f'reset W model.a at epoch : {epoch}')
        if (epoch == 1) & (train_config.init_training_single_type):
            lr_embedding = train_config.learning_rate_embedding_start
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
            model.train()


        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio * 0.2)
        else:
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2 )

        plot_frequency = int(Niter // 20)
        if epoch ==0:
            print(f'{Niter} iterations per epoch, {plot_frequency} iterations per plot')
            logger.info(f'{Niter} iterations per epoch')

        regularizer.set_epoch(epoch, plot_frequency)

        total_loss = 0
        total_loss_regul = 0
        run = 0
        last_connectivity_r2 = None  # track last R² for progress display

        # Progress reporting: print 20 times during training (for subprocess mode)
        report_interval = max(1, Niter // 20)
        tqdm_disabled = os.environ.get('TQDM_DISABLE', '0') == '1'
        epoch_start_time = time.time()

        time.sleep(1.0)
        pbar = trange(Niter, ncols=150, disable=tqdm_disabled)
        for N in pbar:

            if has_missing_activity:
                optimizer_missing_activity.zero_grad()
            if model_f is not None:
                optimizer_f.zero_grad()
            optimizer.zero_grad()

            dataset_batch = []
            ids_batch = []
            ids_index = 0

            loss = torch.zeros(1, device=device)

            regularizer.reset_iteration()

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 4 - time_step)

                if recurrent_training or neural_ODE_training:
                    k = k - k % time_step

                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)

                ids = np.arange(n_neurons)

                if not (torch.isnan(x).any()):

                    # special case regularizations (kept outside LossRegularizer)
                    if has_missing_activity:
                        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                        missing_activity = baseline_value + model_missing_activity[run](t).squeeze()
                        if (train_config.coeff_missing_activity>0):
                            loss_missing_activity = (missing_activity[ids] - x[ids, 3].clone().detach()).norm(2)
                            regul_term = loss_missing_activity * train_config.coeff_missing_activity
                            loss = loss + regul_term
                        ids_missing = torch.argwhere(x[:, 3] == baseline_value)
                        x[ids_missing,3] = missing_activity[ids_missing]
                    # external input reconstruction (when learn_external_input=True)
                    if (model_f is not None):
                        nnr_f_T_period = model_config.nnr_f_T_period
                        if (external_input_type == 'visual') :
                            n_input_neurons = simulation_config.n_input_neurons
                            learned_input = model_f(time=k / n_frames) ** 2
                            print(n_input_neurons, learned_input.shape)
                            x[:n_input_neurons, 4:5] = learned_input
                            x[n_input_neurons:n_neurons, 4:5] = 1
                        elif external_input_type == 'signal':
                            t_norm = torch.tensor([[k / nnr_f_T_period]], dtype=torch.float32, device=device)
                            if inr_type == 'siren_t':
                                x[:, 4] = model_f(t_norm).squeeze()
                            elif inr_type == 'lowrank':
                                t_idx = torch.tensor([k], dtype=torch.long, device=device)
                                x[:, 4] = model_f(t_idx).squeeze()
                            elif inr_type == 'ngp':
                                x[:, 4] = model_f(t_norm).squeeze()
                            # siren_id and siren_x would need position/id info - not implemented in training loop yet

                    regul_loss = regularizer.compute(
                        model=model,
                        x=x,
                        in_features=None,
                        ids=ids,
                        ids_batch=None,
                        edges=edges,
                        device=device,
                        xnorm=xnorm,
                        index_weight=index_weight if train_config.coeff_W_sign > 0 else None
                    )

                    loss = loss + regul_loss

                    if batch_ratio < 1:
                        ids_ = np.random.permutation(ids.shape[0])[:int(ids.shape[0] * batch_ratio)]
                        ids = np.sort(ids_)
                        edges = edges_all.clone().detach()
                        mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                        edges = edges[:, mask]

                    if recurrent_training or neural_ODE_training:
                        y = torch.tensor(x_list[run][k + time_step, :, 3:4], dtype=torch.float32, device=device).detach()
                    else:
                        y = torch.tensor(y_list[run][k], device=device) / ynorm


                    if not (torch.isnan(y).any()):

                        dataset = pyg_Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                            x_batch = x[:, 3:4]
                            y_batch = y
                            k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                            ids_batch = ids
                        else:
                            data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 3:4]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            k_batch = torch.cat(
                                (k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                        ids_index += x.shape[0]

            if not (dataset_batch == []):

                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                for batch in batch_loader:
                    if regularizer.needs_update_regul():
                        pred, in_features = model(batch, data_id=data_id, k=k_batch, return_all=True)
                        update_regul_loss = regularizer.compute_update_regul(
                            model=model,
                            in_features=in_features,
                            ids_batch=ids_batch,
                            device=device,
                            x=x,
                            xnorm=xnorm,
                            ids=ids
                        )
                        loss = loss + update_regul_loss
                    else:
                        pred = model(batch, data_id=data_id, k=k_batch)

                if neural_ODE_training:
                    ode_loss, pred_x = neural_ode_loss_Signal(
                        model=model,
                        dataset_batch=dataset_batch,
                        x_list=x_list,
                        run=run,
                        k_batch=k_batch,
                        time_step=time_step,
                        batch_size=batch_size,
                        n_neurons=n_neurons,
                        ids_batch=ids_batch,
                        delta_t=delta_t,
                        device=device,
                        data_id=data_id,
                        y_batch=y_batch,
                        noise_level=noise_recurrent_level,
                        ode_method=ode_method,
                        rtol=ode_rtol,
                        atol=ode_atol,
                        adjoint=ode_adjoint
                    )
                    loss = loss + ode_loss

                elif recurrent_training:

                    pred_x = x_batch + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

                    if time_step > 1:
                        for step in range(1, time_step):
                            dataset_batch_new = []
                            for b in range(batch_size):
                                start_idx = b * n_neurons
                                end_idx = (b + 1) * n_neurons
                                dataset_batch[b].x[:, 3:4] = pred_x[start_idx:end_idx].reshape(-1, 1)

                                # update external_input for next time step during rollout
                                k_current = k_batch[start_idx, 0].item() + step
                                if model_f is not None:
                                    nnr_f_T_period = model_config.nnr_f_T_period
                                    if external_input_type == 'visual':
                                        n_input_neurons = simulation_config.n_input_neurons
                                        dataset_batch[b].x[:n_input_neurons, 4:5] = model_f(time=k_current / n_frames) ** 2
                                        dataset_batch[b].x[n_input_neurons:n_neurons, 4:5] = 1
                                    elif external_input_type == 'signal':
                                        t_norm = torch.tensor([[k_current / nnr_f_T_period]], dtype=torch.float32, device=device)
                                        if inr_type == 'siren_t':
                                            dataset_batch[b].x[:, 4] = model_f(t_norm).squeeze()
                                        elif inr_type == 'lowrank':
                                            t_idx = torch.tensor([k_current], dtype=torch.long, device=device)
                                            dataset_batch[b].x[:, 4] = model_f(t_idx).squeeze()
                                        elif inr_type == 'ngp':
                                            dataset_batch[b].x[:, 4] = model_f(t_norm).squeeze()
                                else:
                                    # use ground truth external_input from x_list
                                    x_next = torch.tensor(x_list[run][k_current], dtype=torch.float32, device=device)
                                    dataset_batch[b].x[:, 4:5] = x_next[:, 4:5]

                                dataset_batch_new.append(dataset_batch[b])
                            batch_loader_recur = DataLoader(dataset_batch_new, batch_size=batch_size, shuffle=False)
                            for batch in batch_loader_recur:
                                pred = model(batch, data_id=data_id, k=k_batch + step)
                            pred_x = pred_x + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

                    loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (delta_t * time_step)).norm(2)

                else:

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)



                 # PDE_N3 is special, embedding changes over time

                if 'PDE_N3' in model_config.signal_model_name:
                    loss = loss + train_config.coeff_model_a * (model.a[ind_a + 1] - model.a[ind_a]).norm(2)

                # omega L2 regularization for learnable omega in SIREN (encourages smaller omega)
                if model_f is not None and train_config.coeff_omega_f_L2 > 0:
                    if hasattr(model_f, 'get_omega_L2_loss'):
                        omega_L2_loss = model_f.get_omega_L2_loss()
                        loss = loss + train_config.coeff_omega_f_L2 * omega_L2_loss

                loss.backward()
                optimizer.step()
                regularizer.finalize_iteration()

                if has_missing_activity:
                    optimizer_missing_activity.step()
                if model_f is not None:
                    optimizer_f.step()

                regul_total_this_iter = regularizer.get_iteration_total()
                total_loss += loss.item()
                total_loss_regul += regul_total_this_iter

                if regularizer.should_record():
                    # store in dictionary lists
                    current_loss = loss.item()
                    loss_components['loss'].append((current_loss - regul_total_this_iter) / n_neurons)

                    last_connectivity_r2 = plot_training_signal(config, model, x, connectivity, log_dir, epoch, N, n_neurons, type_list, cmap, mc, device)
                    if last_connectivity_r2 is not None:
                        # color code: green (>0.95), yellow (0.7-0.95), orange (0.3-0.7), red (<0.3)
                        if last_connectivity_r2 > 0.9:
                            r2_color = '\033[92m'  # green
                        elif last_connectivity_r2 > 0.7:
                            r2_color = '\033[93m'  # yellow
                        elif last_connectivity_r2 > 0.3:
                            r2_color = '\033[38;5;208m'  # orange
                        else:
                            r2_color = '\033[91m'  # red
                        pbar.set_postfix_str(f'{r2_color}R²={last_connectivity_r2:.3f}\033[0m')

                    # merge loss_components with regularizer history for plotting
                    plot_dict = {**regularizer.get_history(), **loss_components}
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=N, debug=False,
                                   current_loss=current_loss / n_neurons, current_regul=regul_total_this_iter / n_neurons,
                                   total_loss=total_loss, total_loss_regul=total_loss_regul)

                    if model_f is not None:
                        torch.save({'model_state_dict': model_f.state_dict(),
                                    'optimizer_state_dict': optimizer_f.state_dict()},
                                   os.path.join(log_dir, 'models',
                                                f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                        # plot external_input learned vs ground truth (like train_INR)
                        with torch.no_grad():
                            external_input_gt = x_list[0][:, :, 4]  # (n_frames, n_neurons)
                            if external_input_type == 'visual':
                                # for visual input, just plot spatial snapshot (skip slow frame-by-frame loop)
                                n_input_neurons = simulation_config.n_input_neurons
                                plot_training_signal_visual_input(x, n_input_neurons, external_input_type, log_dir, epoch, N)
                                pred_all = None  # skip time series plot for visual
                            else:
                                nnr_f_T_period = model_config.nnr_f_T_period
                                if inr_type == 'siren_t':
                                    time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
                                    pred_all = model_f(time_input).cpu().numpy()
                                elif inr_type == 'lowrank':
                                    pred_all = model_f().cpu().numpy()
                                elif inr_type == 'ngp':
                                    time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
                                    pred_all = model_f(time_input).cpu().numpy()
                                else:
                                    pred_all = None
                            # plot predicted vs ground truth external input
                            if pred_all is not None:
                                gt_np = external_input_gt[:n_frames]  # ensure same length as pred
                                pred_np = pred_all
                                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                                fig.patch.set_facecolor('black')
                                ax.set_facecolor('black')
                                ax.set_axis_off()
                                n_traces = 10
                                trace_ids = np.linspace(0, n_neurons - 1, n_traces, dtype=int)
                                offset = np.abs(gt_np).max() * 1.5
                                t = np.arange(gt_np.shape[0])
                                for j, n_idx in enumerate(trace_ids):
                                    y0 = j * offset
                                    ax.plot(t, gt_np[:, n_idx] + y0, color='darkgreen', lw=2.0, alpha=0.95)
                                    ax.plot(t, pred_np[:, n_idx] + y0, color='white', lw=0.5, alpha=0.95)
                                ax.set_xlim(0, min(20000, gt_np.shape[0]))
                                ax.set_ylim(-offset * 0.5, offset * (n_traces + 0.5))
                                mse = ((pred_np - gt_np) ** 2).mean()
                                omega_str = ''
                                if hasattr(model_f, 'get_omegas'):
                                    omegas = model_f.get_omegas()
                                    if omegas:
                                        omega_str = f'  ω: {omegas[0]:.1f}'
                                ax.text(0.02, 0.98, f'MSE: {mse:.6f}{omega_str}', transform=ax.transAxes, va='top', ha='left', fontsize=12, color='white')
                                out_dir = os.path.join(log_dir, 'tmp_training', 'external_input')
                                os.makedirs(out_dir, exist_ok=True)
                                plt.tight_layout()
                                plt.savefig(f"{out_dir}/inr_{epoch}_{N}.png", dpi=150)
                                plt.close()

                    if has_missing_activity:
                        with torch.no_grad():
                            plot_training_signal_missing_activity(n_frames, k, x_list, baseline_value,
                                                                  model_missing_activity, log_dir, epoch, N, device)
                        torch.save({'model_state_dict': model_missing_activity.state_dict(),
                                    'optimizer_state_dict': optimizer_missing_activity.state_dict()},
                                   os.path.join(log_dir, 'models',
                                                f'best_model_missing_activity_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

            # Print progress when tqdm is disabled (subprocess mode)
            if tqdm_disabled and N > 0 and N % report_interval == 0:
                elapsed = time.time() - epoch_start_time
                progress_pct = (N / Niter) * 100
                steps_per_sec = N / elapsed if elapsed > 0 else 0
                eta_seconds = (Niter - N) / steps_per_sec if steps_per_sec > 0 else 0
                r2_str = f"R²={last_connectivity_r2:.3f}" if last_connectivity_r2 is not None else "R²=..."
                print(f"  {progress_pct:5.1f}% | step {N:6d}/{Niter} | {r2_str} | {steps_per_sec:.1f} it/s | eta: {eta_seconds/60:.1f}m", flush=True)

        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (total_loss - total_loss_regul) / n_neurons

        print("epoch {}. loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info("Epoch {}. Loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info(f'recurrent_parameters: {recurrent_parameters[0]:.2f}')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if model_f is not None:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_f.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))
            # Print omega values at end of each epoch
            if hasattr(model_f, 'get_omegas'):
                omegas = model_f.get_omegas()
                if omegas:
                    print(f"  model_f omegas after epoch {epoch}: {omegas}")

        list_loss.append(epoch_pred_loss)
        list_loss_regul.append(epoch_regul_loss)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color=mc, linewidth=1)
        plt.xlim([0, n_epochs])
        plt.ylabel('loss', fontsize=24)
        plt.xlabel('epochs', fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        embedding_files = glob.glob(f"./{log_dir}/tmp_training/embedding/*.tif")
        if embedding_files:
            last_file = max(embedding_files, key=os.path.getctime)  # or use os.path.getmtime for modification time
            filename = os.path.basename(last_file)
            last_epoch, last_N = filename.replace('.tif', '').split('_')

            # Load and display last saved figures
            from tifffile import imread

            def safe_load_and_display(ax, filepath):
                """Load and display image if file exists, otherwise leave panel empty."""
                if os.path.exists(filepath):
                    img = imread(filepath)
                    ax.imshow(img)
                ax.axis('off')

            # Plot 2: Last embedding
            ax = fig.add_subplot(2, 3, 2)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/embedding/{last_epoch}_{last_N}.tif")

            # Plot 3: Last weight comparison
            ax = fig.add_subplot(2, 3, 3)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.tif")

            # Plot 4: Last phi function
            ax = fig.add_subplot(2, 3, 4)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/function/MLP0/func_{last_epoch}_{last_N}.tif")

            # Plot 5: Last edge function
            ax = fig.add_subplot(2, 3, 5)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/function/MLP1/func_{last_epoch}_{last_N}.tif")

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.tif")
        plt.close()

        if replace_with_cluster:

            if (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):

                embedding = to_numpy(model.a.squeeze())
                model_MLP = model.lin_phi
                update_type = model.update_type

                func_list, proj_interaction = analyze_edge_function(rr=torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device),
                                                                     vizualize=True, config=config,
                                                                     model_MLP=model_MLP, model=model,
                                                                     n_nodes=0,
                                                                     n_neurons=n_neurons, ynorm=ynorm,
                                                                     type_list=to_numpy(x[:, 6]),  # neuron_type is at column 6
                                                                     cmap=cmap, update_type=update_type, device=device)

                labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding, train_config.cluster_distance_threshold, type_list, n_neuron_types, embedding_cluster)

                model_a_ = model.a.clone().detach()
                for n in range(n_clusters):
                    pos = np.argwhere(labels == n).squeeze().astype(int)
                    pos = np.array(pos)
                    if pos.size > 0:
                        median_center = model_a_[pos, :]
                        median_center = torch.median(median_center, dim=0).values
                        model_a_[pos, :] = median_center

                # Constrain embedding domain
                with torch.no_grad():
                    model.a.copy_(model_a_)
                print('regul_embedding: replaced')
                logger.info('regul_embedding: replaced')

                # Constrain function domain
                if train_config.sparsity == 'replace_embedding_function':

                    logger.info('replace_embedding_function')
                    y_func_list = func_list * 0

                    fig.add_subplot(2, 5, 9)
                    for n in np.unique(new_labels):
                        pos = np.argwhere(new_labels == n)
                        pos = pos.squeeze()
                        if pos.size > 0:
                            target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                            y_func_list[pos] = target_func
                        plt.plot(to_numpy(target_func) * to_numpy(ynorm), linewidth=2, alpha=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()

                    lr_embedding = 1E-12
                    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                         lr_update=lr_update, lr_W=lr_W,
                                                                         lr_modulation=lr_modulation)
                    for sub_epochs in trange(20, ncols=100):
                        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
                        pred = []
                        optimizer.zero_grad()
                        for n in range(n_neurons):
                            embedding = model.a[n, :].clone().detach() * torch.ones((1000, model_config.embedding_dim),
                                                                                     device=device)
                            in_features = get_in_features_update(rr=rr[:, None], model=model, embedding=embedding, device=device)
                            pred.append(model.lin_phi(in_features.float()))
                        pred = torch.stack(pred)
                        loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                        logger.info(f'    loss: {np.round(loss.item() / n_neurons, 3)}')
                        loss.backward()
                        optimizer.step()
                if train_config.fix_cluster_embedding:
                    lr = 1E-12
                    lr_embedding = 1E-12
                    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                         lr_update=lr_update, lr_W=lr_W,
                                                                         lr_modulation=lr_modulation)
                    logger.info(
                        f'learning rates: lr_W {lr_W}, lr {lr}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')
            else:
                lr = train_config.learning_rate_start
                lr_embedding = train_config.learning_rate_embedding_start
                optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                     lr_update=lr_update, lr_W=lr_W,
                                                                     lr_modulation=lr_modulation)
                logger.info( f'learning rates: lr_W {lr_W}, lr {lr}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')

    # Calculate and log training time
    training_time = time.time() - training_start_time
    training_time_min = training_time / 60.0
    print(f"training completed in {training_time_min:.1f} minutes")
    logger.info(f"training completed in {training_time_min:.1f} minutes")

    if log_file is not None:
        log_file.write(f"training_time_min: {training_time_min:.1f}\n")
        log_file.write(f"n_epochs: {n_epochs}\n")
        log_file.write(f"data_augmentation_loop: {data_augmentation_loop}\n")
        log_file.write(f"time_step: {time_step}\n")
        log_file.write(f"recurrent_training: {recurrent_training}\n")
        log_file.write(f"batch_size: {target_batch_size}\n")
        log_file.write(f"learning_rate_W: {lr_W}\n")
        log_file.write(f"learning_rate: {train_config.learning_rate_start}\n")
        log_file.write(f"coeff_W_L1: {train_config.coeff_W_L1}\n")



def data_train_INR(config=None, device=None, total_steps=5000, erase=False):
    """
    Train nnr_f (SIREN/INR network) on external_input data from x_list.

    This pre-trains the implicit neural representation (INR) network before
    joint learning with GNN. The INR learns to map time -> external_input
    for all neurons.

    INR types:
        siren_t: input=t, output=n_neurons (works for n_neurons < 100)
        siren_id: input=(t, id/n_neurons), output=1 (scales better for large n_neurons)
        siren_x: input=(t, x, y), output=1 (uses neuron positions)
        ngp: instantNGP hash encoding

    Args:
        config: NeuralGraphConfig object
        device: torch device
        total_steps: number of training steps (default: 5000)
        erase: whether to erase existing log files (default: False)

    Returns:
        nnr_f: trained SIREN model
        loss_list: list of training losses
    """

    # create log directory
    log_dir, logger = create_log_dir(config, erase)
    output_folder = os.path.join(log_dir, 'tmp_training', 'external_input')
    os.makedirs(output_folder, exist_ok=True)

    dataset_name = config.dataset
    data_folder = f"graphs_data/{dataset_name}/"
    print(f"loading data from: {data_folder}")

    # load x_list data
    x_list = np.load(f"{data_folder}x_list_0.npy")
    print(f"x_list shape: {x_list.shape}")  # (n_frames, n_neurons, n_features)

    n_frames, n_neurons, n_features = x_list.shape
    print(f"n_frames: {n_frames}, n_neurons: {n_neurons}, n_features: {n_features}")

    # extract external_input from x_list (column 4)
    external_input = x_list[:, :, 4]  # shape: (n_frames, n_neurons)

    # SVD analysis
    U, S, Vt = np.linalg.svd(external_input, full_matrices=False)

    # effective rank
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    rank_90 = np.searchsorted(cumvar, 0.90) + 1
    rank_99 = np.searchsorted(cumvar, 0.99) + 1

    print(f"effective rank (90% var): {rank_90}")
    print(f"effective rank (99% var): {rank_99}")
    print(f"compression: {n_frames * n_neurons / (rank_99 * (n_frames + n_neurons)):.1f}x")
    print()

    # extract neuron positions from x_list (columns 1, 2) - use first frame as reference
    neuron_positions = x_list[0, :, 1:3]  # shape: (n_neurons, 2)


    # extract neuron ids from x_list (column 0)
    neuron_ids = x_list[0, :, 0]  # shape: (n_neurons,)

    # get nnr_f config parameters
    model_config = config.graph_model
    hidden_dim_nnr_f = getattr(model_config, 'hidden_dim_nnr_f', 1024)
    n_layers_nnr_f = getattr(model_config, 'n_layers_nnr_f', 3)
    outermost_linear_nnr_f = getattr(model_config, 'outermost_linear_nnr_f', True)
    omega_f = getattr(model_config, 'omega_f', 1024)
    nnr_f_T_period = getattr(model_config, 'nnr_f_T_period', 10000)

    # get training config parameters
    training_config = config.training
    batch_size = getattr(training_config, 'batch_size', 8)
    learning_rate = getattr(training_config, 'learning_rate_NNR_f', 1e-6)

    # get simulation config for calculation check
    sim_config = config.simulation
    delta_t = getattr(sim_config, 'delta_t', 0.01)
    oscillation_frequency = getattr(sim_config, 'oscillation_frequency', 0.1)

    # calculation check
    total_sim_time = n_frames * delta_t
    period_time_units = 1.0 / oscillation_frequency if oscillation_frequency > 0 else float('inf')
    period_frames = period_time_units / delta_t if oscillation_frequency > 0 else float('inf')
    total_cycles = total_sim_time / period_time_units if oscillation_frequency > 0 else 0
    normalized_time_max = n_frames / nnr_f_T_period
    cycles_in_normalized_range = total_cycles * normalized_time_max
    recommended_omega = 2 * np.pi * cycles_in_normalized_range

    # get INR type from config
    inr_type = getattr(model_config, 'inr_type', 'siren_t')

    print("siren calculation check:")
    print(f"  total simulation time: {n_frames} × {delta_t} = {total_sim_time:.1f} time units")
    print(f"  period: 1/{oscillation_frequency} = {period_time_units:.1f} time units = {period_frames:.0f} frames")
    print(f"  total cycles: {total_cycles:.0f}")
    print(f"  normalized input range: [0, {n_frames}/{nnr_f_T_period}] = [0, {normalized_time_max:.2f}]")
    print(f"  cycles in normalized range: {total_cycles:.0f} × {normalized_time_max:.2f} = {cycles_in_normalized_range:.1f}")
    print(f"  recommended omega_f: 2π × {cycles_in_normalized_range:.1f} ≈ {recommended_omega:.0f}")
    print(f"  omega_f (config): {omega_f}")
    if omega_f > 5 * recommended_omega:
        print(f"  ⚠️  omega_f is {omega_f/recommended_omega:.1f}× recommended — may cause slow convergence")

    # data dimensions to learn
    data_dims = n_frames * n_neurons
    print(f"\ndata to learn: {n_frames:,} frames × {n_neurons:,} neurons = {data_dims:,.0f} values")

    # determine input/output dimensions based on inr_type
    if inr_type == 'siren_t':
        input_size_nnr_f = 1
        output_size_nnr_f = n_neurons
    elif inr_type == 'siren_id':
        input_size_nnr_f = 2  # (t, id)
        output_size_nnr_f = 1
    elif inr_type == 'siren_x':
        input_size_nnr_f = 3  # (t, x, y)
        output_size_nnr_f = 1
    elif inr_type == 'ngp':
        input_size_nnr_f = getattr(model_config, 'input_size_nnr_f', 1)
        output_size_nnr_f = getattr(model_config, 'output_size_nnr_f', n_neurons)
    elif inr_type == 'lowrank':
        # lowrank doesn't use input/output sizes in the same way
        pass
    else:
        raise ValueError(f"unknown inr_type: {inr_type}")

    # create INR model based on type
    if inr_type == 'ngp':

        # get NGP config parameters
        ngp_n_levels = getattr(model_config, 'ngp_n_levels', 24)
        ngp_n_features_per_level = getattr(model_config, 'ngp_n_features_per_level', 2)
        ngp_log2_hashmap_size = getattr(model_config, 'ngp_log2_hashmap_size', 22)
        ngp_base_resolution = getattr(model_config, 'ngp_base_resolution', 16)
        ngp_per_level_scale = getattr(model_config, 'ngp_per_level_scale', 1.4)
        ngp_n_neurons = getattr(model_config, 'ngp_n_neurons', 128)
        ngp_n_hidden_layers = getattr(model_config, 'ngp_n_hidden_layers', 4)

        nnr_f = HashEncodingMLP(
            n_input_dims=input_size_nnr_f,
            n_output_dims=output_size_nnr_f,
            n_levels=ngp_n_levels,
            n_features_per_level=ngp_n_features_per_level,
            log2_hashmap_size=ngp_log2_hashmap_size,
            base_resolution=ngp_base_resolution,
            per_level_scale=ngp_per_level_scale,
            n_neurons=ngp_n_neurons,
            n_hidden_layers=ngp_n_hidden_layers,
            output_activation='none'
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        encoding_params = sum(p.numel() for p in nnr_f.encoding.parameters())
        mlp_params = sum(p.numel() for p in nnr_f.mlp.parameters())
        total_params = encoding_params + mlp_params

        print("\nusing HashEncodingMLP (instantNGP):")
        print(f"  hash encoding: {ngp_n_levels} levels × {ngp_n_features_per_level} features")
        print(f"  hash table: 2^{ngp_log2_hashmap_size} = {2**ngp_log2_hashmap_size:,} entries")
        print(f"  mlp: {ngp_n_neurons} × {ngp_n_hidden_layers} hidden → {output_size_nnr_f}")
        print(f"  parameters: {total_params:,} (encoding: {encoding_params:,}, mlp: {mlp_params:,})")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")

    elif inr_type in ['siren_t', 'siren_id', 'siren_x']:
        # create SIREN model for nnr_f
        omega_f_learning = getattr(model_config, 'omega_f_learning', False)
        nnr_f = Siren(
            in_features=input_size_nnr_f,
            hidden_features=hidden_dim_nnr_f,
            hidden_layers=n_layers_nnr_f,
            out_features=output_size_nnr_f,
            outermost_linear=outermost_linear_nnr_f,
            first_omega_0=omega_f,
            hidden_omega_0=omega_f,
            learnable_omega=omega_f_learning
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print(f"\nusing SIREN ({inr_type}):")
        print(f"  architecture: {input_size_nnr_f} → {hidden_dim_nnr_f} × {n_layers_nnr_f} hidden → {output_size_nnr_f}")
        print(f"  omega_f: {omega_f} (learnable: {omega_f_learning})")
        if omega_f_learning and hasattr(nnr_f, 'get_omegas'):
            print(f"  initial omegas: {nnr_f.get_omegas()}")
        print(f"  parameters: {total_params:,}")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")

    elif inr_type == 'lowrank':
        # get lowrank config parameters
        lowrank_rank = getattr(model_config, 'lowrank_rank', 64)
        lowrank_svd_init = getattr(model_config, 'lowrank_svd_init', True)

        # create LowRankINR model
        init_data = external_input if lowrank_svd_init else None
        nnr_f = LowRankINR(
            n_frames=n_frames,
            n_neurons=n_neurons,
            rank=lowrank_rank,
            init_data=init_data
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print("\nusing LowRankINR:")
        print(f"  rank: {lowrank_rank}")
        print(f"  U: ({n_frames}, {lowrank_rank}), V: ({lowrank_rank}, {n_neurons})")
        print(f"  parameters: {total_params:,}")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")
        print(f"  SVD init: {lowrank_svd_init}")

    print(f"\ntraining: batch_size={batch_size}, learning_rate={learning_rate}")

    # prepare training data
    ground_truth = torch.tensor(external_input, dtype=torch.float32, device=device)  # (n_frames, n_neurons)

    # prepare inputs based on inr_type
    if inr_type == 'siren_t':
        # input: normalized time, output: all neurons
        time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period

    elif inr_type == 'siren_id':
        # input: (t, id), output: 1
        # normalize id by n_neurons
        neuron_ids_norm = torch.tensor(neuron_ids / n_neurons, dtype=torch.float32, device=device)  # (n_neurons,)

    elif inr_type == 'siren_x':
        # input: (t, x, y), output: 1
        # positions are already normalized
        neuron_pos = torch.tensor(neuron_positions, dtype=torch.float32, device=device)  # (n_neurons, 2)

    steps_til_summary = 5000

    # Separate omega parameters from other parameters for different learning rates
    omega_f_learning = getattr(model_config, 'omega_f_learning', False)
    learning_rate_omega_f = getattr(training_config, 'learning_rate_omega_f', learning_rate)
    omega_params = [p for name, p in nnr_f.named_parameters() if 'omega' in name]
    other_params = [p for name, p in nnr_f.named_parameters() if 'omega' not in name]
    if omega_params and omega_f_learning:
        optim = torch.optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': omega_params, 'lr': learning_rate_omega_f}
        ])
        print(f"using separate learning rates: network={learning_rate}, omega={learning_rate_omega_f}")
    else:
        optim = torch.optim.Adam(lr=learning_rate, params=nnr_f.parameters())

    print(f"training nnr_f for {total_steps} steps...")

    loss_list = []
    pbar = trange(total_steps + 1, ncols=150)
    for step in pbar:

        if inr_type == 'siren_t':
            # sample batch_size time frames
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = time_input[sample_ids]  # (batch_size, 1)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_neurons)
            pred = nnr_f(time_batch)  # (batch_size, n_neurons)

        elif inr_type == 'siren_id':
            # sample batch_size time frames, predict all neurons for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all neurons: (batch_size, n_neurons, 2)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_neurons, 1)
            id_expanded = neuron_ids_norm[None, :, None].expand(batch_size, n_neurons, 1)
            input_batch = torch.cat([t_expanded, id_expanded], dim=2)  # (batch_size, n_neurons, 2)
            input_batch = input_batch.reshape(batch_size * n_neurons, 2)  # (batch_size * n_neurons, 2)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_neurons)  # (batch_size * n_neurons,)
            pred = nnr_f(input_batch).squeeze()  # (batch_size * n_neurons,)

        elif inr_type == 'siren_x':
            # sample batch_size time frames, predict all neurons for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all neurons: (batch_size, n_neurons, 3)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_neurons, 1)
            pos_expanded = neuron_pos[None, :, :].expand(batch_size, n_neurons, 2)
            input_batch = torch.cat([t_expanded, pos_expanded], dim=2)  # (batch_size, n_neurons, 3)
            input_batch = input_batch.reshape(batch_size * n_neurons, 3)  # (batch_size * n_neurons, 3)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_neurons)  # (batch_size * n_neurons,)
            pred = nnr_f(input_batch).squeeze()  # (batch_size * n_neurons,)

        elif inr_type == 'ngp':
            # sample batch_size time frames (same as siren_t)
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device).unsqueeze(1)
            gt_batch = ground_truth[sample_ids]
            pred = nnr_f(time_batch)

        elif inr_type == 'lowrank':
            # sample batch_size time frames
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_indices = torch.tensor(sample_ids, dtype=torch.long, device=device)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_neurons)
            pred = nnr_f(t_indices)  # (batch_size, n_neurons)

        # compute loss
        if inr_type == 'ngp':
            # relative L2 error - convert targets to match output dtype (tcnn uses float16)
            relative_l2_error = (pred - gt_batch.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)
            loss = relative_l2_error.mean()
        else:
            # standard MSE for SIREN
            loss = ((pred - gt_batch) ** 2).mean()

        # omega L2 regularization for learnable omega in SIREN (encourages smaller omega)
        coeff_omega_f_L2 = getattr(training_config, 'coeff_omega_f_L2', 0.0)
        if omega_f_learning and coeff_omega_f_L2 > 0 and hasattr(nnr_f, 'get_omega_L2_loss'):
            omega_L2_loss = nnr_f.get_omega_L2_loss()
            loss = loss + coeff_omega_f_L2 * omega_L2_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.6f}")

        if step % steps_til_summary == 0:
            with torch.no_grad():
                # compute predictions for all frames
                if inr_type == 'siren_t':
                    pred_all = nnr_f(time_input)  # (n_frames, n_neurons)

                elif inr_type == 'siren_id':
                    # predict all (t, id) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                        input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)  # (n_neurons, 2)
                        pred_t = nnr_f(input_t).squeeze()  # (n_neurons,)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_neurons)

                elif inr_type == 'siren_x':
                    # predict all (t, x, y) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                        input_t = torch.cat([t_val, neuron_pos], dim=1)  # (n_neurons, 3)
                        pred_t = nnr_f(input_t).squeeze()  # (n_neurons,)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_neurons)

                elif inr_type == 'ngp':
                    time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
                    pred_all = nnr_f(time_all)

                elif inr_type == 'lowrank':
                    pred_all = nnr_f()  # returns full (n_frames, n_neurons) matrix

                gt_np = ground_truth.cpu().numpy()
                pred_np = pred_all.cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.patch.set_facecolor('black')

                # loss plot
                axes[0].set_facecolor('black')
                axes[0].plot(loss_list, color='white', lw=0.1)
                axes[0].set_xlabel('step', color='white', fontsize=12)
                loss_label = 'Relative L2 Loss' if inr_type == 'ngp' else 'MSE Loss'
                axes[0].set_ylabel(loss_label, color='white', fontsize=12)
                axes[0].set_yscale('log')
                axes[0].tick_params(colors='white', labelsize=11)
                for spine in axes[0].spines.values():
                    spine.set_color('white')

                # traces plot (10 neurons, darkgreen=GT, white=pred)
                axes[1].set_facecolor('black')
                axes[1].set_axis_off()
                n_traces = 10
                trace_ids = np.linspace(0, n_neurons - 1, n_traces, dtype=int)
                offset = np.abs(gt_np).max() * 1.5
                t = np.arange(n_frames)

                for j, n_idx in enumerate(trace_ids):
                    y0 = j * offset
                    axes[1].plot(t, gt_np[:, n_idx] + y0, color='darkgreen', lw=2.0, alpha=0.95)
                    axes[1].plot(t, pred_np[:, n_idx] + y0, color='white', lw=0.5, alpha=0.95)

                axes[1].set_xlim(0, min(20000, n_frames))
                axes[1].set_ylim(-offset * 0.5, offset * (n_traces + 0.5))
                mse = ((pred_np - gt_np) ** 2).mean()
                omega_str = ''
                if hasattr(nnr_f, 'get_omegas'):
                    omegas = nnr_f.get_omegas()
                    if omegas:
                        omega_str = f'  ω: {omegas[0]:.1f}'
                axes[1].text(0.02, 0.98, f'MSE: {mse:.6f}{omega_str}',
                            transform=axes[1].transAxes, va='top', ha='left',
                            fontsize=12, color='white')

                plt.tight_layout()
                plt.savefig(f"{output_folder}/{inr_type}_{step}.png", dpi=150)
                plt.close()

    # save trained model
    # save_path = f"{output_folder}/nnr_f_{inr_type}_pretrained.pt"
    # torch.save(nnr_f.state_dict(), save_path)
    # print(f"\nsaved pretrained nnr_f to: {save_path}")

    # compute final MSE
    with torch.no_grad():
        if inr_type == 'siren_t':
            pred_all = nnr_f(time_input)
        elif inr_type == 'siren_id':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)
                pred_t = nnr_f(input_t).squeeze()
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'siren_x':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                input_t = torch.cat([t_val, neuron_pos], dim=1)
                pred_t = nnr_f(input_t).squeeze()
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'ngp':
            time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
            pred_all = nnr_f(time_all)

        final_mse = ((pred_all - ground_truth) ** 2).mean().item()
        print(f"final MSE: {final_mse:.6f}")
        if hasattr(nnr_f, 'get_omegas'):
            print(f"final omegas: {nnr_f.get_omegas()}")

    return nnr_f, loss_list







def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, n_rollout_frames=600,
              ratio=1, run=0, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[],
              rollout_without_noise: bool = False, log_file=None):

    dataset_name = config.dataset
    print(f"\033[94mdataset_name: {dataset_name}\033[0m")
    print(f"\033[92m{config.description}\033[0m")

    if test_mode == "":
        test_mode = "test_ablation_0"

    data_test_signal(config, config_file, visualize, style, verbose, best_model, step, n_rollout_frames,ratio, run, test_mode, sample_embedding, particle_of_interest, new_params, device, log_file)



def data_test_signal(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, n_rollout_frames=600, ratio=1, run=0, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[], log_file=None):
    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons
    n_input_neurons = simulation_config.n_input_neurons
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    time_window = training_config.time_window
    time_step = training_config.time_step
    neural_ODE_training = training_config.neural_ODE_training
    ode_method = training_config.ode_method
    ode_rtol = training_config.ode_rtol
    ode_atol = training_config.ode_atol

    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension

    has_missing_activity = training_config.has_missing_activity
    has_excitation = ('excitation' in model_config.update_type)
    baseline_value = simulation_config.baseline_value

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    if 'latex' in style:
        print('latex style...')
        plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'


    field_type = model_config.field_type
    # fallback to external_input_type for signal configs that don't have field_type
    external_input_type = getattr(simulation_config, 'external_input_type', '')
    if field_type == '' and external_input_type != '':
        field_type = external_input_type
    if field_type != '':
        n_input_neurons_per_axis = int(np.sqrt(n_input_neurons))

    log_dir = 'log/' + config.config_file
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)
    os.makedirs(f'./{log_dir}/results/Fig', exist_ok=True)
    files = glob.glob(f"./{log_dir}/results/Fig/*")
    for f in files:
        os.remove(f)

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')
    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"

    print('load data...')

    x_list = []
    y_list = []

    if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
        x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    else:
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        y = torch.tensor(y, dtype=torch.float32, device=device)
        x_list.append(x)
        y_list.append(y)
        x = x_list[0][0].clone().detach()
        n_neurons = int(x.shape[0])
        config.simulation.n_neurons = n_neurons
        n_frames = len(x_list[0])
        index_particles = get_index_particles(x, n_neuron_types, dimension)
        if n_neuron_types > 1000:
            index_particles = []
            for n in range(3):
                index = np.arange(n_neurons * n // 3, n_neurons * (n + 1) // 3)
                index_particles.append(index)
                n_neuron_types = 3
    ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
    vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)
    if vnorm == 0:
        vnorm = ynorm

    connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)
    if training_config.with_connectivity_mask:
        model_mask = (connectivity > 0) * 1.0
        adj_t = model_mask.float() * 1
        adj_t = adj_t.t()
        edge_index = adj_t.nonzero().t().contiguous()
    else:
        edge_index = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)

    edge_index_generated = edge_index.clone().detach()


    if ('modulation' in model_config.field_type) | ('visual' in model_config.field_type):
        print('load gt movie ...')
        im = imread(f"graphs_data/{simulation_config.node_value_map}")
        A1 = torch.zeros((n_neurons, 1), device=device)

        # neuron_index = torch.randint(0, n_neurons, (6,))
        neuron_gt_list = []
        neuron_pred_list = []
        modulation_gt_list = []
        modulation_pred_list = []

        if os.path.exists(f'./graphs_data/{dataset_name}/X1.pt') > 0:
            X1_first = torch.load(f'./graphs_data/{dataset_name}/X1.pt', map_location=device)
            X_msg = torch.load(f'./graphs_data/{dataset_name}/X_msg.pt', map_location=device)
        else:
            xc, yc = get_equidistant_points(n_points=n_neurons)
            X1_first = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
            perm = torch.randperm(X1_first.size(0))
            X1_first = X1_first[perm]
            torch.save(X1_first, f'./graphs_data/{dataset_name}/X1_first.pt')
            xc, yc = get_equidistant_points(n_points=n_neurons ** 2)
            X_msg = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
            perm = torch.randperm(X_msg.size(0))
            X_msg = X_msg[perm]
            torch.save(X_msg, f'./graphs_data/{dataset_name}/X_msg.pt')

    model_generator, bc_pos, bc_dpos = choose_model(config=config, W=connectivity, device=device)

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    model.particle_of_interest = particle_of_interest
    if training_config.with_connectivity_mask:
        model.mask = (connectivity > 0) * 1.0
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    state_dict = torch.load(net, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)
    model.eval()


    # only load model_f if learn_external_input was True during training
    model_f = None  # initialize to None, will be loaded if needed
    learn_external_input = training_config.learn_external_input
    if learn_external_input and (('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type)):
        model_f = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                        hidden_features=model_config.hidden_dim_nnr,
                        hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                        hidden_omega_0=model_config.omega,
                        outermost_linear=model_config.outermost_linear_nnr)
        net = f'{log_dir}/models/best_model_f_with_0_graphs_{best_model}.pt'
        state_dict = torch.load(net, map_location=device)
        model_f.load_state_dict(state_dict['model_state_dict'])
        model_f.to(device=device)
        model_f.eval()
    if learn_external_input and (('modulation' in model_config.field_type) | ('visual' in model_config.field_type)):
        # use _nnr_f params to match choose_inr_model in training
        # use n_input_neurons (same as training) for image_width
        n_input_neurons = simulation_config.n_input_neurons
        n_input_neurons_per_axis = int(np.sqrt(n_input_neurons))
        model_f = Siren_Network(image_width=n_input_neurons_per_axis, in_features=model_config.input_size_nnr_f,
                                out_features=model_config.output_size_nnr_f,
                                hidden_features=model_config.hidden_dim_nnr_f,
                                hidden_layers=model_config.n_layers_nnr_f,
                                outermost_linear=model_config.outermost_linear_nnr_f,
                                device=device,
                                first_omega_0=model_config.omega_f, hidden_omega_0=model_config.omega_f)
        net = f'{log_dir}/models/best_model_f_with_0_graphs_{best_model}.pt'
        state_dict = torch.load(net, map_location=device)
        model_f.load_state_dict(state_dict['model_state_dict'])
        model_f.to(device=device)
        model_f.eval()
    if has_missing_activity:
        model_missing_activity = nn.ModuleList([
            Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                    hidden_features=model_config.hidden_dim_nnr,
                    hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                    hidden_omega_0=model_config.omega,
                    outermost_linear=model_config.outermost_linear_nnr)
            for n in range(n_runs)
        ])
        model_missing_activity.to(device=device)
        net = f'{log_dir}/models/best_model_missing_activity_with_{n_runs - 1}_graphs_{best_model}.pt'
        state_dict = torch.load(net, map_location=device)
        model_missing_activity.load_state_dict(state_dict['model_state_dict'])
        model_missing_activity.to(device=device)
        model_missing_activity.eval()

    rmserr_list = []
    geomloss_list = []
    angle_list = []
    time.sleep(1)

    if time_window > 0:
        start_it = time_window
        n_frames - 1
    else:
        start_it = 0
        n_frames - 1

    start_it = 0

    x = x_list[0][start_it].clone().detach()
    x_generated = x_list[0][start_it].clone().detach()

    if 'test_ablation' in test_mode:
        #  test_mode="test_ablation_0 by default
        ablation_ratio = int(test_mode.split('_')[-1]) / 100
        if ablation_ratio > 0:
            print(f'\033[93mtest ablation ratio {ablation_ratio} \033[0m')
            n_ablation = int(n_neurons * ablation_ratio)
            index_ablation = np.random.choice(np.arange(n_neurons), n_ablation, replace=False)
            with torch.no_grad():
                model.W[index_ablation, :] = 0
                model_generator.W[index_ablation, :] = 0
    else:
        ablation_ratio = 0

    if 'test_inactivity' in test_mode:
        #  test_mode="test_inactivity_100"
        inactivity_ratio = int(test_mode.split('_')[-1]) / 100
        if inactivity_ratio > 0:
            print(f'\033[93mtest inactivity ratio {inactivity_ratio} \033[0m')
        n_inactivity = int(n_neurons * inactivity_ratio)
        index_inactivity = np.random.choice(np.arange(n_neurons), n_inactivity, replace=False)

        x[index_inactivity, 6] = 0
        x_generated[index_inactivity, 6] = 0

        with torch.no_grad():
            model.W[index_inactivity, :] = 0
            model.W[:, index_inactivity] = 0
            model_generator.W[index_inactivity, :] = 0
            model_generator.W[:, index_inactivity] = 0
    else:
        inactivity_ratio = 0

    if 'test_permutation' in test_mode:
        permutation_ratio = int(test_mode.split('_')[-1]) / 100
        if permutation_ratio > 0:
            print(f'\033[93mtest permutation ratio {permutation_ratio} \033[0m')
        n_permutation = int(n_neurons * permutation_ratio)
        index_permutation = np.random.choice(np.arange(n_neurons), n_permutation, replace=False)
        rnd_perm = torch.randperm(n_permutation)

        x_permuted = x[index_permutation, 5].clone().detach()
        x_generated_permuted = x_generated[index_permutation, 5].clone().detach()

        x[index_permutation, 5] = x_permuted[rnd_perm]
        x_generated[index_permutation, 5] = x_generated_permuted[rnd_perm]

        a_permuted = model.a[index_permutation].clone().detach()
        with torch.no_grad():
            model.a[index_permutation] = a_permuted[rnd_perm]
    else:
        permutation_ratio = 0

    if new_params is not None:
        print('set new parameters for testing ...')


        plt.figure(figsize=(10, 10))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        edge_index_, connectivity, mask = init_connectivity(
                simulation_config.connectivity_file,
                simulation_config.connectivity_type,
                simulation_config.connectivity_filling_factor,
                new_params[0],
                n_neurons,
                n_neuron_types,
                dataset_name,
                device,
                connectivity_rank=simulation_config.connectivity_rank,
                Dale_law=simulation_config.Dale_law,
                Dale_law_factor=simulation_config.Dale_law_factor,
            )


        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/new connectivity.png', dpi=300)
        plt.close()

        second_correction = np.load(f'{log_dir}/second_correction.npy')
        print(f'second_correction: {second_correction}')

        with torch.no_grad():
            model_generator.W = torch.nn.Parameter(torch.tensor(connectivity, device=device))
            model.W = torch.nn.Parameter(model_generator.W.clone() * torch.tensor(second_correction, device=device))

        cell_types = to_numpy(x[:, 6]).astype(int)
        type_counts = [np.sum(cell_types == n) for n in range(n_neuron_types)]
        plt.figure(figsize=(10, 10))
        plt.bar(range(n_neuron_types), type_counts,
                    color=[cmap.color(n) for n in range(n_neuron_types)])

        plt.xlabel('neuron type', fontsize=48)
        plt.ylabel('count', fontsize=48)
        plt.xticks(range(n_neuron_types),fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(axis='y', alpha=0.3)
        for i, count in enumerate(type_counts):
            plt.text(i, count + max(type_counts)*0.01, str(count),
                    ha='center', va='bottom', fontsize=32)
        plt.ylim(0, 1000)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/neuron_type_histogram.png', dpi=300)
        plt.close()


        first_cell_id_neurons = []
        for n in range(n_neuron_types):
            index = np.arange(n_neurons * n // n_neuron_types, n_neurons * (n + 1) // n_neuron_types)
            first_cell_id_neurons.append(index)
        id=0
        for n in range(n_neuron_types):
            print(f'neuron type {n}, first cell id {id}')
            x[id:id+int(new_params[n+1]*n_neurons/100), 6] = n
            x_generated[id:id+int(new_params[n+1]*n_neurons/100), 6] = n
            id = id + int(new_params[n+1]*n_neurons/100)
        print(f'last cell id {id}, total number of neurons {n_neurons}')

        first_embedding = model.a.clone().detach()
        model_a_ = nn.Parameter(torch.tensor(np.ones((int(n_neurons), model.embedding_dim)), device=device, requires_grad=False,dtype=torch.float32))
        for n in range(n_neurons):
            t = to_numpy(x[n, 6]).astype(int)
            index = first_cell_id_neurons[t][np.random.randint(len(first_cell_id_neurons[t]))]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((int(n_neurons), model.embedding_dim)), device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_[n]

                cell_types = to_numpy(x[:, 6]).astype(int)
        type_counts = [np.sum(cell_types == n) for n in range(n_neuron_types)]
        plt.figure(figsize=(10, 10))
        plt.bar(range(n_neuron_types), type_counts,
                    color=[cmap.color(n) for n in range(n_neuron_types)])

        plt.xlabel('neuron type', fontsize=48)
        plt.ylabel('count', fontsize=48)
        plt.xticks(range(n_neuron_types),fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(axis='y', alpha=0.3)
        for i, count in enumerate(type_counts):
            plt.text(i, count + max(type_counts)*0.01, str(count),
                    ha='center', va='bottom', fontsize=32)
        plt.ylim(0, 1000)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/new_neuron_type_histogram.png', dpi=300)
        plt.close()

    # n_neurons = x.shape[0]
    neuron_gt_list = []
    neuron_pred_list = []
    neuron_generated_list = []

    x_inference_list = []
    x_generated_list = []

    R2_list = []
    it_list = []
    id_fig = 0


    n_test_frames = n_rollout_frames
    it_step = step

    print('rollout inference...')

    for it in trange(start_it, start_it + n_test_frames, ncols=100):

        if it < n_frames - 4:
            x0 = x_list[0][it].clone().detach()
            x0_next = x_list[0][(it + time_step)].clone().detach()
            y_list[0][it].clone().detach()
        if has_excitation:
            x[:, 10: 10 + model_config.excitation_dim] = x0[:, 10: 10 + model_config.excitation_dim]

        x0[:, 3] = torch.where(torch.isnan(x0[:, 3]), baseline_value, x0[:, 3])
        x[:, 3]  = torch.where(torch.isnan(x[:, 3]),  baseline_value, x[:, 3])
        x_generated[:, 3] = torch.where(torch.isnan(x_generated[:, 3]), baseline_value, x_generated[:, 3])


        x_inference_list.append(x[:, 3:4].clone().detach())
        x_generated_list.append(x_generated[:, 3:4].clone().detach())

        if ablation_ratio > 0:
            rmserr = torch.sqrt(torch.mean((x_generated[:n_neurons, 3] - x0[:, 3]) ** 2))
        else:
            rmserr = torch.sqrt(torch.mean((x[:n_neurons, 3] - x0[:, 3]) ** 2))

        neuron_gt_list.append(x0[:, 3:4])
        neuron_pred_list.append(x[:n_neurons, 3:4].clone().detach())
        neuron_generated_list.append(x_generated[:n_neurons, 3:4].clone().detach())

        if ('short_term_plasticity' in field_type) | ('modulation' in field_type):
            modulation_gt_list.append(x0[:, 4:5])
            modulation_pred_list.append(x[:, 4:5].clone().detach())
        rmserr_list.append(rmserr.item())

        if config.training.shared_embedding:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device)
        else:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device) * run

        # update calculations
        if 'visual' in field_type:
            if model_f is not None:
                x[:n_input_neurons, 4:5] = model_f(time=it / n_frames) ** 2
            else:
                # fallback: use ground truth from image file
                im_ = im[int(it / n_frames * 256)].squeeze()
                im_ = np.rot90(im_, 3)
                im_ = np.reshape(im_, (n_input_neurons_per_axis * n_input_neurons_per_axis))
                x[:n_input_neurons, 4:5] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)
            x[n_input_neurons:n_neurons, 4:5] = 1
        elif 'learnable_short_term_plasticity' in field_type:
            alpha = (it % model.embedding_step) / model.embedding_step
            x[:, 4] = alpha * model.b[:, it // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,
                                                                                                it // model.embedding_step] ** 2
        elif ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
            if model_f is not None:
                t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                t[:, 0, :] = torch.tensor(it / n_frames, dtype=torch.float32, device=device)
                x[:, 4] = model_f(t).squeeze() ** 2
        elif 'modulation' in field_type:
            if model_f is not None:
                x[:, 4:5] = model_f(time=it / n_frames) ** 2
            else:
                # fallback: use ground truth from image file
                im_ = im[int(it / n_frames * 256)].squeeze()
                im_ = np.rot90(im_, 3)
                im_ = np.reshape(im_, (n_input_neurons_per_axis * n_input_neurons_per_axis))
                x[:, 4:5] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)

        if has_missing_activity:
            t = torch.tensor([it / n_frames], dtype=torch.float32, device=device)
            missing_activity = baseline_value + model_missing_activity[run](t).squeeze()
            ids_missing = torch.argwhere(x[:, 3] == baseline_value)
            x[ids_missing, 3] = missing_activity[ids_missing]

        with torch.no_grad():
            dataset = pyg_Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            if neural_ODE_training:
                # Use Neural ODE integration with time_step=1
                u0 = x[:, 3].flatten()
                u_final, _ = integrate_neural_ode_Signal(
                    model=model,
                    u0=u0,
                    data_template=dataset,
                    data_id=data_id,
                    time_steps=1,
                    delta_t=delta_t,
                    neurons_per_sample=n_neurons,
                    batch_size=1,
                    x_list=None,
                    run=0,
                    device=device,
                    k_batch=torch.tensor([it], device=device),
                    ode_method=ode_method,
                    rtol=ode_rtol,
                    atol=ode_atol,
                    adjoint=False,
                    noise_level=0.0
                )
                y = (u_final.view(-1, 1) - x[:, 3:4]) / delta_t
            else:
                pred = model(dataset, data_id=data_id, k=it)
                y = pred
            dataset = pyg_Data(x=x_generated, pos=x[:, 1:3], edge_index=edge_index_generated)
            if "PDE_N3" in model_config.signal_model_name:
                pred_generator = model_generator(dataset, data_id=data_id, alpha=it/n_frames)
            else:
                pred_generator = model_generator(dataset, data_id=data_id)

        # signal update
        x[:n_neurons, 3:4] = x[:n_neurons, 3:4] + y[:n_neurons] * delta_t
        x_generated[:n_neurons, 3:4] = x_generated[:n_neurons, 3:4] + pred_generator[:n_neurons] * delta_t

        if 'test_inactivity' in test_mode:
            x[index_inactivity, 3:4] = 0
            x_generated[index_inactivity, 3:4] = 0

        # if 'CElegans' in dataset_name:
        #     x[:n_neurons, 6:7] = torch.clamp(x[:n_neurons, 6:7], min=0, max=10)

        # vizualization
        if 'plot_data' in test_mode:
            x = x_list[0][it].clone().detach()

        if (it % step == 0) & (it >= 0) & visualize:

            num = f"{it:06}"

            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            ax.tick_params(axis='both', which='major', pad=15)

            if ('visual' in field_type):
                if 'plot_data' in test_mode:
                    plt.close()

                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_input_neurons_per_axis * n_input_neurons_per_axis))
                    if ('modulation' in field_type):
                        A1[:, 0:1] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)
                    if ('visual' in field_type):
                        A1[:n_input_neurons, 0:1] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)
                        A1[n_input_neurons:n_neurons, 0:1] = 1

                fig = plt.figure(figsize=(8, 12))
                plt.subplot(211)
                plt.title(r'$b_i$', fontsize=48)
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=8, c=to_numpy(A1[:, 0]), cmap='viridis', vmin=0,
                            vmax=2)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(212)
                plt.title(r'$x_i$', fontsize=48)
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=8, c=to_numpy(x[:, 3:4]), cmap='viridis',
                            vmin=-10,
                            vmax=10)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{num}.tif", dpi=80)
                plt.close()

            else:

                plt.close()
                mpl.rcParams['savefig.pad_inches'] = 0

                black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])
                LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])

                plt.figure(figsize=(10, 10))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=700, c=to_numpy(x[:, 3]), alpha=1, edgecolors='none', vmin =2 , vmax=8, cmap=black_to_green)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-0.6, 0.6])
                plt.ylim([-0.6, 0.6])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=80)
                plt.close()


                if ('short_term_plasticity' in field_type) | ('modulation' in field_type):

                    modulation_gt_list_ = torch.cat(modulation_gt_list, 0)
                    modulation_pred_list_ = torch.cat(modulation_pred_list, 0)
                    modulation_gt_list_ = torch.reshape(modulation_gt_list_,
                                                        (modulation_gt_list_.shape[0] // n_neurons, n_neurons))
                    modulation_pred_list_ = torch.reshape(modulation_pred_list_,
                                                          (modulation_pred_list_.shape[0] // n_neurons,
                                                           n_neurons))

                    plt.figure(figsize=(20, 10))
                    if 'latex' in style:
                        plt.rcParams['text.usetex'] = False
                        plt.rcParams['font.family'] = 'sans-serif'

                    ax = plt.subplot(122)
                    plt.scatter(to_numpy(modulation_gt_list_[-1, :]), to_numpy(modulation_pred_list_[-1, :]), s=10,
                                c=mc)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    x_data = to_numpy(modulation_gt_list_[-1, :])
                    y_data = to_numpy(modulation_pred_list_[-1, :])
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    1 - (ss_res / ss_tot)
                    plt.xlabel(r'true modulation', fontsize=48)
                    plt.ylabel(r'learned modulation', fontsize=48)
                    # plt.text(0.05, 0.9 * lin_fit[0], f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    # plt.text(0.05, 0.8 * lin_fit[0], f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                    ax = plt.subplot(121)
                    plt.plot(modulation_gt_list_[:, n[0]].detach().cpu().numpy(), c='k', linewidth=8, label='true',
                             alpha=0.25)
                    plt.plot(modulation_pred_list_[:, n[0]].detach().cpu().numpy() / lin_fit[0], linewidth=4, c='k',
                             label='learned')
                    plt.legend(fontsize=24)
                    plt.plot(modulation_gt_list_[:, n[1:10]].detach().cpu().numpy(), c='k', linewidth=8, alpha=0.25)
                    plt.plot(modulation_pred_list_[:, n[1:10]].detach().cpu().numpy() / lin_fit[0], linewidth=4)
                    plt.xlim([0, 1400])
                    plt.xlabel(r'time-points', fontsize=48)
                    plt.ylabel(r'modulation', fontsize=48)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.ylim([0, 2])
                    # plt.text(40, 26, f'time: {it}', fontsize=34)

        if (it % it_step == 0) & (it > 0) & (it <=n_test_frames):

            num = f"{id_fig:06}"
            id_fig += 1

            if n_neurons <= 101:
                n = np.arange(0, n_neurons, 4)
            else:
                n = [20, 30, 100, 150, 260, 270, 520, 620, 720, 780]

            neuron_gt_list_ = torch.cat(neuron_gt_list, 0)
            neuron_pred_list_ = torch.cat(neuron_pred_list, 0)
            neuron_generated_list_ = torch.cat(neuron_generated_list, 0)
            neuron_gt_list_ = torch.reshape(neuron_gt_list_, (neuron_gt_list_.shape[0] // n_neurons, n_neurons))
            neuron_pred_list_ = torch.reshape(neuron_pred_list_, (neuron_pred_list_.shape[0] // n_neurons, n_neurons))
            neuron_generated_list_ = torch.reshape(neuron_generated_list_, (neuron_generated_list_.shape[0] // n_neurons, n_neurons))

            mpl.rcParams.update({
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.size": 12,
                "axes.labelsize": 14,
                "legend.fontsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            })

            plt.figure(figsize=(20, 10))

            ax = plt.subplot(121)
            # Plot ground truth with distinct gray color, visible in legend
            for i in range(len(n)):
                if ablation_ratio > 0:
                    label = f'true ablation {ablation_ratio}' if i == 0 else None
                elif inactivity_ratio > 0:
                    label = f'true inactivity {inactivity_ratio}' if i == 0 else None
                elif permutation_ratio > 0:
                    label = f'true permutation {permutation_ratio}' if i == 0 else None
                else:
                    label = 'true' if i == 0 else None
                plt.plot(neuron_generated_list_[:, n[i]].detach().cpu().numpy() + i * 25,
                        c='gray', linewidth=8, alpha=0.5, label=label)

            # Plot predictions with colored lines
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for i in range(len(n)):
                label = 'learned' if i == 0 else None

                if 'test_generated' in test_mode:
                    plt.plot(neuron_generated_list_[:, n[i]].detach().cpu().numpy() + i * 25,
                            linewidth=3, c=colors[i%10], label=label)
                else:
                    plt.plot(neuron_pred_list_[:, n[i]].detach().cpu().numpy() + i * 25,
                            linewidth=3, c=colors[i%10], label=label)

            plt.xlim([0, len(neuron_gt_list_)])

            # Auto ylim from ground truth range (ignore predictions if exploded)
            y_gt = np.concatenate([neuron_gt_list_[:, n[i]].detach().cpu().numpy() + i*25 for i in range(len(n))])
            y_pred = np.concatenate([neuron_pred_list_[:, n[i]].detach().cpu().numpy() + i*25 for i in range(len(n))])

            if np.abs(y_pred).max() > 10 * np.abs(y_gt).max():  # Explosion
                ylim = [y_gt.min() - 10, y_gt.max() + 10]
            else:
                y_all = np.concatenate([y_gt, y_pred])
                margin = (y_all.max() - y_all.min()) * 0.05
                ylim = [y_all.min() - margin, y_all.max() + margin]

            plt.xlim([0, n_test_frames])
            plt.ylim(ylim)
            plt.xlabel('frame', fontsize=48)
            if 'PDE_N11' in config.graph_model.signal_model_name:
                plt.ylabel('$h_i$', fontsize=48)
            else:
                plt.ylabel('$x_i$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks([])

            ax = plt.subplot(222)
            x_data = to_numpy(neuron_generated_list_[-1, :])
            y_data = to_numpy(neuron_pred_list_[-1, :])

            # print(f"x: [{x_data.min():.2e}, {x_data.max():.2e}]")
            # print(f"y: [{y_data.min():.2e}, {y_data.max():.2e}], std={y_data.std():.2e}")

            # Detect severe collapse (constant predictions) or explosion
            severe_collapse = y_data.std() < 0.1 * x_data.std()
            explosion = np.abs(y_data).max() > 1e10

            if not (severe_collapse or explosion):
                # Normal/mild collapse: fit and show all data
                mask = (np.abs(x_data - np.median(x_data)) < 3*np.std(x_data)) & \
                    (np.abs(y_data - np.median(y_data)) < 3*np.std(y_data))
                if mask.sum() > 10:
                    lin_fit, _ = curve_fit(linear_model, x_data[mask], y_data[mask])
                    slope, intercept = lin_fit
                    r2 = 1 - np.sum((y_data - linear_model(x_data, *lin_fit))**2) / np.sum((y_data - np.mean(y_data))**2)
                else:
                    slope, intercept, r2 = 0, 0, 0

                # Auto limits from combined data
                all_data = np.concatenate([x_data, y_data])
                margin = (all_data.max() - all_data.min()) * 0.1
                lim = [all_data.min() - margin, all_data.max() + margin]

                plt.scatter(x_data, y_data, s=20, c=mc, alpha=0.8, edgecolors='none', linewidths=0.5)
                if mask.sum() > 10:
                    x_line = np.array(lim)
                    plt.plot(x_line, linear_model(x_line, slope, intercept), 'r--', linewidth=2)
                plt.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)
                plt.text(0.05, 0.95, f'$R^2$: {r2:.3f}\nslope: {slope:.3f}',
                        transform=plt.gca().transAxes, fontsize=24, va='top')
            else:
                # Severe collapse/explosion
                lim = [x_data.min()*1.1, 0]
                plt.scatter(x_data, np.clip(y_data, lim[0], lim[1]), s=100, c=mc, alpha=0.8, edgecolors='k', linewidths=0.5)
                plt.text(0.5, 0.5, 'collapsed' if severe_collapse else 'explosion',
                        ha='center', fontsize=48, color='red', alpha=0.3, transform=plt.gca().transAxes)
                r2 = 0

            # plt.xlim([-20,20])
            # plt.ylim([-20,20])
            if 'PDE_N11' in config.graph_model.signal_model_name:
                plt.xlabel('true $h_i$', fontsize=48)
                plt.ylabel('learned $h_i$', fontsize=48)
            else:
                plt.xlabel('true $x_i$', fontsize=48)
                plt.ylabel('learned $x_i$', fontsize=48)
            plt.xticks([])
            plt.yticks([])


            R2_list.append(r2)
            it_list.append(it)


            ax = plt.subplot(224)
            plt.scatter(it_list, R2_list, s=20, c=mc)
            plt.xlim([0, n_test_frames])
            plt.ylim([0, 1])
            plt.axhline(1, color='green', linestyle='--', linewidth=2)
            plt.xlabel('frame', fontsize=48)
            plt.ylabel('$R^2$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()

            if ablation_ratio>0:
                filename = f'comparison_vi_{it}_ablation_{ablation_ratio}.png'
            elif inactivity_ratio>0:
                filename = f'comparison_vi_{it}_inactivity_{inactivity_ratio}.png'
            elif permutation_ratio>0:
                filename = f'comparison_vi_{it}_permutation_{permutation_ratio}.png'
            else:
                filename = f'comparison_vi_{it}.png'

            plt.savefig(f'./{log_dir}/results/Fig/Fig_{run}_{num}.png', dpi=80)
            plt.close()
            # print(f'saved figure ./log/{log_dir}/results/{filename}')





    dataset_name_ = dataset_name.split('/')[-1]
    generate_compressed_video_mp4(output_dir=f"./{log_dir}/results/", run=run, output_name=dataset_name_, framerate=20)

    # Copy the last PNG file before erasing Fig folder
    files = glob.glob(f'./{log_dir}/results/Fig/*.png')
    if files:
        files.sort()  # Sort to get the last file
        last_file = files[-1]
        dataset_name_ = dataset_name.split('/')[-1]
        dst_file = f"./{log_dir}/results/{dataset_name_}.png"
        import shutil
        shutil.copy(last_file, dst_file)
        print(f"saved last frame: {dst_file}")

    files = glob.glob(f'./{log_dir}/results/Fig/*')
    for f in files:
        os.remove(f)


    x_inference_list = torch.cat(x_inference_list, 1)
    x_generated_list = torch.cat(x_generated_list, 1)



    print('plot prediction ...')
    # Single panel plot: green=GT, white=prediction, R² on the right
    # Limit to max 50 traces evenly spaced across neurons
    n_traces = min(50, n_neurons)
    trace_ids = np.linspace(0, n_neurons - 1, n_traces, dtype=int)

    # Stack ground truth and prediction lists
    neuron_gt_stacked = torch.cat(neuron_gt_list, dim=0).reshape(-1, n_neurons).T  # [n_neurons, n_frames]
    neuron_pred_stacked = torch.cat(neuron_pred_list, dim=0).reshape(-1, n_neurons).T  # [n_neurons, n_frames]

    activity_gt = to_numpy(neuron_gt_stacked)  # ground truth
    activity_pred = to_numpy(neuron_pred_stacked)  # prediction
    n_frames_plot = activity_gt.shape[1]

    # Compute per-neuron R² for selected traces
    r2_per_neuron = []
    for idx in trace_ids:
        gt_trace = activity_gt[idx]
        pred_trace = activity_pred[idx]
        ss_res = np.sum((gt_trace - pred_trace) ** 2)
        ss_tot = np.sum((gt_trace - np.mean(gt_trace)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_per_neuron.append(r2)

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Compute offset based on data range
    offset = np.abs(activity_gt[trace_ids]).max() * 1.5
    if offset == 0:
        offset = 1.0

    for j, n_idx in enumerate(trace_ids):
        y0 = j * offset
        baseline = np.mean(activity_gt[n_idx])
        # Ground truth in green (thicker line)
        ax.plot(activity_gt[n_idx] - baseline + y0, color='green', lw=4.0, alpha=0.9)
        # Prediction in white (or mc for style consistency)
        ax.plot(activity_pred[n_idx] - baseline + y0, color=mc, lw=0.8, alpha=0.9)

        # Neuron index on the left
        ax.text(-n_frames_plot * 0.02, y0, str(n_idx), fontsize=10, va='center', ha='right')

        # R² on the right with color coding
        r2_val = r2_per_neuron[j]
        r2_color = 'red' if r2_val < 0.5 else ('orange' if r2_val < 0.8 else mc)
        ax.text(n_frames_plot * 1.02, y0, f'R²:{r2_val:.2f}', fontsize=9, va='center', ha='left', color=r2_color)

    ax.set_xlim([-n_frames_plot * 0.05, n_frames_plot * 1.1])
    ax.set_ylim([-offset, n_traces * offset])
    ax.set_xlabel('frame', fontsize=24)
    ax.set_ylabel('neuron', fontsize=24)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, n_frames_plot)
    ax.set_yticks([])

    # Add overall R² in title
    mean_r2 = np.mean(r2_per_neuron)
    ax.set_title(f'Activity traces (n={n_traces} of {n_neurons} neurons) | mean R²={mean_r2:.3f}', fontsize=20)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/{dataset_name_}_prediction.pdf", dpi=300)
    plt.close()



    if 'PDE_N' in model_config.signal_model_name:
        torch.save(neuron_gt_list, f"./{log_dir}/neuron_gt_list.pt")
        torch.save(neuron_pred_list, f"./{log_dir}/neuron_pred_list.pt")

    # Comprehensive R² analysis
    if len(R2_list) > 0:
        R2_array = np.array(R2_list)
        it_array = np.array(it_list)

        # Basic statistics
        r2_mean = np.mean(R2_array)
        r2_std = np.std(R2_array)
        r2_min = np.min(R2_array)
        r2_max = np.max(R2_array)
        r2_median = np.median(R2_array)

        # High R² analysis (R² > 0.9)
        high_r2_mask = R2_array > 0.9
        n_frames_high_r2 = np.sum(high_r2_mask)
        pct_frames_high_r2 = 100.0 * n_frames_high_r2 / len(R2_array)

        # Find longest consecutive run of R² > 0.9
        high_r2_runs = []
        current_run_start = None
        current_run_length = 0

        for i, (r2_val, frame_idx) in enumerate(zip(R2_array, it_array)):
            if r2_val > 0.9:
                if current_run_start is None:
                    current_run_start = frame_idx
                    current_run_length = 1
                else:
                    current_run_length += 1
            else:
                if current_run_start is not None:
                    high_r2_runs.append((current_run_start, current_run_length))
                    current_run_start = None
                    current_run_length = 0

        # Don't forget the last run if it extends to the end
        if current_run_start is not None:
            high_r2_runs.append((current_run_start, current_run_length))

        if high_r2_runs:
            longest_run_start, longest_run_length = max(high_r2_runs, key=lambda x: x[1])
        else:
            longest_run_start, longest_run_length = 0, 0
        print(f"mean R2: \033[92m{r2_mean:.4f}\033[0m +/- {r2_std:.4f}")
        print(f"range: [{r2_min:.4f}, {r2_max:.4f}]")
        if log_file:
            log_file.write(f"test_R2: {r2_mean:.4f}\n")

        # Compute Pearson correlation per neuron across time
        from scipy.stats import pearsonr
        neuron_gt_array = torch.stack(neuron_gt_list, dim=0).squeeze(-1)  # [n_frames, n_neurons]
        neuron_pred_array = torch.stack(neuron_pred_list, dim=0).squeeze(-1)  # [n_frames, n_neurons]
        neuron_gt_np = to_numpy(neuron_gt_array)
        neuron_pred_np = to_numpy(neuron_pred_array)

        pearson_list = []
        for i in range(neuron_gt_np.shape[1]):
            gt_trace = neuron_gt_np[:, i]
            pred_trace = neuron_pred_np[:, i]
            valid = ~(np.isnan(gt_trace) | np.isnan(pred_trace))
            if valid.sum() > 1 and np.std(gt_trace[valid]) > 1e-8 and np.std(pred_trace[valid]) > 1e-8:
                pearson_list.append(pearsonr(gt_trace[valid], pred_trace[valid])[0])
            else:
                pearson_list.append(np.nan)
        pearson_array = np.array(pearson_list)
        print(f"Pearson r: \033[92m{np.nanmean(pearson_array):.4f}\033[0m +/- {np.nanstd(pearson_array):.4f} [{np.nanmin(pearson_array):.4f}, {np.nanmax(pearson_array):.4f}]")
        if log_file:
            log_file.write(f"test_pearson: {np.nanmean(pearson_array):.4f}\n")


