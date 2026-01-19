import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch_geometric.data as data
from matplotlib import rc
from neural_gnn.generators.utils import (
    choose_model,
    init_neurons,
    init_mesh,
    generate_compressed_video_mp4,
    init_connectivity,
    get_equidistant_points,
    plot_synaptic_frame_visual,
    plot_synaptic_frame_modulation,
    plot_synaptic_frame_plasticity,
    plot_synaptic_frame_default,
    plot_synaptic_activity_traces,
    plot_synaptic_mlp_functions,
    plot_eigenvalue_spectrum,
    plot_connectivity_matrix,
    plot_external_input_field,
    plot_activity_sample,
)
from neural_gnn.utils import to_numpy, CustomColorMap, check_and_clear_memory, get_datavis_root_dir
from tifffile import imread
from tqdm import tqdm, trange
import os

# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr
import torch_geometric as pyg

# import taichi as ti

def data_generate(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    best_model=None,
    device=None,
    bSave=True,
    log_file=None,
    regenerate_plots_only=False,
):

    has_signal = "PDE_N" in config.graph_model.signal_model_name
    has_fly = "fly" in config.dataset

    dataset_name = config.dataset

    print(f"\033[94mdataset_name: {dataset_name}\033[0m")

    if (os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.npy")) | (
        os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.pt")
    ):
        print("watch out: data already generated")
        # return

    if config.data_folder_name != "none":
        generate_from_data(config=config, device=device, visualize=visualize, style=style, step=step)
    elif has_fly:
        data_generate_fly_voltage(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            bSave=bSave,
        )
    elif has_signal:
        data_generate_synaptic(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            bSave=bSave,
            log_file=log_file,
            regenerate_plots_only=regenerate_plots_only,
        )

    plt.style.use("default")


def mseq_bits(p=8, taps=(8,6,5,4), seed=1, length=None):
    """
    Simple LFSR-based m-sequence generator that returns a numpy array of ±1.
    Default p=8 -> period 2**8 - 1 = 255.
    """
    if length is None:
        length = 2**p - 1
    state = (1 << p) - 1 if seed is None else (seed % (1 << p)) or 1
    bits = []
    for _ in range(length):
        bits.append(1 if (state & 1) else -1)
        fb = 0
        for t in taps:
            fb ^= (state >> (t-1)) & 1
        state = (state >> 1) | (fb << (p-1))
    return np.array(bits, dtype=np.int8)

def assign_columns_from_uv(u_coords, v_coords, n_cols, random_state=0):
    """Cluster photoreceptors into n_cols tiles via k-means on (u,v)."""
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn is required for 'tile_mseq' visual_input_type") from e
    X = np.stack([u_coords, v_coords], axis=1)
    km = KMeans(n_clusters=n_cols, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels

def compute_column_labels(u_coords, v_coords, n_columns, seed=0):
    labels = assign_columns_from_uv(u_coords, v_coords, n_columns, random_state=seed)
    centers = np.zeros((n_columns, 2), dtype=np.float32)
    counts = np.zeros(n_columns, dtype=np.int32)
    for i, lab in enumerate(labels):
        centers[lab, 0] += u_coords[i]
        centers[lab, 1] += v_coords[i]
        counts[lab] += 1
    counts[counts == 0] = 1
    centers /= counts[:, None]
    return labels, centers

def build_neighbor_graph(centers, k=6):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(centers)), algorithm="auto").fit(centers)
    dists, idxs = nbrs.kneighbors(centers)
    adj = [set() for _ in range(len(centers))]
    for i in range(len(centers)):
        for j in idxs[i,1:]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return adj

def greedy_blue_mask(adj, n_cols, target_density=0.5, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    order = rng.permutation(n_cols)
    chosen = np.zeros(n_cols, dtype=bool)
    blocked = np.zeros(n_cols, dtype=bool)
    target = int(target_density * n_cols)
    for i in order:
        if not blocked[i]:
            chosen[i] = True
            for j in adj[i]:
                blocked[j] = True
        if chosen.sum() >= target:
            break
    if chosen.sum() < target:
        remain = np.where(~chosen)[0]
        rng.shuffle(remain)
        for i in remain:
            conflict = any(chosen[j] for j in adj[i])
            if not conflict:
                chosen[i] = True
            if chosen.sum() >= target:
                break
    return chosen

def apply_pairwise_knobs_torch(code_pm1: torch.Tensor,
                                corr_strength: float,
                                flip_prob: float,
                                seed: int) -> torch.Tensor:
    """
    code_pm1: shape [n_tiles], values in approximately {-1, +1}
    corr_strength: 0..1; blends in a global shared ±1 component (↑ pairwise corr)
    flip_prob: 0..1; per-tile random sign flips (decorrelates)
    seed: for reproducibility (we also add tile_idx later to vary per frame)
    """
    out = code_pm1.clone()

    # Torch RNG on correct device
    gen = torch.Generator(device=out.device)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)

    # (1) Optional global shared component
    if corr_strength > 0.0:
        g = torch.randint(0, 2, (1,), generator=gen, device=out.device, dtype=torch.int64)
        g = g.float().mul_(2.0).add_(-1.0)  # {0,1} -> {-1,+1}
        out.mul_(1.0 - float(corr_strength)).add_(float(corr_strength) * g)

    # (2) Optional per-tile random flips
    if flip_prob > 0.0:
        flips = torch.rand(out.shape, generator=gen, device=out.device) < float(flip_prob)
        out[flips] = -out[flips]

    return out





def data_generate_synaptic(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    device=None,
    bSave=True,
    log_file=None,
    regenerate_plots_only=False,
):
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0

    dataset_name = config.dataset
    noise_model_level = training_config.noise_model_level
    measurement_noise_level = training_config.measurement_noise_level
    
    CustomColorMap(config=config)
    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'


    external_input_type = getattr(simulation_config, 'external_input_type', '')
    n_input_neurons = simulation_config.n_input_neurons
    n_input_neurons_per_axis = int(np.sqrt(n_input_neurons))
    has_visual_input = "visual" in external_input_type
    has_modulation = "modulation" in external_input_type

    folder = f"./graphs_data/{dataset_name}/"

    if config.data_folder_name != "none":
        generate_from_data(
            config=config, device=device, visualize=visualize, step=step
        )
        return

    # If regenerate_plots_only, load existing data and regenerate plots
    if regenerate_plots_only:
        x_list_file = f"{folder}/x_list_0.npy"
        connectivity_file = f"{folder}/connectivity.pt"
        if os.path.exists(x_list_file) and os.path.exists(connectivity_file):
            x_list = np.load(x_list_file)
            connectivity = torch.load(connectivity_file, map_location='cpu', weights_only=True)
            plot_synaptic_activity_traces(x_list, n_neurons, n_frames, dataset_name)
            plot_connectivity_matrix(connectivity.t(), f"./graphs_data/{dataset_name}/connectivity_matrix.png",
                                     vmin_vmax_method='percentile', show_title=False)
            # Plot external input field at key frames (Fig 3a)
            frame_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4]
            plot_external_input_field(x_list, n_neurons, dataset_name,
                                      frame_indices=frame_indices,
                                      n_input_neurons=n_input_neurons)
            # Plot activity sample (Fig 3c)
            plot_activity_sample(x_list, n_neurons, n_frames, dataset_name, n_traces=10)
            return
        else:
            print(f"data files not found in {folder}, running full generation...")
    else:
        print("generating data ...")

    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (not ("X1.pt" in f))
                & (not ("Signal" in f))
                & (not ("Viz" in f))
                & (not ("Exc" in f))
                & (f[-3:] != "Fig")
                & (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)

    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_neurons))
        cut = int(n_neurons * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        particle_dropout_mask = np.arange(n_neurons)
    if has_modulation or has_visual_input:
        im = imread(f"graphs_data/{simulation_config.node_value_map}")
    has_permutation = getattr(simulation_config, 'permutation', False)
    if has_permutation:
        permutation_indices = torch.randperm(n_neurons)
        inverse_permutation_indices = torch.argsort(permutation_indices)
        torch.save(
            permutation_indices, f"./graphs_data/{dataset_name}/permutation_indices.pt"
        )
        torch.save(
            inverse_permutation_indices,
            f"./graphs_data/{dataset_name}/inverse_permutation_indices.pt",
        )


    # external input parameters
    external_input_type = getattr(simulation_config, 'external_input_type', 'none')
    signal_input_type = getattr(simulation_config, 'signal_input_type', 'oscillatory')
    has_signal_input = (external_input_type == 'signal')
    has_oscillations = has_signal_input and (signal_input_type == 'oscillatory')
    has_triggered = has_signal_input and (signal_input_type == 'triggered')
    oscillation_amplitude = simulation_config.oscillation_max_amplitude
    oscillation_frequency = torch.tensor(simulation_config.oscillation_frequency, dtype=torch.float32, device=device)
    max_frame = n_frames + 1

    # initialize triggered oscillation parameters (if needed)
    if has_triggered:
        triggered_n_impulses = simulation_config.triggered_n_impulses
        triggered_n_input = simulation_config.triggered_n_input_neurons
        triggered_strength = simulation_config.triggered_impulse_strength
        triggered_duration = simulation_config.triggered_duration_frames
        amplitude_range = simulation_config.triggered_amplitude_range
        frequency_range = simulation_config.triggered_frequency_range

        # generate per-neuron random amplitude
        e_global = oscillation_amplitude * (torch.rand((n_neurons, 1), device=device) * 2 - 1)

        # generate multiple impulse events spread throughout simulation
        buffer = triggered_duration
        available_frames = max_frame - 2 * buffer
        spacing = available_frames // max(1, triggered_n_impulses)

        trigger_frames = []
        trigger_amplitudes = []
        trigger_frequencies = []
        trigger_neurons = []
        trigger_e = []

        for i in range(triggered_n_impulses):
            base_frame = buffer + i * spacing
            jitter = torch.randint(-spacing//4, spacing//4 + 1, (1,), device=device).item() if spacing > 4 else 0
            trigger_frame = max(buffer, min(max_frame - buffer, base_frame + jitter))
            trigger_frames.append(trigger_frame)

            amp_mult = amplitude_range[0] + torch.rand(1, device=device).item() * (amplitude_range[1] - amplitude_range[0])
            trigger_amplitudes.append(amp_mult)

            freq_mult = frequency_range[0] + torch.rand(1, device=device).item() * (frequency_range[1] - frequency_range[0])
            trigger_frequencies.append(freq_mult)

            input_neurons = torch.randperm(n_neurons, device=device)[:triggered_n_input]
            trigger_neurons.append(input_neurons)

            e = oscillation_amplitude * amp_mult * (torch.rand((n_neurons, 1), device=device) * 2 - 1)
            trigger_e.append(e)
    elif has_oscillations:
        # Per-neuron random amplitude for oscillatory input
        e_global = oscillation_amplitude * (torch.rand((n_neurons, 1), device=device) * 2 - 1)

    # open logfile for analysis results (use provided or create local)
    local_log_file = log_file is None
    if local_log_file:
        log_file = open(f"{folder}/analysis.log", 'w')

    for run in range(config.training.n_runs):

        id_fig = 0

        X = torch.zeros((n_neurons, n_frames + 1), device=device)

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_neurons(
            config=config, scenario=scenario, ratio=ratio, device=device
        )

        if simulation_config.shuffle_neuron_types:
            if run == 0:
                index = torch.randperm(n_neurons)
                T1 = T1[index]
                first_T1 = T1.clone().detach()
            else:
                T1 = first_T1.clone().detach()

        if run == 0:
            edge_index, connectivity, mask = init_connectivity(
                simulation_config.connectivity_file,
                simulation_config.connectivity_type,
                simulation_config.connectivity_filling_factor,
                T1,
                n_neurons,
                n_neuron_types,
                dataset_name,
                device,
                connectivity_rank=simulation_config.connectivity_rank,
                Dale_law=simulation_config.Dale_law,
                Dale_law_factor=simulation_config.Dale_law_factor,
            )
            torch.save(edge_index, f"./graphs_data/{dataset_name}/edge_index.pt")
            torch.save(mask, f"./graphs_data/{dataset_name}/mask.pt")
            torch.save(connectivity, f"./graphs_data/{dataset_name}/connectivity.pt")

            # Plot eigenvalue spectrum and connectivity matrix
            plot_eigenvalue_spectrum(connectivity, dataset_name, mc=mc, log_file=log_file)
            plot_connectivity_matrix(connectivity, f"./graphs_data/{dataset_name}/connectivity_matrix.png",
                                     vmin_vmax_method='percentile', show_title=False)

        if has_modulation:
            if run == 0:
                X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = (
                    init_mesh(config, device=device)
                )
                X1 = X1_mesh
        elif has_visual_input:
            if run == 0:
                X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = (
                    init_mesh(config, device=device)
                )
                pos_x, pos_y = get_equidistant_points(n_points=n_input_neurons)
                X1 = (
                    torch.tensor(
                        np.stack((pos_x, pos_y), axis=1), dtype=torch.float32, device=device
                    )
                    / 2
                )
                X1[:, 1] = X1[:, 1] + 1.5
                X1[:, 0] = X1[:, 0] + 0.5
                X1 = torch.cat((X1_mesh, X1[0 : n_neurons - n_input_neurons]), 0)

        model, bc_pos, bc_dpos = choose_model(config=config, W=connectivity, device=device)

        # NEW x tensor layout (like flyvis):
        # x[:, 0]   = index (neuron ID)
        # x[:, 1:3] = positions (x, y)
        # x[:, 3]   = signal u (state)
        # x[:, 4]   = external_input
        # x[:, 5]   = plasticity p (PDE_N6/N7)
        # x[:, 6]   = neuron_type
        # x[:, 7]   = calcium
        x = torch.zeros((n_neurons, 8), dtype=torch.float32, device=device)
        x[:, 0] = torch.arange(n_neurons, dtype=torch.float32, device=device)  # index
        x[:, 1:3] = X1.clone().detach()  # positions
        x[:, 3] = H1[:, 0].clone().detach()  # signal state u
        x[:, 4] = 0  # external input (set per frame)
        x[:, 5] = 1  # plasticity p (init to 1 for PDE_N6/N7)
        x[:, 6] = T1.squeeze().clone().detach()  # neuron_type
        x[:, 7] = 0  # calcium

        check_and_clear_memory(
            device=device,
            iteration_number=0,
            every_n_iterations=1,
            memory_percentage_threshold=0.6,
        )

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1, ncols=150):
            with torch.no_grad():

                # compute external input for this frame
                external_input = torch.zeros((n_neurons, 1), device=device)

                if (has_modulation) & (it >= 0):
                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_input_neurons_per_axis * n_input_neurons_per_axis))
                    if has_permutation:
                        im_ = im_[permutation_indices]
                    external_input[:, 0] = torch.tensor(im_, dtype=torch.float32, device=device)
                elif (has_visual_input) & (it >= 0):
                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_input_neurons_per_axis * n_input_neurons_per_axis))
                    external_input[:n_input_neurons, 0] = torch.tensor(im_, dtype=torch.float32, device=device)
                    external_input[n_input_neurons:n_neurons, 0] = 1
                    # Save reconstructed image from x[:,4] for the first frame
                    if it == 0 and run == 0:
                        img_reconstructed = to_numpy(external_input[:n_input_neurons, 0].reshape(n_input_neurons_per_axis, n_input_neurons_per_axis))
                        val_min = np.min(img_reconstructed)
                        val_max = np.max(img_reconstructed)
                        val_std = np.std(img_reconstructed)
                        img_reconstructed = np.rot90(img_reconstructed, k=1)
                        plt.figure(figsize=(8, 8))
                        plt.imshow(img_reconstructed, cmap='gray')
                        plt.text(0.02, 0.98, f'min={val_min:.2f} max={val_max:.2f} std={val_std:.2f}',
                                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                                 color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
                        plt.axis('off')
                        plt.savefig(f"{folder}/external_input_frame0.png", dpi=150)
                        plt.close()
                elif has_oscillations:
                    # oscillatory external input (frequency in cycles per time unit)
                    t = it * delta_t
                    external_input = e_global * torch.cos((2*np.pi)*oscillation_frequency*t)
                elif has_triggered:
                    # triggered oscillation input
                    for i in range(triggered_n_impulses):
                        trig_frame = trigger_frames[i]
                        # add impulse at trigger frame
                        if it == trig_frame:
                            impulse = torch.zeros((n_neurons, 1), device=device)
                            impulse[trigger_neurons[i]] = triggered_strength * trigger_amplitudes[i]
                            external_input = external_input + impulse
                        # add oscillatory response after trigger
                        if trig_frame <= it < trig_frame + triggered_duration:
                            t_since_trigger = it - trig_frame
                            freq_mult = trigger_frequencies[i]
                            osc = trigger_e[i] * torch.sin((2*np.pi)*oscillation_frequency*freq_mult*t_since_trigger / triggered_duration)
                            external_input = external_input + osc

                # update x tensor for this frame
                x[:, 4] = external_input.squeeze()  # external input

                X[:, it] = x[:, 3].clone().detach()  # store signal state
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                # model prediction (PDE_N4 uses external_input_mode from config)
                if "PDE_N3" in model_config.signal_model_name:
                    y = model(dataset, has_field=False, alpha=it / n_frames, frame=it)
                elif "PDE_N6" in model_config.signal_model_name:
                    (y, p_out) = model(dataset, has_field=False, frame=it)
                elif "PDE_N7" in model_config.signal_model_name:
                    (y, p_out) = model(dataset, has_field=False, frame=it)
                else:
                    y = model(dataset, frame=it)

            # append list
            if (it >= 0) & bSave:
                x_list.append(to_numpy(x))
                y_list.append(to_numpy(y))

            # field update - update x tensor directly
            if (config.graph_model.signal_model_name == "PDE_N6") | (config.graph_model.signal_model_name == "PDE_N7"):
                # Signal update
                du = y.squeeze()
                x[:, 3] = x[:, 3] + du * delta_t
                if noise_model_level > 0:
                    x[:, 3] = x[:, 3] + torch.randn(n_neurons, device=device) * noise_model_level
                # Plasticity update
                dp = p_out.squeeze()
                x[:, 5] = torch.relu(x[:, 5] + dp * delta_t)
            else:
                du = y.squeeze()
                x[:, 3] = x[:, 3] + du * delta_t
                if noise_model_level > 0:
                    x[:, 3] = x[:, 3] + torch.randn(n_neurons, device=device) * noise_model_level

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})
                if "black" in style:
                    plt.style.use("dark_background")
                matplotlib.rcParams["savefig.pad_inches"] = 0
                num = f"{id_fig:06}"
                id_fig += 1

                if has_visual_input:
                    plot_synaptic_frame_visual(x[:, 1:3], x[:, 4:5], x[:, 3:4], dataset_name, run, num)
                elif has_modulation:
                    plot_synaptic_frame_modulation(x[:, 1:3], x[:, 4:5], x[:, 3:4], dataset_name, run, num)
                else:
                    if ("PDE_N6" in model_config.signal_model_name) | (
                        "PDE_N7" in model_config.signal_model_name
                    ):
                        plot_synaptic_frame_plasticity(x[:, 1:3], x, dataset_name, run, num)
                    else:
                        plot_synaptic_frame_default(x[:, 1:3], x, dataset_name, run, num)

        print(f"generated {len(x_list)} frames total")

        if visualize & (run == run_vizualized):
            print('generating lossless video ...')

            output_name = dataset_name.split('signal_')[1] if 'signal_' in dataset_name else 'no_id'
            src = f"./graphs_data/{dataset_name}/Fig/Fig_0_000000.png"
            dst = f"./graphs_data/{dataset_name}/input_{output_name}.png"
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())

            generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                          output_name=output_name, framerate=20)

            files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
            for f in files:
                os.remove(f)

            print('Ising analysis ...')
            x_list = np.array(x_list)
            y_list = np.array(y_list)

        if bSave:
            x_list = np.array(x_list)
            y_list = np.array(y_list)

        if measurement_noise_level > 0:
            np.save(f"graphs_data/{dataset_name}/raw_x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/raw_y_list_{run}.npy", y_list)
            torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")
            for k in range(x_list.shape[0]):
                x_list[k, :, 3] = x_list[k, :, 3] + np.random.normal(
                    0, measurement_noise_level, x_list.shape[1]
                )
            for k in range(1, x_list.shape[0] - 1):
                y_list[k] = (x_list[k + 1, :, 3:4] - x_list[k, :, 3:4]) / delta_t

            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
        else:
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
            if has_particle_dropout:
                torch.save(
                    x_removed_list,
                    f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
                )
                np.save(
                    f"graphs_data/{dataset_name}/particle_dropout_mask.npy",
                    particle_dropout_mask,
                )
                np.save(
                    f"graphs_data/{dataset_name}/inv_particle_dropout_mask.npy",
                    inv_particle_dropout_mask,
                )

        torch.save(model.p, f"graphs_data/{dataset_name}/model_p_{run}.pt")
        print("data saved ...")


        if run == run_vizualized:
            plot_synaptic_activity_traces(x_list, n_neurons, n_frames, dataset_name, model=model)
            plot_synaptic_mlp_functions(model, x_list, n_neurons, dataset_name, config.plotting.colormap, device,
                                        signal_model_name=config.graph_model.signal_model_name)
            # Plot external input field at key frames (Fig 3a)
            frame_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4]
            plot_external_input_field(x_list, n_neurons, dataset_name,
                                      frame_indices=frame_indices,
                                      n_input_neurons=n_input_neurons)
            # Plot activity sample (Fig 3c)
            plot_activity_sample(x_list, n_neurons, n_frames, dataset_name, n_traces=10)

            # SVD analysis of activity
            print('svd analysis ...')
            from neural_gnn.models.utils import analyze_data_svd
            style_param = 'dark_background' if 'black' in style else None
            analyze_data_svd(x_list, folder, config=config, style=style_param, save_in_subfolder=False, log_file=log_file)

    # close logfile only if we created it locally
    if local_log_file:
        log_file.close()
