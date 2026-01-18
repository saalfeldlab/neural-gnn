from typing import Optional, Literal, Annotated
import yaml
from pydantic import BaseModel, ConfigDict, Field

# Sub-config schemas for NeuralGraph


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: int = 2
    n_frames: int = 1000
    start_frame: int = 0
    seed: int = 42

    model_id: str = "000"
    ensemble_id: str = "0000"

    sub_sampling: int = 1
    delta_t: float = 1

    boundary: Literal["periodic", "no", "periodic_special", "wall"] = "periodic"
    min_radius: float = 0.0
    max_radius: float = 0.1

    n_neurons: int = 1000
    n_neuron_types: int = 5
    n_input_neurons: int = 0
    n_edges: int = 0
    max_edges: float = 1.0e6
    n_extra_null_edges: int = 0

    baseline_value: float = -999.0
    shuffle_neuron_types: bool = False

    noise_visual_input: float = 0.0
    only_noise_visual_input: float = 0.0
    visual_input_type: str = ""  # for flyvis experiments
    blank_freq: int = 2  # Frequency of blank frames in visual input
    simulation_initial_state: bool = False


    # external input configuration
    external_input_type: Literal["none", "signal", "visual", "modulation"] = "none"
    external_input_mode: Literal["additive", "multiplicative", "none"] = "none"
    permutation: bool = False  # whether to apply random permutation to external input

    # signal input parameters (external_input_type == "signal")
    signal_input_type: Literal["oscillatory", "triggered"] = "oscillatory"
    oscillation_max_amplitude: float = 1.0
    oscillation_frequency: float = 5.0

    # triggered oscillation parameters (signal_input_type == "triggered")
    triggered_n_impulses: int = 5  # number of impulse events
    triggered_n_input_neurons: int = 10  # number of neurons receiving impulse input per event
    triggered_impulse_strength: float = 5.0  # base strength of impulse (will vary randomly)
    triggered_min_start_frame: int = 50  # minimum frame for first trigger
    triggered_max_start_frame: int = 150  # maximum frame for first trigger (ignored if n_impulses > 1)
    triggered_duration_frames: int = 200  # duration of oscillation response per impulse
    triggered_amplitude_range: list[float] = [0.5, 2.0]  # min/max amplitude multiplier
    triggered_frequency_range: list[float] = [0.5, 2.0]  # min/max frequency multiplier

    tile_contrast: float = 0.2
    tile_corr_strength: float = 0.0   # correlation knob for tile_mseq / tile_blue_noise
    tile_flip_prob: float = 0.05      # per-frame random flip probability
    tile_seed: int = 42

    n_nodes: Optional[int] = None
    node_value_map: Optional[str] = "input_data/pattern_Null.tif"

    adjacency_matrix: str = ""
    short_term_plasticity_mode: str = "depression"

    connectivity_file: str = ""
    connectivity_init: list[float] = [-1]
    connectivity_filling_factor: float = 1
    connectivity_type: str = "none"  # none, Lorentz, Gaussian, uniform, chaotic, ring attractor, low_rank, successor, null, Lorentz_structured_X_Y
    connectivity_rank: int = 1
    connectivity_parameter: float = 1.0

    Dale_law: bool = False
    Dale_law_factor: float = 0.5  # fraction of excitatory (positive) columns, rest are inhibitory

    excitation_value_map: Optional[str] = None
    excitation: str = "none"

    params: list[list[float]]
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

    calcium_type: Literal["none", "leaky", "multi-compartment", "saturation"] = "none"
    calcium_activation: Literal["softplus", "relu", "identity", "tanh"] = "softplus"
    calcium_tau: float = 0.5  # decay time constant (same units as delta_t)
    calcium_alpha: float = 1.0  # scale factor to convert [Ca] to fluorescence
    calcium_beta: float = 0.0  # baseline offset for fluorescence
    calcium_initial: float = 0.0  # initial calcium concentration
    calcium_noise_level: float = 0.0  # optional Gaussian noise added to [Ca] updates
    calcium_saturation_kd: float = 1.0  # for nonlinear saturation models
    calcium_num_compartments: int = 1
    calcium_down_sample: int = 1  # down-sample [Ca] time series by this factor

    pos_init: str = "uniform"
    dpos_init: float = 0



class ClaudeConfig(BaseModel):
    """Configuration for Claude-driven exploration experiments."""
    model_config = ConfigDict(extra="forbid")

    n_epochs: int = 1  # number of epochs per iteration
    data_augmentation_loop: int = 100  # data augmentation loop count
    n_iter_block: int = 24  # number of iterations per simulation block
    ucb_c: float = 1.414  # UCB exploration constant: UCB(k) = R²_k + c * sqrt(ln(N) / n_k)


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    particle_model_name: str = ""
    cell_model_name: str = ""
    mesh_model_name: str = ""
    signal_model_name: str = ""
    prediction: Literal["first_derivative", "2nd_derivative","next_activity"] = "2nd_derivative"
    integration: Literal["Euler", "Runge-Kutta"] = "Euler"

    aggr_type: str
    embedding_dim: int = 2

    field_type: str = ""
    field_grid: Optional[str] = ""

    input_size: int = 1
    output_size: int = 1
    hidden_dim: int = 1
    n_layers: int = 1

    input_size_2: int = 1
    output_size_2: int = 1
    hidden_dim_2: int = 1
    n_layers_2: int = 1


    input_size_decoder: int = 1
    output_size_decoder: int = 1
    hidden_dim_decoder: int = 1
    n_layers_decoder: int = 1

    input_size_encoder: int = 1
    output_size_encoder: int = 1
    hidden_dim_encoder: int = 1
    n_layers_encoder: int = 1

    lin_edge_positive: bool = False

    update_type: Literal[
        "linear",
        "mlp",
        "pre_mlp",
        "2steps",
        "none",
        "no_pos",
        "generic",
        "excitation",
        "generic_excitation",
        "embedding_MLP",
        "test_field",
    ] = "none"

    MLP_activation: Literal[
        "relu", 
        "tanh", 
        "sigmoid", 
        "leaky_relu", 
        "soft_relu", 
        "none"
    ] = "relu"


    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1

    kernel_type: str = "mlp"

    input_size_nnr: int = 3
    n_layers_nnr: int = 5
    hidden_dim_nnr: int = 128
    output_size_nnr: int = 1
    outermost_linear_nnr: bool = True
    omega: float = 80.0

    input_size_nnr_f: int = 3
    n_layers_nnr_f: int = 5
    hidden_dim_nnr_f: int = 128
    output_size_nnr_f: int = 1
    outermost_linear_nnr_f: bool = True
    omega_f: float = 80.0
    omega_f_learning: bool = False  # make omega learnable during training

    nnr_f_xy_period: float = 1.0
    nnr_f_T_period: float = 1.0

    # INR type for external input learning
    # siren_t: input=t, output=n_neurons (current implementation, works for n_neurons < 100)
    # siren_id: input=(t, id), output=1 (scales better for large n_neurons)
    # siren_x: input=(t, x, y), output=1 (uses neuron positions)
    # ngp: instantNGP hash encoding
    # lowrank: low-rank matrix factorization U @ V (not a neural network)
    inr_type: Literal["siren_t", "siren_id", "siren_x", "ngp", "lowrank"] = "siren_t"

    # LowRank factorization parameters
    lowrank_rank: int = 64  # rank of the factorization (params = rank * (n_frames + n_neurons))
    lowrank_svd_init: bool = True  # initialize with SVD of the data

    # InstantNGP (hash encoding) parameters
    ngp_n_levels: int = 24
    ngp_n_features_per_level: int = 2
    ngp_log2_hashmap_size: int = 22
    ngp_base_resolution: int = 16
    ngp_per_level_scale: float = 1.4
    ngp_n_neurons: int = 128
    ngp_n_hidden_layers: int = 4

    input_size_modulation: int = 2
    n_layers_modulation: int = 3
    hidden_dim_modulation: int = 64
    output_size_modulation: int = 1

    input_size_excitation: int = 3
    n_layers_excitation: int = 5
    hidden_dim_excitation: int = 128

    excitation_dim: int = 1

    latent_dim: int = 64
    latent_update_steps: int = 50
    stochastic_latent: bool = True
    latent_init_std: float = 1.0  # only used if you later add 'init from noise' modes

    # encoder sizes (x -> [mu, logvar])
    input_size_encoder: int = 1      # set to n_neurons in your YAML
    n_layers_encoder: int = 3
    hidden_dim_encoder: int = 256
    latent_n_layers_update: int = 2
    latent_hidden_dim_update: int = 64
    output_size_decoder: int = 1      # set to n_neurons in your YAML
    n_layers_decoder: int = 3
    hidden_dim_decoder:  int = 256


class ZarrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_fluo: str = ""
    store_seg: str = ""

    axis: int = 0
    frame: int = 0
    contrast: str = "1,99.9"
    rendering: str = "1,99.9"
    dz_um: float = 4
    dy_um: float = 0.406
    dx_um: float = 0.406
    labels_opacity: float = 0.7
    show_boundaries: bool = False   


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colormap: str = "tab10"
    arrow_length: int = 10
    marker_size: int = 100
    xlim: list[float] = [-0.1, 0.1]
    ylim: list[float] = [-0.1, 0.1]
    embedding_lim: list[float] = [-40, 40]
    speedlim: list[float] = [0, 1]
    pic_folder: str = "none"
    pic_format: str = "jpg"
    pic_size: list[int] = [1000, 1100]
    data_embedding: int = 1
    plot_batch_size: int = 1000
    label_style: str = "MLP"  # "MLP" for MLP_0, MLP_1 labels; "greek" for phi, f labels

    # MLP plot axis limits
    mlp0_xlim: list[float] = [-5, 5]
    mlp0_ylim: list[float] = [-8, 8]
    mlp1_xlim: list[float] = [-5, 5]
    mlp1_ylim: list[float] = [-1.1, 1.1]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda:\d+)$")] = "auto"

    n_epochs: int = 20
    n_epochs_init: int = 99999
    epoch_reset: int = -1
    epoch_reset_freq: int = 99999
    batch_size: int = 1
    batch_ratio: float = 1
    small_init_batch_size: bool = True
    embedding_step: int = 1000
    shared_embedding: bool = False
    embedding_trial: bool = False
    remove_self: bool = True

    pretrained_model: str = ""
    pre_trained_W: str = ""

    multi_connectivity: bool = False
    with_connectivity_mask: bool = False
    has_missing_activity: bool = False

    epoch_distance_replace: int = 20
    warm_up_length: int = 10
    sequence_length: int = 32

    denoiser: bool = False
    denoiser_type: Literal["none", "window", "LSTM", "Gaussian_filter", "wavelet"] = ("none")
    denoiser_param: float = 1.0

    training_selected_neurons: bool = False
    selected_neuron_ids: list[int] = [1]

    time_window: int = 0

    n_runs: int = 2
    seed: int = 42
    clamp: float = 0
    pred_limit: float = 1.0e10

    particle_dropout: float = 0
    n_ghosts: int = 0
    ghost_method: Literal["none", "tensor", "MLP"] = "none"
    ghost_logvar: float = -12

    sparsity_freq: int = 5
    sparsity: Literal[
        "none",
        "replace_embedding",
        "replace_embedding_function",
        "replace_state",
        "replace_track",
    ] = "none"
    fix_cluster_embedding: bool = False
    cluster_method: Literal[
        "kmeans",
        "kmeans_auto_plot",
        "kmeans_auto_embedding",
        "distance_plot",
        "distance_embedding",
        "distance_both",
        "inconsistent_plot",
        "inconsistent_embedding",
        "none",
    ] = "distance_plot"
    cluster_distance_threshold: float = 0.01
    cluster_connectivity: Literal["single", "average"] = "single"

    Ising_filter: str = "none"

    init_training_single_type: bool = False
    training_single_type: bool = False

    low_rank_factorization: bool = False
    low_rank: int = 20

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    learning_rate_update_start: float = 0.0
    learning_rate_modulation_start: float = 0.0001
    learning_rate_W_start: float = 0.0001

    learning_rate_end: float = 0.0005
    learning_rate_embedding_end: float = 0.0001
    learning_rate_modulation_end: float = 0.0001
    Learning_rate_W_end: float = 0.0001

    learning_rate_missing_activity: float = 0.0001
    learning_rate_NNR: float = 0.0001
    learning_rate_NNR_f_start: float = 0.0
    learning_rate_NNR_f: float = 0.0001
    learning_rate_omega_f: float = 0.0001
    coeff_omega_f_L2: float = 0.0
    training_NNR_start_epoch: int = 0

    learning_rate_encoder: float = 0.0001
    learning_rate_latent_update: float = 0.0001
    learning_rate_decoder: float = 0.0001

    coeff_W_L1: float = 0.0
    coeff_W_L1_rate: float = 0.5
    coeff_W_L1_ghost: float = 0
    coeff_W_L2: float = 0.0
    coeff_W_sign: float = 0
    W_sign_temperature: float = 10.0

    coeff_lin_phi_zero: float = 0
    coeff_entropy_loss: float = 0
    coeff_edge_diff: float = 0
    coeff_update_diff: float = 0
    coeff_update_msg_diff: float = 0
    coeff_update_msg_sign: float = 0
    coeff_update_u_diff: float = 0
    coeff_NNR_f: float = 0

    coeff_permutation: float = 100

    coeff_TV_norm: float = 0
    coeff_missing_activity: float = 0
    coeff_edge_norm: float = 0

    coeff_edge_weight_L1: float = 0
    coeff_edge_weight_L1_rate: float = 0.5
    coeff_phi_weight_L1: float = 0
    coeff_phi_weight_L1_rate: float = 0.5

    coeff_edge_weight_L2: float = 0
    coeff_phi_weight_L2: float = 0

    coeff_model_a: float = 0
    coeff_model_b: float = 0
    coeff_lin_modulation: float = 0
    coeff_continuous: float = 0

    noise_level: float = 0
    measurement_noise_level: float = 0
    noise_model_level: float = 0
    loss_noise_level: float = 0.0

    # external input learning
    learn_external_input: bool = False

    data_augmentation_loop: int = 40

    recurrent_training: bool = False
    recurrent_training_start_epoch: int = 0
    recurrent_loop: int = 0
    noise_recurrent_level: float = 0.0

    neural_ODE_training: bool = False
    ode_method: Literal["dopri5", "rk4", "euler", "midpoint", "heun3"] = "dopri5"
    ode_rtol: float = 1e-4
    ode_atol: float = 1e-5
    ode_adjoint: bool = True
    ode_state_clamp: float = 10.0
    ode_stab_lambda: float = 0.0
    grad_clip_W: float = 0.0

    time_step: int = 1
    recurrent_sequence: str = ""
    recurrent_parameters: list[float] = [0, 0]

    regul_matrix: bool = False
    sub_batches: int = 1
    sequence: list[str] = ["to track", "to cell"]

    MPM_trainer : str = "F"



class NeuralGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = "NeuralGraph"
    dataset: str
    data_folder_name: str = "none"
    connectome_folder_name: str = "none"
    data_folder_mesh_name: str = "none"
    config_file: str = "none"

    
    simulation: SimulationConfig
    graph_model: GraphModelConfig
    claude: Optional[ClaudeConfig] = None
    plotting: PlottingConfig
    training: TrainingConfig
    zarr: Optional[ZarrConfig] = None

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return NeuralGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == "__main__":
    config_file = "../../config/arbitrary_3.yaml"  # Insert path to config file
    config = NeuralGraphConfig.from_yaml(config_file)
    print(config.pretty())

    print("Successfully loaded config file. Model description:", config.description)