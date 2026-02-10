# %% [raw]
# ---
# title: "Supplementary Figures 8 and 9: Sparse connectivity (5% to 100%)"
# author: Cédric Allier, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
# execute:
#   echo: false
# image: "graphs_data/signal/signal_fig_supp_8/connectivity_matrix.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Supplementary Figures 8 and 9**.
# Performance of GNN for connectivity matrices with varying sparsity levels.
# This notebook displays connectivity matrix comparison, $\phi^*$ plots, $\psi^*$ plots, and learned embedding for each sparsity level.
#
# **Simulation parameters (constant across all experiments):**
#
# - N_neurons: 1000
# - N_types: 4 parameterized by $\tau_i$={0.5,1}, $s_i$={1,2} and $g_i$=10
# - N_frames: 100,000
# - Connectivity weights: random, Cauchy distribution
#
# The simulation follows Equation 2 from the paper:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \tanh(x_j)$$
#
# **Variable: Connectivity sparsity**
#
# | Config | Sparsity |
# |--------|----------|
# | signal_fig_supp_8 | 5% |
# | signal_fig_supp_8_3 | 10% |
# | signal_fig_supp_8_2 | 20% |
# | signal_fig_supp_8_1 | 50% |
# | signal_fig_2 | 100% |

# %%
#| output: false
import os
import warnings

from neural_gnn.config import NeuralGraphConfig
from neural_gnn.generators.graph_data_generator import data_generate
from neural_gnn.models.graph_trainer import data_train
from neural_gnn.utils import set_device, add_pre_folder, load_and_display
from GNN_PlotFigure import data_plot, plot_r2_over_iterations

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Configuration

# %%
#| echo: true
#| output: false
import glob

print()
print("=" * 80)
print("Supplementary Figure 8: Effect of Connectivity Sparsity")
print("=" * 80)

# All configs to process (config_name, sparsity)
config_list = [
    ('signal_fig_supp_8', '5%'),
    ('signal_fig_supp_8_3', '10%'),
    ('signal_fig_supp_8_2', '20%'),
    ('signal_fig_supp_8_1', '50%'),
    ('signal_fig_2', '100%'),
]

device = []
best_model = ''
config_root = "./config"

# %% [markdown]
# ## Steps 1-3: Generate, Train, and Plot for all configs
# Loop over all sparsity levels: generate data, train GNN, and generate plots.
# Skips steps if data/models already exist.

# %%
#| echo: true
#| output: false
for config_file_, sparsity in config_list:
    print()
    print("=" * 80)
    print(f"Processing: {config_file_} ({sparsity} sparsity)")
    print("=" * 80)

    config_file, pre_folder = add_pre_folder(config_file_)

    # Load config
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.config_file = config_file
    config.dataset = config_file

    if device == []:
        device = set_device(config.training.device)

    log_dir = f'./log/{config_file}'
    graphs_dir = f'./graphs_data/{config_file}'

    # STEP 1: GENERATE
    print()
    print("-" * 80)
    print("STEP 1: GENERATE - Simulating neural activity")
    print("-" * 80)

    data_file = f'{graphs_dir}/x_list_0.npy'
    if os.path.exists(data_file):
        print(f"data already exists at {graphs_dir}/")
        print("skipping simulation, regenerating figures...")
        data_generate(
            config,
            device=device,
            visualize=False,
            run_vizualized=0,
            style="color",
            alpha=1,
            erase=False,
            bSave=True,
            step=2,
            regenerate_plots_only=True,
        )
    else:
        print(f"simulating {config.simulation.n_neurons} neurons, {config.simulation.n_frames} frames")
        print(f"output: {graphs_dir}/")
        data_generate(
            config,
            device=device,
            visualize=False,
            run_vizualized=0,
            style="color",
            alpha=1,
            erase=False,
            bSave=True,
            step=2,
        )

    # STEP 2: TRAIN
    print()
    print("-" * 80)
    print("STEP 2: TRAIN - Training GNN")
    print("-" * 80)

    model_files = glob.glob(f'{log_dir}/models/*.pt')
    if model_files:
        print(f"trained model already exists at {log_dir}/models/")
        print("skipping training (delete models folder to retrain)")
    else:
        print(f"training for {config.training.n_epochs} epochs")
        print(f"sparsity: {sparsity}")
        data_train(
            config=config,
            erase=False,
            best_model=best_model,
            style='color',
            device=device
        )

    # STEP 3: PLOT
    print()
    print("-" * 80)
    print("STEP 3: PLOT - Generating figures")
    print("-" * 80)

    folder_name = f'{log_dir}/tmp_results/'
    os.makedirs(folder_name, exist_ok=True)

    data_plot(
        config=config,
        config_file=config_file,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
        apply_weight_correction=True,
        plot_eigen_analysis=False
    )

    # STEP 4: TRAINING PROGRESSION (R² over iterations)
    print()
    print("-" * 80)
    print("STEP 4: TRAINING PROGRESSION - Computing R² over iterations")
    print("-" * 80)

    r2_file = f'{log_dir}/results/all/r2_over_iterations.json'
    if os.path.exists(r2_file):
        print(f"R² data already exists at {r2_file}")
        print("skipping (delete results/all/ folder to recompute)")
    else:
        data_plot(
            config=config,
            config_file=config_file,
            epoch_list=['all'],
            style='color',
            extended='plots',
            device=device,
            apply_weight_correction=True,
            plot_eigen_analysis=False,
        )

# %% [markdown]
# ## Activity Time Series
#
# Sample of 100 time series for each sparsity level.

# %%
#| fig-cap: "Supp. Fig 8b: Sample of 100 time series (5% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8/activity.png")

# %%
#| fig-cap: "Sample of 100 time series (10% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8_3/activity.png")

# %%
#| fig-cap: "Sample of 100 time series (20% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8_2/activity.png")

# %%
#| fig-cap: "Sample of 100 time series (50% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8_1/activity.png")

# %%
#| fig-cap: "Sample of 100 time series (100% connectivity)"
load_and_display("./graphs_data/signal/signal_fig_2/activity.png")

# %% [markdown]
# ## True Connectivity Matrix $W_{ij}$
#
# True connectivity matrix for each sparsity level.

# %%
#| fig-cap: "Supp. Fig 8c: True connectivity $W_{ij}$ (5% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8/connectivity_matrix.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$ (10% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8_3/connectivity_matrix.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$ (20% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8_2/connectivity_matrix.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$ (50% sparsity)"
load_and_display("./graphs_data/signal/signal_fig_supp_8_1/connectivity_matrix.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$ (100% connectivity)"
load_and_display("./graphs_data/signal/signal_fig_2/connectivity_matrix.png")

# %% [markdown]
# ## Connectivity Matrix Comparison
#
# Learned vs true connectivity matrix $W_{ij}$ after training.
# The scatter plot shows $R^2$ and slope metrics.

# %%
#| fig-cap: "Supp. Fig 8e: Connectivity comparison (5% sparsity)"
load_and_display("./log/signal/signal_fig_supp_8/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (10% sparsity)"
load_and_display("./log/signal/signal_fig_supp_8_3/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (20% sparsity)"
load_and_display("./log/signal/signal_fig_supp_8_2/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (50% sparsity)"
load_and_display("./log/signal/signal_fig_supp_8_1/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (100% connectivity)"
load_and_display("./log/signal/signal_fig_2/results/weights_comparison_corrected.png")

# %% [markdown]
# ## Update Function $\phi^*(\mathbf{a}_i, x)$ (MLP0)
#
# Learned update functions after training. Each curve represents one neuron.
# Colors indicate true neuron types. True functions overlaid in gray.

# %%
#| fig-cap: "Supp. Fig 8g: Update functions $\\phi^*(a_i, x)$ (5% sparsity). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (10% sparsity). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8_3/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (20% sparsity). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8_2/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (50% sparsity). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8_1/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (100% connectivity). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_2/results/MLP0.png")

# %% [markdown]
# ## Transfer Function $\psi^*(x)$ (MLP1)
#
# Learned transfer function after training, normalized to max=1.
# True function overlaid in gray.

# %%
#| fig-cap: "Supp. Fig 8h: Transfer function $\\psi^*(x)$ (5% sparsity). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (10% sparsity). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8_3/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (20% sparsity). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8_2/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (50% sparsity). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_8_1/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (100% connectivity). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_2/results/MLP1_corrected.png")

# %% [markdown]
# ## Latent Embeddings $\mathbf{a}_i$
#
# Learned latent vectors for all neurons. Colors indicate true neuron types.

# %%
#| fig-cap: "Supp. Fig 8f: Latent embeddings $a_i$ (5% sparsity)."
load_and_display("./log/signal/signal_fig_supp_8/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (10% sparsity)."
load_and_display("./log/signal/signal_fig_supp_8_3/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (20% sparsity)."
load_and_display("./log/signal/signal_fig_supp_8_2/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (50% sparsity)."
load_and_display("./log/signal/signal_fig_supp_8_1/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (100% connectivity)."
load_and_display("./log/signal/signal_fig_2/results/embedding.png")

# %% [markdown]
# ## R² Connectivity Over Training Iterations
# 1000 densely connected neurons with 4 neuron-dependent update functions.
# The plot displays $R^2$ for the comparison between true and learned connectivity matrices $W_{ij}$
# as a function of training iterations for different connectivity filling factors (colors).
# All comparisons are made at equal numbers of gradient descent iterations.

# %%
#| echo: true
#| output: false
print()
print("-" * 80)
print("Generating R² over iterations comparison plot")
print("-" * 80)
output_r2 = plot_r2_over_iterations(
    config_list=config_list,
    output_path='./log/signal/tmp_results/r2_over_iterations_sparsity.png',
    device=device,
)

# %%
#| fig-cap: "1000 densely connected neurons with 4 neuron-dependent update functions. $R^2$ for the comparison between true and learned connectivity matrices $W_{ij}$ as a function of training iterations for different connectivity filling factors (colors). All comparisons are made at equal numbers of gradient descent iterations."
load_and_display('./log/signal/tmp_results/r2_over_iterations_sparsity.png')
