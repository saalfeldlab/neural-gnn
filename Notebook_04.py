# %% [raw]
# ---
# title: "Supplementary Figure 8: Sparse connectivity (5% to 100%)"
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
# execute:
#   echo: false
# image: "graphs_data/signal/signal_fig_supp_8/connectivity_matrix.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Supplementary Figure 8**.
# Performance of GNN for connectivity matrices with varying sparsity levels.
# This notebook displays connectivity matrix comparison and $\phi^*$ plots for each sparsity level.
#
# **Simulation parameters (constant across all experiments):**
#
# - N_neurons: 1000
# - N_types: 4 parameterized by $\tau_i$={0.5,1}, $s_i$={1,2} and $g_i$=10
# - N_frames: 100,000
# - Connectivity weights: random, Cauchy distribution
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
from GNN_PlotFigure import data_plot

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Configuration

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("Supplementary Figure 8: Effect of Connectivity Sparsity")
print("=" * 80)

device = []
best_model = ''
config_file_ = 'signal_fig_supp_8'  # 5% sparsity

print()
config_root = "./config"
config_file, pre_folder = add_pre_folder(config_file_)

# load config
config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
config.config_file = config_file
config.dataset = config_file

if device == []:
    device = set_device(config.training.device)

log_dir = f'./log/{config_file}'
graphs_dir = f'./graphs_data/{config_file}'

# %% [markdown]
# ## Step 1: Generate Data
# Generate synthetic neural activity data.

# %%
#| echo: true
#| output: false
# STEP 1: GENERATE
print()
print("-" * 80)
print("STEP 1: GENERATE - Simulating neural activity")
print("-" * 80)

data_file = f'{graphs_dir}/x_list_0.npy'
if os.path.exists(data_file):
    print(f"data already exists at {graphs_dir}/")
    print("skipping simulation...")
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

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN.

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN")
print("-" * 80)

# Check if trained model already exists (any .pt file in models folder)
import glob
model_files = glob.glob(f'{log_dir}/models/*.pt')

if model_files:
    print(f"trained model already exists at {log_dir}/models/")
    print("skipping training...")
else:
    print(f"training for {config.training.n_epochs} epochs")
    print(f"sparsity: 5%")
    data_train(
        config=config,
        erase=False,
        best_model=best_model,
        style='color',
        device=device
    )

# %% [markdown]
# ## Step 3: Generate Plots
# Generate evaluation plots.

# %%
#| echo: true
#| output: false
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
    apply_weight_correction=True
)

# %% [markdown]
# ## Activity Time Series
#
# Sample of 100 time series for each sparsity level.

# %%
#| fig-cap: "Sample of 100 time series (5% sparsity)"
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
#| fig-cap: "True connectivity $W_{ij}$ (5% sparsity)"
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
#| fig-cap: "Connectivity comparison (5% sparsity)"
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
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (5% sparsity). True functions are overlaid in light gray."
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
