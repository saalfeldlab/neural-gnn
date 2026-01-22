# %% [raw]
# ---
# title: "Supplementary Figure 7: Effect of training dataset size"
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
# execute:
#   echo: false
# image: "log/signal/signal_fig_supp_7_5/results/MLP0.png"
# ---

# %% [markdown]
# This script reproduce the panels of paper's **Supplementary Figure 7**. 
# Performance scales with the length of the training series.
#
# **Supplementary Figure 7** shows R² vs epochs for different training dataset sizes.
# This notebook displays connectivity matrix comparison and MLP0 plots for each dataset size.
#
# **Simulation parameters (constant across all experiments):**
#
# - N_neurons: 1000
# - N_types: 4 parameterized by $\tau_i$={0.5,1}, $s_i$={1,2} and $g_i$=10
# - Connectivity: 100% (dense), Lorentz distribution
#
# **Variable: Training dataset size (n_frames)**
#
# | Config | n_frames |
# |--------|----------|
# | signal_fig_2 | 100,000 |
# | signal_fig_supp_7_1 | 50,000 |
# | signal_fig_supp_7_2 | 40,000 |
# | signal_fig_supp_7_3 | 30,000 |
# | signal_fig_supp_7_4 | 20,000 |
# | signal_fig_supp_7_5 | 10,000 |

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
device = []
best_model = ''
config_root = "./config"

# List of config files for Supplementary Figure 7
config_files = [
    'signal_fig_2',         # 100,000 frames (baseline)
    'signal_fig_supp_7_1',  # 50,000 frames
    'signal_fig_supp_7_2',  # 40,000 frames
    'signal_fig_supp_7_3',  # 30,000 frames
    'signal_fig_supp_7_4',  # 20,000 frames
    'signal_fig_supp_7_5',  # 10,000 frames
]

# %% [markdown]
# ## Step 1: Generate Data
# Generate synthetic neural activity data for each dataset size.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("Supplementary Figure 7: Effect of Training Dataset Size")
print("=" * 80)

for config_file_ in config_files:
    print()
    print("-" * 80)
    print(f"GENERATE: {config_file_}")
    print("-" * 80)

    config_file, pre_folder = add_pre_folder(config_file_)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.config_file = config_file
    config.dataset = config_file

    if device == []:
        device = set_device(config.training.device)

    graphs_dir = f'./graphs_data/{config_file}'
    data_file = f'{graphs_dir}/x_list_0.npy'

    if os.path.exists(data_file):
        print(f"  Data already exists at {graphs_dir}/")
        print("  Skipping simulation...")
    else:
        print(f"  Simulating {config.simulation.n_neurons} neurons, {config.simulation.n_frames} frames")
        print(f"  Output: {graphs_dir}/")
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
# Train the GNN for each dataset size.

# %%
#| echo: true
#| output: false
for config_file_ in config_files:
    print()
    print("-" * 80)
    print(f"TRAIN: {config_file_}")
    print("-" * 80)

    config_file, pre_folder = add_pre_folder(config_file_)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.config_file = config_file
    config.dataset = config_file

    log_dir = f'./log/{config_file}'
    # Check if trained model already exists (any .pt file in models folder)
    import glob
    model_files = glob.glob(f'{log_dir}/models/*.pt')

    if model_files:
        print(f"  Trained model already exists at {log_dir}/models/")
        print("  Skipping training...")
    else:
        print(f"  Training for {config.training.n_epochs} epochs")
        print(f"  n_frames: {config.simulation.n_frames}")
        data_train(
            config=config,
            erase=False,
            best_model=best_model,
            style='color',
            device=device
        )

# %% [markdown]
# ## Step 3: Generate Plots
# Generate evaluation plots for each dataset size.

# %%
#| echo: true
#| output: false
for config_file_ in config_files:
    print()
    print("-" * 80)
    print(f"PLOT: {config_file_}")
    print("-" * 80)

    config_file, pre_folder = add_pre_folder(config_file_)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.config_file = config_file
    config.dataset = config_file

    log_dir = f'./log/{config_file}'
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
# ## Connectivity Matrix Comparison (Final)
#
# Learned vs true connectivity matrix $W_{ij}$ after training.
# The scatter plot shows $R^2$ and slope metrics.

# %%
#| fig-cap: "Connectivity comparison (100,000 frames - baseline)"
load_and_display("./log/signal/signal_fig_2/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (50,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_1/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (40,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_2/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (30,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_3/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (20,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_4/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (10,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_5/results/weights_comparison_corrected.png")

# %% [markdown]
# ## Update Function $\phi^*(\mathbf{a}_i, x)$ (MLP0) - Final
#
# Learned update functions after training. Each curve represents one neuron.
# Colors indicate true neuron types. True functions overlaid in gray.

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (100,000 frames - baseline)"
load_and_display("./log/signal/signal_fig_2/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (50,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_1/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (40,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_2/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (30,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_3/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (20,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_4/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (10,000 frames)"
load_and_display("./log/signal/signal_fig_supp_7_5/results/MLP0.png")
