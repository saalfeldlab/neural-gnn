# %% [raw]
# ---
# title: "Supplementary Figure 7: Effect of training dataset size"
# author: Cédric Allier, Stephan Saalfeld
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
# This notebook displays connectivity matrix comparison, $\phi^*$ plots, $\psi^*$ plots, and learned embedding for each dataset size.
#
# **Simulation parameters (constant across all experiments):**
#
# - N_neurons: 1000
# - N_types: 4 parameterized by $\tau_i$={0.5,1}, $s_i$={1,2} and $g_i$=10
# - Connectivity: 100% (dense), Lorentz distribution
#
# The simulation follows Equation 2 from the paper:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \tanh(x_j)$$
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
print("Supplementary Figure 7: Effect of Training Dataset Size")
print("=" * 80)

# All configs to process (config_name, n_frames)
config_list = [
    ('signal_fig_2', 100000),
    ('signal_fig_supp_7_1', 50000),
    ('signal_fig_supp_7_2', 40000),
    ('signal_fig_supp_7_3', 30000),
    ('signal_fig_supp_7_4', 20000),
    ('signal_fig_supp_7_5', 10000),
]

device = []
best_model = ''
config_root = "./config"

# %% [markdown]
# ## Steps 1-3: Generate, Train, and Plot for all configs
# Loop over all dataset sizes: generate data, train GNN, and generate plots.
# Skips steps if data/models already exist.

# %%
#| echo: true
#| output: false
for config_file_, n_frames in config_list:
    print()
    print("=" * 80)
    print(f"Processing: {config_file_} (n_frames={n_frames:,})")
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
        print(f"n_frames: {config.simulation.n_frames}")
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
# ## Connectivity Matrix Comparison
#
# Learned vs true connectivity matrix $W_{ij}$ after training.
# The scatter plot shows $R^2$ and slope metrics.

# %%
#| fig-cap: "Fig 2e: Connectivity comparison (n_frames=100,000)"
load_and_display("./log/signal/signal_fig_2/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (n_frames=50,000)"
load_and_display("./log/signal/signal_fig_supp_7_1/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (n_frames=40,000)"
load_and_display("./log/signal/signal_fig_supp_7_2/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (n_frames=30,000)"
load_and_display("./log/signal/signal_fig_supp_7_3/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (n_frames=20,000)"
load_and_display("./log/signal/signal_fig_supp_7_4/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Connectivity comparison (n_frames=10,000)"
load_and_display("./log/signal/signal_fig_supp_7_5/results/weights_comparison_corrected.png")

# %% [markdown]
# ## Update Function $\phi^*(\mathbf{a}_i, x)$ (MLP0)
#
# Learned update functions after training. Each curve represents one neuron.
# Colors indicate true neuron types. True functions overlaid in gray.

# %%
#| fig-cap: "Fig 2g: Update functions $\\phi^*(a_i, x)$ (n_frames=100,000). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_2/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (n_frames=50,000). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_1/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (n_frames=40,000). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_2/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (n_frames=30,000). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_3/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (n_frames=20,000). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_4/results/MLP0.png")

# %%
#| fig-cap: "Update functions $\\phi^*(a_i, x)$ (n_frames=10,000). True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_5/results/MLP0.png")

# %% [markdown]
# ## Transfer Function $\psi^*(x)$ (MLP1)
#
# Learned transfer function after training, normalized to max=1.
# True function overlaid in gray.

# %%
#| fig-cap: "Fig 2h: Transfer function $\\psi^*(x)$ (n_frames=100,000). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_2/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (n_frames=50,000). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_1/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (n_frames=40,000). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_2/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (n_frames=30,000). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_3/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (n_frames=20,000). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_4/results/MLP1_corrected.png")

# %%
#| fig-cap: "Transfer function $\\psi^*(x)$ (n_frames=10,000). True function overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_7_5/results/MLP1_corrected.png")

# %% [markdown]
# ## Latent Embeddings $\mathbf{a}_i$
#
# Learned latent vectors for all neurons. Colors indicate true neuron types.

# %%
#| fig-cap: "Fig 2f: Latent embeddings $a_i$ (n_frames=100,000)."
load_and_display("./log/signal/signal_fig_2/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (n_frames=50,000)."
load_and_display("./log/signal/signal_fig_supp_7_1/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (n_frames=40,000)."
load_and_display("./log/signal/signal_fig_supp_7_2/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (n_frames=30,000)."
load_and_display("./log/signal/signal_fig_supp_7_3/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (n_frames=20,000)."
load_and_display("./log/signal/signal_fig_supp_7_4/results/embedding.png")

# %%
#| fig-cap: "Latent embeddings $a_i$ (n_frames=10,000)."
load_and_display("./log/signal/signal_fig_supp_7_5/results/embedding.png")

# %% [markdown]
# ## R² Connectivity Over Training Iterations
# $R^2$ between learned and true connectivity $W_{ij}$ plotted as a function
# of training iterations for each dataset size.

# %%
#| echo: true
#| output: false
print()
print("-" * 80)
print("Generating R² over iterations comparison plot")
print("-" * 80)
output_r2 = plot_r2_over_iterations(
    config_list=config_list,
    output_path='./log/signal/tmp_results/r2_over_iterations.png',
    device=device,
)

# %%
#| fig-cap: "$Supp. Fig 7: R^2$ connectivity over training iterations for different dataset sizes."
load_and_display('./log/signal/tmp_results/r2_over_iterations.png')
