# %% [raw]
# ---
# title: "Supplementary Figure 10: effect of Gaussian noise"
# author: Cédric Allier, MichaelInnerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
#   - Noise
# execute:
#   echo: false
# image: "graphs_data/signal/signal_fig_supp_10/activity.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Supplementary Figure 10**.
# Gaussian noise i sinjected into the simulated dynamics (SNR of ∼10 dB). 
#
# **Simulation parameters:**
#
# - N_neurons: 1000
# - N_types: 4 parameterized by $\tau_i$={0.5,1}, $s_i$={1,2} and $g_i$=10
# - N_frames: 100,000
# - Connectivity: 100% (dense)
# - Connectivity weights: random, Lorentz distribution
# - Noise: Gaussian (~10 dB SNR)
# - External inputs: none
#
# The simulation follows Equation 2 from the paper:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \psi(x_j) + \eta_i(t)$$
#
# where $\eta_i(t)$ is Gaussian noise.

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
# ## Configuration and Setup

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("Supplementary Figure 10: 1000 neurons, 4 types, dense connectivity, Gaussian noise")
print("=" * 80)

device = []
best_model = ''
config_file_ = 'signal_fig_supp_10'

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
# Generate synthetic neural activity data with Gaussian noise using the PDE_N2 model.
# This creates the training dataset with 1000 neurons of 4 different types and 100,000 time points.
#
# **Outputs:**
#
# - Panel (a): Activity time series used for GNN training
# - Panel (b): Sample of 10 time series
# - Panel (c): True connectivity matrix $W_{ij}$

# %%
#| echo: true
#| output: false
# STEP 1: GENERATE
print()
print("-" * 80)
print("STEP 1: GENERATE - Simulating neural activity with Gaussian noise")
print("-" * 80)

# Check if data already exists
data_file = f'{graphs_dir}/x_list_0.npy'
if os.path.exists(data_file):
    print(f"  Data already exists at {graphs_dir}/")
    print("  Skipping simulation, regenerating figures...")
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
    print(f"  Simulating {config.simulation.n_neurons} neurons, {config.simulation.n_neuron_types} types")
    print(f"  Generating {config.simulation.n_frames} time frames with Gaussian noise")
    print(f"  Output: {graphs_dir}/")
    print()
    data_generate(
        config,
        device=device,
        visualize=False,
        run_vizualized=0,
        style="color",
        alpha=1,
        erase=True,
        bSave=True,
        step=2,
    )

# %%
#| fig-cap: "Panel (b): Sample of 10 time series taken from the activity data with Gaussian noise (~10 dB SNR)."
load_and_display(f"./graphs_data/signal/signal_fig_supp_10/activity.png")

# %%
#| fig-cap: "Panel (c): True connectivity $W_{ij}$. The inset shows 20×20 weights."
load_and_display("./graphs_data/signal/signal_fig_supp_10/connectivity_matrix.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN to learn connectivity W, latent embeddings $a_i$, and functions $\phi^*, \psi^*$.
# The GNN learns to predict $dx_i/dt$ from the noisy observed activity $x_i$.
#
# The GNN optimizes the update rule (Equation 3 from the paper):
#
# $$\hat{\dot{x}}_i = \phi^*(\mathbf{a}_i, x_i) + \sum_j W_{ij} \psi^*(x_j)$$
#
# where $\phi^*$ and $\psi^*$ are MLPs (ReLU, hidden dim=64, 3 layers).
# $\mathbf{a}_i$ is a learnable 2D latent vector per neuron, and $W$ is the learnable connectivity matrix.
#

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN to learn W, embeddings, phi, psi from noisy data")
print("-" * 80)

# Check if trained model already exists
model_file = f'{log_dir}/models/best_model_with_0_graphs_0_0.pt'
if os.path.exists(model_file):
    print(f"  Trained model already exists at {log_dir}/models/")
    print("  Skipping training (delete models folder to retrain)")
else:
    print(f"  Training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
    print(f"  Learning: connectivity W, latent vectors a_i, functions phi* and psi*")
    print(f"  Models: {log_dir}/models/")
    print(f"  Training plots: {log_dir}/tmp_training")
    print(f"  Tensorboard: tensorboard --logdir {log_dir}/")
    print()
    data_train(
        config=config,
        erase=True,
        best_model=best_model,
        style='black',
        device=device
    )

# %% [markdown]
# ## Step 3: GNN Evaluation
# Figures matching Supplementary Figure 10 from the paper.
#
# **Figure panels:**
#
# - Panel (d): Learned connectivity matrix
# - Panel (e): Comparison of learned vs true connectivity
# - Panel (f): Learned latent vectors $\mathbf{a}_i$
# - Panel (g): Learned update functions $\phi^*(\mathbf{a}_i, x)$
# - Panel (h): Learned transfer function $\psi^*(x)$

# %%
#| echo: true
#| output: false
# STEP 3: GNN EVALUATION
print()
print("-" * 80)
print("STEP 3: GNN EVALUATION - Generating Supplementary Figure 10 panels (d-h)")
print("-" * 80)
print(f"  Panel (d): Learned connectivity matrix")
print(f"  Panel (e): W learned vs true (R^2, slope)")
print(f"  Panel (f): Latent vectors a_i (4 clusters)")
print(f"  Panel (g): Update functions phi*(a_i, x)")
print(f"  Panel (h): Transfer function psi*(x)")
print(f"  Output: {log_dir}/results/")
print()
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)

# %% [markdown]
# ### Supplementary Figure 10: GNN Evaluation Results

# %%
#| fig-cap: "Panel (d): Learned connectivity."
load_and_display("./log/signal/signal_fig_supp_10/results/connectivity_learned.png")

# %%
#| fig-cap: "Panel (e): Comparison of learned and true connectivity (given $g_i$=10)."
load_and_display("./log/signal/signal_fig_supp_10/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Panel (f): Learned latent vectors $a_i$ of all neurons."
load_and_display("./log/signal/signal_fig_supp_10/results/embedding.png")

# %%
#| fig-cap: "Panel (g): Learned update functions $\\phi^*(a_i, x)$. The plot shows 1000 overlaid curves, one for each vector $a_i$. Colors indicate true neuron types. True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_10/results/MLP0.png")

# %%
#| fig-cap: "Panel (h): Learned transfer function $\\psi^*(x)$, normalized to a maximum value of 1. True function is overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_10/results/MLP1_corrected.png")
