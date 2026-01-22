# %% [raw]
# ---
# title: "Supplementary Figure 14: neuron-dependent transfer functions"
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
#   - Neuron-dependent
# execute:
#   echo: false
# image: "log/signal/signal_fig_supp_14/results/MLP1_neuron_neuron.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Supplementary Figure 14**.
# Training with neuron-neuron dependent transfer functions (transmitters & receptors)
# of the form $\psi(\mathbf{a}_i, \mathbf{a}_j, x_j)$.
#
# **Simulation parameters:**
#
# - N_neurons: 1000
# - N_types: 4 parameterized by $\tau_i$={0.5,1}, $s_i$={1,2} and $g_i$=10
# - N_frames: 100,000
# - Connectivity: 100% (dense)
# - Connectivity weights: random, Lorentz distribution
# - Noise: none
# - External inputs: none
# - Transfer function width $\gamma_i$={1,2,4,8} (receiver-dependent)
# - Linear slope $\theta_j$={0, 0.013, 0.027, 0.040} (sender-dependent)
#
# The simulation follows an extended version of Equation 2:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \psi_{ij}(x_j)$$
#
# where the transfer function depends on both sender $j$ and receiver $i$:
#
# $$\psi_{ij}(x_j) = \tanh\left(\frac{x_j}{\gamma_i}\right) - \theta_j \cdot x_j$$
#
# The GNN jointly optimizes the shared MLP $\psi^*$ and latent vectors $\mathbf{a}_i$ to
# accurately identify the neuron-neuron dependent transfer functions:
#
# $$\hat{\dot{x}}_i = \phi^*(\mathbf{a}_i, x_i) + \sum_j W_{ij} \cdot \psi^*(\mathbf{a}_i, \mathbf{a}_j, x_j)$$

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
print("Supplementary Figure 14: 1000 neurons, 4 types, neuron-dependent transfer functions")
print("=" * 80)

device = []
best_model = ''
config_file_ = 'signal_fig_supp_14'

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
# Generate synthetic neural activity data using the PDE_N5 model with neuron-dependent
# transfer functions. Each pair of neuron types has different transfer function characteristics
# depending on both source ($\mathbf{a}_j$) and target ($\mathbf{a}_i$) embeddings.
#
# **Outputs:**
#
# - Sample time series
# - True connectivity matrix $W_{ij}$

# %%
#| echo: true
#| output: false
# STEP 1: GENERATE
print()
print("-" * 80)
print("STEP 1: GENERATE - Simulating neural activity (neuron-dependent transfer functions)")
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
    print(f"  Generating {config.simulation.n_frames} time frames")
    print(f"  Transfer function widths gamma_i = [1, 2, 4, 8]")
    print(f"  Output: {graphs_dir}/")
    print()
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

# %%
#| fig-cap: "Sample time series taken from the activity data (neuron-dependent transfer functions)."
load_and_display(f"./graphs_data/signal/signal_fig_supp_14/activity.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$. The inset shows 20×20 weights."
load_and_display("./graphs_data/signal/signal_fig_supp_14/connectivity_matrix.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN to learn connectivity W, latent embeddings $a_i$, and functions $\phi^*, \psi^*$.
# The GNN must learn neuron-neuron dependent transfer functions $\psi^*(a_i, a_j, x_j)$.

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN to learn neuron-dependent transfer functions")
print("-" * 80)

# Check if trained model already exists (any .pt file in models folder)
import glob
model_files = glob.glob(f'{log_dir}/models/*.pt')
if model_files:
    print(f"  Trained model already exists at {log_dir}/models/")
    print("  Skipping training (delete models folder to retrain)")
else:
    print(f"  Training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
    print(f"  Learning: connectivity W, latent vectors a_i, neuron-dependent psi*(a_i, a_j, x_j)")
    print(f"  Models: {log_dir}/models/")
    print(f"  Training plots: {log_dir}/tmp_training")
    print(f"  Tensorboard: tensorboard --logdir {log_dir}/")
    print()
    data_train(
        config=config,
        erase=False,
        best_model=best_model,
        style='color',
        device=device
    )

# %% [markdown]
# ## Step 3: GNN Evaluation
# Figures matching Supplementary Figure 14 from the paper.
#
# **Figure panels:**
#
# - (a) Activity time series used for GNN training (10^5 time-points)
# - (b) Sample of 10 time series taken from (a)
# - (c) True connectivity $W_{ij}$
# - (d) Learned connectivity
# - (e) Comparison between learned and true connectivity
# - (f) Learned latent vectors $\mathbf{a}_i$
# - (g) Learned update functions $\phi^*(\mathbf{a}, x)$
# - (h) Learned transfer functions $\psi^*(\mathbf{a}_i, \mathbf{a}_j, x)$ (colors indicate true neuron types, true functions overlaid in light gray)

# %%
#| echo: true
#| output: false
# STEP 3: GNN EVALUATION
print()
print("-" * 80)
print("STEP 3: GNN EVALUATION - Generating Supplementary Figure 14 panels")
print("-" * 80)
print(f"  Learned connectivity matrix")
print(f"  W learned vs true (R^2, slope)")
print(f"  Latent vectors a_i (4 clusters)")
print(f"  Update functions phi*(a_i, x)")
print(f"  Transfer functions psi*(a_i, a_j, x) - neuron-neuron dependent")
print(f"  Output: {log_dir}/results/")
print()
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)

# %% [markdown]
# ### Supplementary Figure 14: GNN Evaluation Results

# %%
#| fig-cap: "Learned connectivity."
load_and_display("./log/signal/signal_fig_supp_14/results/connectivity_learned.png")

# %%
#| fig-cap: "Comparison of learned and true connectivity (given $g_i$=10). Expected: $R^2$=0.99, slope=0.99."
load_and_display("./log/signal/signal_fig_supp_14/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Learned latent vectors $a_i$ of all neurons."
load_and_display("./log/signal/signal_fig_supp_14/results/embedding.png")

# %%
#| fig-cap: "Learned update functions $\\phi^*(a_i, x)$. The plot shows 1000 overlaid curves. Colors indicate true neuron types. True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_14/results/MLP0.png")

# %%
#| fig-cap: "Learned transfer functions $\\psi^*(a_i, a_j, x_j)$. 2x2 montage: each panel corresponds to a receiving neuron type (border color), showing curves for all sending neuron types (line colors). True functions in gray."
load_and_display("./log/signal/signal_fig_supp_14/results/MLP1_neuron_neuron.png")
