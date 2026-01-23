# %% [raw]
# ---
# title: "Supplementary Figure 13: heterogeneous transfer functions"
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
#   - Transmitters
# execute:
#   echo: false
# image: "log/signal/signal_fig_supp_13/results/MLP1_corrected.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Supplementary Figure 13**.
# Test with neuron-specific transfer functions of the form $\psi(x_j/\gamma_i)$.
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
# - Transfer function $\gamma_i$={1,2,4,8} (neuron-specific)
#
# The simulation follows an extended version of Equation 2:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \psi\left(\frac{x_j}{\gamma_i}\right)$$
#
# The GNN jointly optimizes the shared MLP $\psi^*$ and latent vectors $\mathbf{a}_i$ to
# accurately identify the neuron-specific transfer functions.

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
print("Supplementary Figure 13: 1000 neurons, 4 types, heterogeneous transfer functions")
print("=" * 80)

device = []
best_model = ''
config_file_ = 'signal_fig_supp_13'

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
# Generate synthetic neural activity data using the PDE_N4 model with neuron-specific
# transfer functions. Each neuron type has a different parameter $\gamma_i$ in
# the transfer function $\psi(x_j/\gamma_i)$.
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
print("STEP 1: GENERATE - Simulating neural activity (heterogeneous transfer functions)")
print("-" * 80)

# Check if data already exists
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
    print(f"simulating {config.simulation.n_neurons} neurons, {config.simulation.n_neuron_types} types")
    print(f"generating {config.simulation.n_frames} time frames")
    print(f"transfer function gamma_i = [1, 2, 4, 8]")
    print(f"output: {graphs_dir}/")
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
#| fig-cap: "Sample time series taken from the activity data (heterogeneous transfer functions)."
load_and_display(f"./graphs_data/signal/signal_fig_supp_13/activity.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$. The inset shows 20×20 weights."
load_and_display("./graphs_data/signal/signal_fig_supp_13/connectivity_matrix.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN to learn connectivity $W$, latent embeddings $\mathbf{a}_i$, and functions $\phi^*, \psi^*$.
# The GNN must learn neuron-specific transfer functions $\psi(x_j/\gamma_i)$.
#
# The GNN optimizes the update rule:
#
# $$\hat{\dot{x}}_i = \phi^*(\mathbf{a}_i, x_i) + \sum_j W_{ij} \psi^*(\mathbf{a}_j, x_j)$$
#
# where the transfer function $\psi^*$ now depends on the latent vector $\mathbf{a}_j$.

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN to learn heterogeneous transfer functions")
print("-" * 80)

# Check if trained model already exists (any .pt file in models folder)
import glob
model_files = glob.glob(f'{log_dir}/models/*.pt')
if model_files:
    print(f"trained model already exists at {log_dir}/models/")
    print("skipping training (delete models folder to retrain)")
else:
    print(f"training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
    print(f"learning: connectivity W, latent vectors a_i, neuron-specific psi*(a_j, x_j)")
    print(f"models: {log_dir}/models/")
    print(f"training plots: {log_dir}/tmp_training")
    print(f"tensorboard: tensorboard --logdir {log_dir}/")
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
# Figures matching Supplementary Figure 13 from the paper.
#
# **Figure panels:**
#
# - Learned connectivity matrix
# - Comparison of learned vs true connectivity (expected: $R^2$=0.99, slope=0.99)
# - Learned latent vectors $\mathbf{a}_i$
# - Learned update functions $\phi^*(\mathbf{a}_i, x)$
# - Learned transfer functions $\psi^*(\mathbf{a}_j, x)$

# %%
#| echo: true
#| output: false
# STEP 3: GNN EVALUATION
print()
print("-" * 80)
print("STEP 3: GNN EVALUATION - Generating Supplementary Figure 13 panels")
print("-" * 80)
print(f"learned connectivity matrix")
print(f"W learned vs true (R^2, slope)")
print(f"latent vectors a_i (4 clusters)")
print(f"update functions phi*(a_i, x)")
print(f"transfer functions psi*(a_j, x)")
print(f"output: {log_dir}/results/")
print()
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)

# %% [markdown]
# ### Supplementary Figure 13: GNN Evaluation Results

# %%
#| fig-cap: "Learned connectivity."
load_and_display("./log/signal/signal_fig_supp_13/results/connectivity_learned.png")

# %%
#| fig-cap: "Comparison of learned and true connectivity (given $g_i$=10). Expected: $R^2$=0.99, slope=0.99."
load_and_display("./log/signal/signal_fig_supp_13/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Learned latent vectors $a_i$ of all neurons."
load_and_display("./log/signal/signal_fig_supp_13/results/embedding.png")

# %%
#| fig-cap: "Learned update functions $\\phi^*(a_i, x)$. The plot shows 1000 overlaid curves. Colors indicate true neuron types. True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_13/results/MLP0.png")

# %%
#| fig-cap: "Learned transfer functions $\\psi^*(a_j, x)$ ($\\gamma$=1,2,4,8), normalized to a maximum value of 1. True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_13/results/MLP1_corrected.png")
