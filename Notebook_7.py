# %% [raw]
# ---
# title: "Supplementary Figure 12: many types - 32 neuron types"
# author: Cédric Allier, MichaelInnerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
#   - Many Types
# execute:
#   echo: false
# image: "graphs_data/signal/signal_fig_supp_12/activity.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Supplementary Figure 12**.
# Test with 32 different neuron types (update functions).
#
# **Simulation parameters:**
#
# - N_neurons: 1000
# - N_types: 32 parameterized by $s_i$={1,2,3,4,5,6,7,8} and $\tau_i$={0.25,0.5,0.75,1.0}
# - N_frames: 100,000
# - Connectivity: 100% (dense)
# - Connectivity weights: random, Lorentz distribution
# - Noise: none
# - External inputs: none
#
# The simulation follows Equation 2 from the paper:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \psi(x_j)$$
#
# Classification accuracy expected: 0.99

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
print("Supplementary Figure 12: 1000 neurons, 32 types, dense connectivity")
print("=" * 80)

device = []
best_model = ''
config_file_ = 'signal_fig_supp_12'

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
# Generate synthetic neural activity data using the PDE_N2 model with 32 neuron types.
# This tests the GNN's ability to learn many distinct update functions.
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
print("STEP 1: GENERATE - Simulating neural activity (32 neuron types)")
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
#| fig-cap: "Sample time series taken from the activity data (32 neuron types)."
load_and_display(f"./graphs_data/signal/signal_fig_supp_12/activity.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$. The inset shows 20×20 weights."
load_and_display("./graphs_data/signal/signal_fig_supp_12/connectivity_matrix.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN to learn connectivity W, latent embeddings $a_i$, and functions $\phi^*, \psi^*$.
# The GNN must learn to distinguish 32 different update functions.
#
# The GNN optimizes the update rule (Equation 3 from the paper):
#
# $$\hat{\dot{x}}_i = \phi^*(\mathbf{a}_i, x_i) + \sum_j W_{ij} \psi^*(x_j)$$

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN to learn 32 neuron types")
print("-" * 80)

# Check if trained model already exists
model_file = f'{log_dir}/models/best_model_with_0_graphs_0_0.pt'
if os.path.exists(model_file):
    print(f"  Trained model already exists at {log_dir}/models/")
    print("  Skipping training (delete models folder to retrain)")
else:
    print(f"  Training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
    print(f"  Learning: connectivity W, latent vectors a_i for 32 types, functions phi* and psi*")
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
# Figures matching Supplementary Figure 12 from the paper.
#
# **Figure panels:**
#
# - Learned connectivity matrix
# - Comparison of learned vs true connectivity
# - Learned latent vectors $\mathbf{a}_i$ (32 clusters expected)
# - Learned update functions $\phi^*(\mathbf{a}_i, x)$ (32 distinct functions)
# - Learned transfer function $\psi^*(x)$

# %%
#| echo: true
#| output: false
# STEP 3: GNN EVALUATION
print()
print("-" * 80)
print("STEP 3: GNN EVALUATION - Generating Supplementary Figure 12 panels")
print("-" * 80)
print(f"  Learned connectivity matrix")
print(f"  W learned vs true (R^2, slope)")
print(f"  Latent vectors a_i (32 clusters)")
print(f"  Update functions phi*(a_i, x) - 32 distinct functions")
print(f"  Transfer function psi*(x)")
print(f"  Output: {log_dir}/results/")
print()
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)

# %% [markdown]
# ### Supplementary Figure 12: GNN Evaluation Results

# %%
#| fig-cap: "Learned connectivity."
load_and_display("./log/signal/signal_fig_supp_12/results/connectivity_learned.png")

# %%
#| fig-cap: "Comparison of learned and true connectivity (given $g_i$=10)."
load_and_display("./log/signal/signal_fig_supp_12/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Learned latent vectors $a_i$ of all neurons. 32 clusters expected (one per neuron type)."
load_and_display("./log/signal/signal_fig_supp_12/results/embedding.png")

# %%
#| fig-cap: "Learned update functions $\\phi^*(a_i, x)$. The plot shows 1000 overlaid curves representing 32 distinct update functions. Colors indicate true neuron types. True functions are overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_12/results/MLP0.png")

# %%
#| fig-cap: "Learned transfer function $\\psi^*(x)$, normalized to a maximum value of 1. True function is overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_12/results/MLP1_corrected.png")
