# %% [raw]
# ---
# title: "Supplementary Figure 3: 1000 neurons with 4 types, training with fixed embedding"
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
# execute:
#   echo: false
# image: "log/signal/signal_fig_supp_3/results/embedding.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Supplementary Figure 3**.
# To assess the importance of learning latent neuron types, we trained a GNN with fixed embedding.
# Models that ignore the heterogeneity of neural populations are poor
# approximations of the underlying dynamics
#
# **Simulation parameters:**
#
# - N_neurons: 1000
# - N_types: 4 (parameterized by $\tau_i$={0.5,1} and $s_i$={1,2})
# - N_frames: 100,000
# - Connectivity: 100% (dense)
# - Noise: none
# - External inputs: none
# - Embedding: none (single type training)
#
# The simulation follows Equation 2 from the paper:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \psi(x_j)$$

# %%
#| output: false
import os
import warnings

from neural_gnn.config import NeuralGraphConfig
from neural_gnn.generators.graph_data_generator import data_generate
from neural_gnn.models.graph_trainer import data_train, data_test
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
print("Supplementary Figure 3: 1000 neurons, 4 types, dense connectivity, no embedding")
print("=" * 80)

device = []
best_model = ''
config_file_ = 'signal_fig_supp_3'

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
# Generate synthetic neural activity data using the PDE_N2 model.
# This creates the training dataset with 1000 neurons over 100,000 time points.
#
# **Outputs:**
#
# - Sample of 100 time series
# - True connectivity matrix $W_{ij}$

# %%
#| echo: true
#| output: false
# STEP 1: GENERATE
print()
print("-" * 80)
print("STEP 1: GENERATE - Simulating neural activity")
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
#| fig-cap: "Sample of 100 time series taken from the activity data."

load_and_display(f"./graphs_data/signal/signal_fig_supp_3/activity.png")

# %%
#| fig-cap: "True connectivity $W_{ij}$. The inset shows 20×20 weights."

load_and_display("./graphs_data/signal/signal_fig_supp_3/connectivity_matrix.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN to learn connectivity W and functions phi/psi (without latent embeddings).
# The GNN learns to predict dx/dt from the observed activity x.
#
# **Learning targets:**
#
# - Connectivity matrix $W$
# - Update function $\phi^*(x)$
# - Transfer function $\psi^*(x)$

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN to learn W, phi, psi (no embeddings)")
print("-" * 80)

# Check if trained model already exists (any .pt file in models folder)
import glob
model_files = glob.glob(f'{log_dir}/models/*.pt')
if model_files:
    print(f"trained model already exists at {log_dir}/models/")
    print("skipping training (delete models folder to retrain)")
else:
    print(f"training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
    print(f"learning: connectivity W, functions phi* and psi* (no embeddings)")
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
# Figures matching Supplementary Figure 3 from the paper.
#
# **Figure panels:**
#
# - Learned connectivity matrix
# - Comparison of learned vs true connectivity
# - Learned update functions $\phi^*(x)$
# - Learned transfer function $\psi^*(x)$

# %%
#| echo: true
#| output: false
# STEP 3: GNN EVALUATION
print()
print("-" * 80)
print("STEP 3: GNN EVALUATION - Generating Supplementary Figure 3 panels")
print("-" * 80)
print(f"learned connectivity matrix")
print(f"W learned vs true (R^2, slope)")
print(f"update functions phi*(x)")
print(f"transfer function psi*(x)")
print(f"output: {log_dir}/results/")
print()
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)

# %% [markdown]
# ### Supplementary Figure 3: GNN Evaluation Results

# %%
#| fig-cap: "Learned connectivity."
load_and_display("./log/signal/signal_fig_supp_3/results/connectivity_learned.png")

# %%
#| fig-cap: "Comparison of learned and true connectivity (given $g_i$=10)."
load_and_display("./log/signal/signal_fig_supp_3/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Learned update functions $\\phi^*(x)$. True function is overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_3/results/MLP0.png")

# %%
#| fig-cap: "Learned transfer function $\\psi^*(x)$, normalized to a maximum value of 1. True function is overlaid in light gray."
load_and_display("./log/signal/signal_fig_supp_3/results/MLP1_corrected.png")

# %% [markdown]
# ## Step 4: Test Model
# Test the trained GNN model. Evaluates prediction accuracy and performs rollout inference.

# %%
#| echo: true
#| output: false
# STEP 4: TEST
print()
print("-" * 80)
print("STEP 4: TEST - Evaluating trained model")
print("-" * 80)
print(f"testing prediction accuracy and rollout inference")
print(f"output: {log_dir}/results/")
print()
config.simulation.noise_model_level = 0.0

data_test(
    config=config,
    visualize=False,
    style="color name continuous_slice",
    verbose=False,
    best_model='best',
    run=0,
    test_mode="",
    sample_embedding=False,
    step=10,
    n_rollout_frames=1000,
    device=device,
    particle_of_interest=0,
    new_params=None,
)

# %% [markdown]
# ### Rollout Results
# Display the rollout comparison figures showing:
# - Left panel: activity traces (ground truth gray, learned colored)
# - Right panel: scatter plot of true vs learned $x_i$ with $R^2$ and slope

# %%
#| fig-cap: "Rollout comparison up to time-point 400."
load_and_display(f"{log_dir}/results/Fig_0_000039.png")

# %%
#| fig-cap: "Rollout comparison up to time-point 800."
load_and_display(f"{log_dir}/results/Fig_0_000079.png")
