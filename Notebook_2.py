# %% [raw]
# ---
# title: "Figure 2: Baseline - 1000 neurons with 4 types"
# author: Cédric Allier, Michael Bhaskara, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - Simulation
#   - GNN Training
# execute:
#   echo: false
# image: "graphs_data/signal/signal_fig_2/activity.png"
# ---

# %% [markdown]
# This script reproduces the panels of paper's **Figure 2** and other supplementary panels related to the same dataset.
#
# **Simulation parameters:**
#
# - N_neurons: 1000
# - N_types: 4 (parameterized by tau_i={0.5,1} and s_i={1,2})P
# - N_frames: 100,000
# - Connectivity: 100% (dense)
# - Noise: none
# - External inputs: none
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
from GNN_PlotFigure import data_plot, create_training_montage

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Configuration and Setup

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("Figure supp 3 and 4: 1000 neurons, 4 types, dense connectivity, no embedding")
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
# This creates the training dataset with 1000 neurons and 100,000 time points.
#
# **Outputs:**
#
# - Figure 2b: Sample of 10 time series
# - Figure 2c: True connectivity matrix W_ij

# %%
#| echo: true
#| output: false
# STEP 1: GENERATE
print()
print("-" * 80)
print("STEP 1: GENERATE - Simulating neural activity (Fig 2a-c)")
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
#| fig-cap: "Fig 2b: Sample of 10 time series taken from the activity data."

load_and_display(f"./graphs_data/signal/signal_fig_2/activity.png")

# %%
#| fig-cap: "Fig 2c: True connectivity W_ij. The inset shows 20×20 weights."

load_and_display("./graphs_data/signal/signal_fig_2/connectivity_matrix.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN to learn connectivity W, latent embeddings a_i, and functions phi/psi.
# The GNN learns to predict dx/dt from the observed activity x.
#
# **Learning targets:**
#
# - Connectivity matrix W
# - Latent vectors a_i
# - Update function phi*(a_i, x)
# - Transfer function psi*(x)

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN to learn W, embeddings, phi, psi")
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
# Figures matching Figure 2 from the paper.
#
# **Figure panels:**
#
# - Fig 2d: Learned connectivity matrix
# - Fig 2e: Comparison of learned vs true connectivity 
# - Fig 2f: Learned latent vectors a_i 
# - Fig 2g: Learned update functions phi*(a_i, x) 
# - Fig 2h: Learned transfer function psi*(x) 

# %%
#| echo: true
#| output: false
# STEP 3: GNN EVALUATION
print()
print("-" * 80)
print("STEP 3: GNN EVALUATION - Generating Figure 2 panels (d-h)")
print("-" * 80)
print(f"  Fig 2d: Learned connectivity matrix")
print(f"  Fig 2e: W learned vs true (R^2, slope)")
print(f"  Fig 2f: Latent vectors a_i (4 clusters)")
print(f"  Fig 2g: Update functions phi*(a_i, x)")
print(f"  Fig 2h: Transfer function psi*(x)")
print(f"  Output: {log_dir}/results/")
print()
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)

# %% [markdown]
# ### Figures 2d-2h: GNN Evaluation Results

# %%
#| fig-cap: "Fig 2d: Learned connectivity."
load_and_display("./log/signal/signal_fig_2/results/connectivity_learned.png")

# %%
#| fig-cap: "Fig 2e: Comparison of learned and true connectivity (given g_i=10)."
load_and_display("./log/signal/signal_fig_2/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Fig 2f: Learned latent vectors a_i of all neurons."
load_and_display("./log/signal/signal_fig_2/results/embedding.png")

# %%
#| fig-cap: "Fig 2g: Learned update functions φ*(a_i, x). The plot shows 1000 overlaid curves, one for each vector a_i."
load_and_display("./log/signal/signal_fig_2/results/MLP0.png")

# %%
#| fig-cap: "Fig 2h: Learned transfer function ψ*(x), normalized to a maximum value of 1. Colors indicate true neuron types. True function is overlaid in light gray."
load_and_display("./log/signal/signal_fig_2/results/MLP1_corrected.png")

# %% [markdown]
# ## Step 4: GNN Training Visualization
# Generate training progression figures showing how the GNN learns across epochs.
#
# **Visualizations:**
#
# - Row a: Latent embeddings a_i evolution 
# - Row b: Update functions phi*(a_i, x) 
# - Row c: Transfer function psi*(x)
# - Row d: Connectivity matrix W 
# - Row e: W learned vs true scatter plot

# %%
#| echo: true
#| output: false
# STEP 4: GNN TRAINING VISUALIZATION
print()
print("-" * 80)
print("STEP 4: GNN TRAINING - Generating training progression figures")
print("-" * 80)
print(f"  Generating plots for all training epochs")
print(f"  Output: {log_dir}/results/all/")
print()
data_plot(config=config, config_file=config_file, epoch_list=['all'], style='color', extended='plots', device=device, apply_weight_correction=True)

# Create montage from individual epoch plots
print()
print("  Creating training montage (8 columns x 5 rows)...")
create_training_montage(config=config, n_cols=8)

# %%
#| fig-cap: "Supplementary Figure 1: Results plotted over 20 epochs. (a) Learned latent vectors a_i. (b) Learned update functions φ*(a,x). (c) Learned transfer function ψ*(x), normalized to max=1. (d) Learned connectivity W_ij. (e) Comparison of learned and true connectivity. Colors indicate true neuron types."

load_and_display("./log/signal/signal_fig_2/results/training_montage.png")

# %% [markdown]
# ## Step 5: Test Model
# Test the trained GNN model. Evaluates prediction accuracy and performs rollout inference.

# %%
#| echo: true
#| output: false
# STEP 5: TEST
print()
print("-" * 80)
print("STEP 5: TEST - Evaluating trained model")
print("-" * 80)
print(f"  Testing prediction accuracy and rollout inference")
print(f"  Output: {log_dir}/results/")
print()
config.training.noise_model_level = 0.0

data_test(
    config=config,
    visualize=True,
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
# Display the rollout comparison figure showing:
# - Left panels: activity traces (ground truth gray, learned colored)
# - Top right: scatter plot of true vs learned $x_i$ with $R^2$ and slope
# - Bottom right: $R^2$ over time

# %%
# Display rollout comparison figure (last frame from rollout)
dataset_name_ = config.dataset.split('/')[-1]
rollout_fig_path = f"{log_dir}/results/{dataset_name_}.png"
load_and_display(rollout_fig_path)
