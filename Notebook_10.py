# %% [raw]
# ---
# title: "Figure 3: External Inputs - 2048 neurons with Omega(t)"
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Neural Activity
#   - External Inputs
#   - GNN Training
# execute:
#   echo: false
# image: "graphs_data/signal/signal_fig_3/Fig/Fig_0_000000.png"
# ---

# %% [markdown]
# This script reproduces **Figure 3** from the paper:
# *"Graph neural networks uncover structure and function underlying the activity of neural assemblies"*
#
# **Simulation parameters:**
#
# - N_neurons: 2048 (1024 with external inputs + 1024 without)
# - N_types: 4 (parameterized by tau_i={0.5,1}, s_i={1,2}, gamma_j={1,2,4,8})
# - N_frames: 50,000
# - Connectivity: 100% (dense)
# - Noise: yes (sigma^2=1)
# - External inputs: yes - time-dependent scalar field Omega_i(t)
#
# The simulation follows:
#
# $$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \tanh(x_i) + g_i \Omega_i(t) \sum_j W_{ij} \left(\tanh\left(\frac{x_j}{\gamma_j}\right) - \theta_j x_j\right) + \eta_i(t)$$
#
# The GNN learns:
#
# $$\hat{\dot{x}}_i = \phi^*(\mathbf{a}_i, x_i) + \Omega_i^*(t) \sum_j W_{ij} \psi^*(\mathbf{a}_i, \mathbf{a}_j, x_j)$$
#
# The external input $\Omega_i(t)$ is a spatially-defined scalar field that modulates
# the connectivity for the first 1024 neurons. The remaining 1024 neurons have $\Omega_i = 1$.

# %%
#| output: false
import os
import shutil
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
print("Figure 3: 2048 neurons, 4 types, with external inputs Omega(t)")
print("=" * 80)

device = []
best_model = ''
config_file_ = 'signal_fig_3'

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
# Generate synthetic neural activity data using the PDE_N4 model.
# This creates the training dataset with 2048 neurons and external inputs.
#
# **Outputs:**
#
# - Figure 3a: External inputs Omega_i(t) - time-dependent scalar field
# - Figure 3b: Activity time series
# - Figure 3c: Sample of 10 time series

# %%
#| echo: true
#| output: false
# STEP 1: GENERATE
print()
print("-" * 80)
print("STEP 1: GENERATE - Simulating neural activity with external inputs (Fig 3a-c)")
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
    print(f"  External inputs: {config.simulation.n_input_neurons} neurons modulated by Omega(t)")
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
#| fig-cap: "Fig 3a-b: (Top) External input field $\\Omega_i(t)$ shown on a 32×32 grid (left, first 1024 neurons) and sunflower arrangement (right, remaining 1024 neurons). (Bottom) Neural activity $x_i$ at time $t=0$."
load_and_display(f"{graphs_dir}/Fig/Fig_0_000000.png")

# %%
#| fig-cap: "Fig 3c: Sample activity time series for 100 neurons over 10,000 time steps. Yellow dashed line shows the mean external input."
load_and_display(f"{graphs_dir}/activity.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train the GNN to learn connectivity W, latent embeddings a_i, functions phi/psi,
# and the external input field Omega*(x, y, t) using a coordinate-based MLP (SIREN).
#
# **Learning targets:**
#
# - Connectivity matrix W
# - Latent vectors a_i
# - Update function phi*(a_i, x)
# - Transfer function psi*(x)
# - External input field Omega*(x, y, t) via SIREN network

# %%
#| echo: true
#| output: false
# STEP 2: TRAIN
print()
print("-" * 80)
print("STEP 2: TRAIN - Training GNN to learn W, embeddings, phi, psi, and Omega*")
print("-" * 80)

# Check if trained model already exists (any .pt file in models folder)
import glob
model_files = glob.glob(f'{log_dir}/models/*.pt')
if model_files:
    print(f"  Trained model already exists at {log_dir}/models/")
    print("  Skipping training (delete models folder to retrain)")
else:
    print(f"  Training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
    print(f"  Learning: connectivity W, latent vectors a_i, functions phi*, psi*")
    print(f"  Learning: external input field Omega*(x, y, t) via SIREN network")
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
# ## Step 3: Generate Publication Figures
# Generate publication-quality figures matching Figure 3 from the paper.
#
# **Figure panels:**
#
# - Fig 3d: Comparison of learned vs true connectivity W_ij
# - Fig 3e: Comparison of learned vs true Omega_i(t) values
# - Fig 3f: True field Omega_i(t) at different time-points
# - Fig 3g: Learned field Omega*(t) at different time-points

# %%
#| echo: true
#| output: false
# STEP 3: PLOT
print()
print("-" * 80)
print("STEP 3: PLOT - Generating Figure 3 panels (d-g)")
print("-" * 80)
print(f"  Fig 3d: W learned vs true (R^2, slope)")
print(f"  Fig 3e: Omega learned vs true")
print(f"  Fig 3f: True field Omega_i(t) at different times")
print(f"  Fig 3g: Learned field Omega*(t) at different times")
print(f"  Output: {log_dir}/results/")
print()
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)

# %% [markdown]
# ## Output Files
# Rename output files to match Figure 3 panels.

# %%
#| echo: true
#| output: false
# Rename output files to match Figure 3 panels
print()
print("-" * 80)
print("Renaming output files to Figure 3 panels")
print("-" * 80)

results_dir = f'{log_dir}/results'
os.makedirs(results_dir, exist_ok=True)

# File mapping for simple copies
file_mapping = {
    f'{graphs_dir}/activity_sample.png': f'{results_dir}/Fig3d_activity_sample.png',
    f'{results_dir}/weights_comparison_corrected.png': f'{results_dir}/Fig3e_weights_comparison.png',
}

for src, dst in file_mapping.items():
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  {os.path.basename(dst)}")

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Copy Fig 3a-b from generated frame (plot_synaptic_frame_visual output)
fig_file = f'{graphs_dir}/Fig/Fig_0_000000.png'
if os.path.exists(fig_file):
    shutil.copy2(fig_file, f'{results_dir}/Fig3ab_external_input_activity.png')
    print(f"  Fig3ab_external_input_activity.png")

# Copy Fig 3c: Activity time series
if os.path.exists(f'{graphs_dir}/activity.png'):
    shutil.copy2(f'{graphs_dir}/activity.png', f'{results_dir}/Fig3c_activity_time_series.png')
    print(f"  Fig3c_activity_time_series.png")

# Generate Fig 3f: True field Omega_i(t) montage from field images
print("  Generating Fig3f_omega_field_true.png (5-frame montage)...")
field_dir = f'{results_dir}/field'
frame_indices = [0, 10000, 20000, 30000, 40000]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for idx, frame in enumerate(frame_indices):
    ax = axes[idx]
    # Find true field image for this frame
    true_field_files = sorted(glob.glob(f'{field_dir}/true_field*_{frame}.png'))
    if true_field_files:
        img = mpimg.imread(true_field_files[-1])
        ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f't={frame}', fontsize=12)
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{results_dir}/Fig3f_omega_field_true.png', dpi=150)
plt.close()
print(f"  Fig3f_omega_field_true.png")

# Generate Fig 3g: Learned field Omega*(t) montage from field images
print("  Generating Fig3g_omega_field_learned.png (5-frame montage)...")
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for idx, frame in enumerate(frame_indices):
    ax = axes[idx]
    # Find learned field image for this frame
    learned_field_files = sorted(glob.glob(f'{field_dir}/reconstructed_field_LR*_{frame}.png'))
    if learned_field_files:
        img = mpimg.imread(learned_field_files[-1])
        ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f't={frame}', fontsize=12)
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{results_dir}/Fig3g_omega_field_learned.png', dpi=150)
plt.close()
print(f"  Fig3g_omega_field_learned.png")

print()
print("=" * 80)
print("Figure 3 complete!")
print(f"Results saved to: {log_dir}/results/")
print("=" * 80)

# %% [markdown]
# ## Figure 3 Panels

# %%
#| fig-cap: "Fig 3d: Comparison of learned and true connectivity."
load_and_display(f"{log_dir}/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Fig 3e: Comparison of learned and true $\\Omega_i$ values."
# Find the latest all_comparison file
import glob
all_comp_files = sorted(glob.glob(f"{log_dir}/results/all_comparison*.png"))
if all_comp_files:
    load_and_display(all_comp_files[-1])

# %% [markdown]
# ### Fig 3f-g: True and Learned External Input Fields
# Showing $\Omega_i(t)$ at frames 0, 10000, 20000, 30000, 40000.

# %%
#| fig-cap: "True field $\\Omega_i$ at frame 0."
true_field_files = sorted(glob.glob(f"{log_dir}/results/field/true_field*_0.png"))
if true_field_files:
    load_and_display(true_field_files[-1])

# %%
#| fig-cap: "Learned field $\\Omega^*_i$ at frame 0."
learned_field_files = sorted(glob.glob(f"{log_dir}/results/field/reconstructed_field_LR*_0.png"))
if learned_field_files:
    load_and_display(learned_field_files[-1])

# %%
#| fig-cap: "True field $\\Omega_i$ at frame 10000."
true_field_files = sorted(glob.glob(f"{log_dir}/results/field/true_field*_10000.png"))
if true_field_files:
    load_and_display(true_field_files[-1])

# %%
#| fig-cap: "Learned field $\\Omega^*_i$ at frame 10000."
learned_field_files = sorted(glob.glob(f"{log_dir}/results/field/reconstructed_field_LR*_10000.png"))
if learned_field_files:
    load_and_display(learned_field_files[-1])

# %%
#| fig-cap: "True field $\\Omega_i$ at frame 20000."
true_field_files = sorted(glob.glob(f"{log_dir}/results/field/true_field*_20000.png"))
if true_field_files:
    load_and_display(true_field_files[-1])

# %%
#| fig-cap: "Learned field $\\Omega^*_i$ at frame 20000."
learned_field_files = sorted(glob.glob(f"{log_dir}/results/field/reconstructed_field_LR*_20000.png"))
if learned_field_files:
    load_and_display(learned_field_files[-1])

# %%
#| fig-cap: "True field $\\Omega_i$ at frame 30000."
true_field_files = sorted(glob.glob(f"{log_dir}/results/field/true_field*_30000.png"))
if true_field_files:
    load_and_display(true_field_files[-1])

# %%
#| fig-cap: "Learned field $\\Omega^*_i$ at frame 30000."
learned_field_files = sorted(glob.glob(f"{log_dir}/results/field/reconstructed_field_LR*_30000.png"))
if learned_field_files:
    load_and_display(learned_field_files[-1])

# %%
#| fig-cap: "True field $\\Omega_i$ at frame 40000."
true_field_files = sorted(glob.glob(f"{log_dir}/results/field/true_field*_40000.png"))
if true_field_files:
    load_and_display(true_field_files[-1])

# %%
#| fig-cap: "Learned field $\\Omega^*_i$ at frame 40000."
learned_field_files = sorted(glob.glob(f"{log_dir}/results/field/reconstructed_field_LR*_40000.png"))
if learned_field_files:
    load_and_display(learned_field_files[-1])
