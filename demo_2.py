"""
Demo 2 - Figure 3 from the paper:
"Graph neural networks uncover structure and function underlying the activity of neural assemblies"

This demo reproduces Figure 3: 2048 densely connected neurons with external inputs.
    - N_neurons: 2048 (1024 with external inputs + 1024 without)
    - N_types: 4 (parameterized by tau_i={0.5,1}, s_i={1,2}, gamma_j={1,2,4,8})
    - N_frames: 50,000
    - Connectivity: 100% (dense)
    - Noise: yes (sigma^2=1)
    - External inputs: yes - time-dependent scalar field Omega_i(t)

The simulation follows Equation 2 from the paper with external inputs:
    dx_i/dt = -x_i/tau_i + s_i*tanh(x_i) + g_i * Omega_i(t) * sum_j(W_ij * psi(x_j/gamma_j))

The external input Omega_i(t) is a spatially-defined scalar field that modulates
the connectivity matrix W_ij for the first 1024 neurons. The remaining 1024 neurons
have Omega_i = 1 (no modulation).

Usage:
    python demo_2.py                                    # run full pipeline
    python demo_2.py -o generate signal_demo_2          # generate data only
    python demo_2.py -o train signal_demo_2             # train only
    python demo_2.py -o test signal_demo_2              # test only
    python demo_2.py -o plot signal_demo_2              # plot only
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import shutil

from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test
from NeuralGraph.utils import set_device, add_pre_folder
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="NeuralGraph Demo 2 - Figure 3 data results")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )

    print()
    print("=" * 80)
    print("Demo 2 - Figure 3: 2048 neurons, 4 types, with external inputs Omega(t)")
    print("=" * 80)

    device = []
    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:
        best_model = ''
        task = 'train_plot'  # generate_train
        config_list = ['signal_demo_2']

    for config_file_ in config_list:
        print()
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)

        # load config
        config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.config_file = config_file
        config.dataset = config_file  # e.g., 'signal/signal_demo_2'

        if device == []:
            device = set_device(config.training.device)

        log_dir = f'./log/{config_file}'
        graphs_dir = f'./graphs_data/{config_file}'

        if "generate" in task:
            # Generate synthetic neural activity data using the PDE_N4 model
            # This creates the training dataset with 2048 neurons and external inputs
            #
            # Figure 3a: External inputs Omega_i(t) - time-dependent scalar field
            # Figure 3b: Activity time series
            # Figure 3c: Sample of 10 time series
            #
            # Output folder: ./graphs_data/signal/signal_demo_2/
            #   - saved graph data (.pt files)
            #   - activity.png: activity time series visualization
            #   - external_input.png: Omega_i(t) field visualization
            print()
            print("-" * 80)
            print("STEP 1: GENERATE - Simulating neural activity with external inputs (Fig 3a-c)")
            print("-" * 80)
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

        if "train" in task:
            # Train the GNN to learn connectivity W, latent embeddings a_i, functions phi/psi,
            # and the external input field Omega*(x, y, t) using a coordinate-based MLP (SIREN)
            #
            # Output folder: ./log/signal/signal_demo_2/
            #   - models/: saved model checkpoints (.pt files)
            #   - tmp_training/: training curves and intermediate results
            print()
            print("-" * 80)
            print("STEP 2: TRAIN - Training GNN to learn W, embeddings, phi, psi, and Omega*")
            print("-" * 80)
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

        if 'plot' in task:
            # Generate publication-quality figures matching Figure 3 from the paper
            #
            # Figure 3d: Comparison of learned vs true connectivity W_ij
            # Figure 3e: Comparison of learned vs true Omega_i(t) values
            # Figure 3f: True field Omega_i(t) at different time-points
            # Figure 3g: Learned field Omega*(t) at different time-points
            #
            # Output folder: ./log/signal/signal_demo_2/results/
            #   - connectivity_comparison.png: W learned vs true (Fig 3d)
            #   - omega_comparison.png: Omega learned vs true (Fig 3e)
            #   - omega_field_true.png: True Omega(t) field (Fig 3f)
            #   - omega_field_learned.png: Learned Omega*(t) field (Fig 3g)
            print()
            print("-" * 80)
            print("STEP 4: PLOT - Generating Figure 3 panels (d-g)")
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

        # Rename output files to match Figure 3 panels
        print()
        print("-" * 80)
        print("Renaming output files to Figure 3 panels")
        print("-" * 80)

        results_dir = f'{log_dir}/results'
        fig_dir = f'{results_dir}/Fig3'
        os.makedirs(fig_dir, exist_ok=True)

        # File mapping: original name -> Figure 3 panel name
        file_mapping = {
            # From graphs_data (generation step)
            f'{graphs_dir}/activity_gt.png': f'{fig_dir}/Fig3b_activity_time_series.png',
            f'{graphs_dir}/activity_gt.pdf': f'{fig_dir}/Fig3b_activity_time_series.pdf',
            f'{graphs_dir}/external_input.png': f'{fig_dir}/Fig3a_external_input_omega.png',
            # From results (plot step)
            f'{results_dir}/activity_gt.pdf': f'{fig_dir}/Fig3bc_activity.pdf',
            f'{results_dir}/weights_comparison_corrected.png': f'{fig_dir}/Fig3d_weights_comparison.png',
            f'{results_dir}/omega_comparison.png': f'{fig_dir}/Fig3e_omega_comparison.png',
            f'{results_dir}/omega_field_true.png': f'{fig_dir}/Fig3f_omega_field_true.png',
            f'{results_dir}/omega_field_learned.png': f'{fig_dir}/Fig3g_omega_field_learned.png',
            f'{results_dir}/field_true.png': f'{fig_dir}/Fig3f_omega_field_true.png',
            f'{results_dir}/field_learned.png': f'{fig_dir}/Fig3g_omega_field_learned.png',
        }

        for src, dst in file_mapping.items():
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  {os.path.basename(dst)}")

        print()
        print("=" * 80)
        print("Demo 2 complete!")
        print(f"Results saved to: {log_dir}/results/")
        print(f"Figure 3 panels: {fig_dir}/")
        print("=" * 80)
