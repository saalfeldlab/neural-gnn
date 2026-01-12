"""
NGP_trainer.py - Sequential Pipeline for Neural Space-Time Model Training

This script implements a 5-stage pipeline for training instant-NGP based
Neural Space-Time Model (NSTM) for activity decomposition:

Stage 1: Train SIREN on discrete neuron activities (t -> [activity_1, ..., activity_n])
Stage 2: Generate activity images from matplotlib scatter rendering
Stage 3: Generate warped motion frames (activity × boat + sinusoidal warp)
Stage 4: Train NeuralRenderer to match matplotlib scatter rendering
Stage 5: Train NSTM (deformation + fixed_scene) on warped frames
         - Three options for activity in Stage 5:
           Option 1: Use matplotlib activity (Stage 2) - set use_neural_renderer_for_nstm=False
           Option 2: Use fixed NeuralRenderer (Stage 4) - set use_neural_renderer_for_nstm=True, neural_renderer_learnable=False
           Option 3: Use learnable NeuralRenderer (Stage 4, LR=1e-6) - set use_neural_renderer_for_nstm=True, neural_renderer_learnable=True

Each stage produces visualizations and MP4 videos for validation.
"""

import os
import numpy as np
import torch
import cv2
import subprocess
import shutil
from tqdm import tqdm, trange
from tifffile import imread, imwrite
import matplotlib
import matplotlib.pyplot as plt
from NeuralGraph.utils import to_numpy
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity as ssim

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None
    print("Warning: tinycudann not installed. Falling back to slow mode.")

# Functions train_siren, train_nstm, and apply_sinusoidal_warp are defined below


def compute_gt_deformation_field(frame_idx, num_frames, res, motion_intensity=0.015):
    """Compute ground truth deformation field for visualization

    Returns:
        dx, dy: Deformation fields of shape (res, res) in normalized coordinates [0, 1]
    """
    # Create coordinate grids in normalized space [0, 1]
    y_grid, x_grid = np.meshgrid(
        np.linspace(0, 1, res),
        np.linspace(0, 1, res),
        indexing='ij'
    )

    # Create displacement fields with time-varying frequency
    t_norm = frame_idx / num_frames
    freq_x = 2 + t_norm * 2
    freq_y = 3 + t_norm * 1

    # Compute normalized deformation (same as in apply_sinusoidal_warp)
    dx = motion_intensity * np.sin(freq_x * np.pi * y_grid) * np.cos(freq_y * np.pi * x_grid)
    dy = motion_intensity * np.cos(freq_x * np.pi * y_grid) * np.sin(freq_y * np.pi * x_grid)

    return dx, dy


def create_network_config():
    """Create network configuration for NSTM"""
    return {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 8,
            "n_features_per_level": 2,
            "log2_hashmap_size": 16,
            "base_resolution": 16,
            "per_level_scale": 1.5
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 32,
            "n_hidden_layers": 2
        }
    }


def get_activity_at_coords(coords_2d, t_normalized, pretrained_activity_net, neuron_positions,
                           nnr_f_T_period, device, dot_size=32, image_size=512,
                           affine_scale=None, affine_bias=None):
    """Get activity values at arbitrary 2D coordinates using Gaussian splatting

    Uses the same Gaussian splatting method as render_activity_image to ensure consistency
    between ground truth rendering and NSTM training. Applies affine transform to map SIREN
    output range to activity image range.

    Args:
        coords_2d: (N, 2) tensor of 2D coordinates in [0, 1]
        t_normalized: scalar, normalized time value in [0, 1]
        pretrained_activity_net: SIREN network mapping t -> [activity_1, ..., activity_n]
        neuron_positions: (n_neurons, 2) tensor of neuron positions in [-0.5, 0.5]
        nnr_f_T_period: Temporal period for SIREN
        device: torch device
        dot_size: Diameter of Gaussian dots in pixels (default 32)
        image_size: Image resolution for pixel conversion (default 512)
        affine_scale: Learnable scale parameter for affine transform (default None)
        affine_bias: Learnable bias parameter for affine transform (default None)

    Returns:
        (N,) tensor of activity values at the queried coordinates
    """
    # Step 1: Query pretrained SIREN to get activities for all neurons at this time
    t_siren = t_normalized * (2 * np.pi / nnr_f_T_period)
    t_tensor = torch.tensor([[t_siren]], dtype=torch.float32, device=device)

    with torch.no_grad():
        neuron_activities = pretrained_activity_net(t_tensor)[0]  # (n_neurons,)

    # Apply affine transform to SIREN outputs: y = scale * x + bias
    # This maps SIREN range (e.g., [-20, 20]) to activity image range (e.g., [0, 120])
    if affine_scale is not None and affine_bias is not None:
        neuron_activities = affine_scale * neuron_activities + affine_bias

    # Step 2: Convert coordinates to pixel space for Gaussian splatting
    # coords_2d in [0, 1] -> pixel coords in [0, image_size-1]
    coords_pixels = coords_2d * (image_size - 1)  # (N, 2)

    # neuron_positions in [-0.5, 0.5] -> pixel coords in [0, image_size-1]
    neuron_positions_pixels = (neuron_positions + 0.5) * (image_size - 1)  # (n_neurons, 2)

    # Step 3: Gaussian splatting - compute Gaussian weights from each neuron to each query point
    # Gaussian sigma from dot size (diameter = 2.355 * sigma for Gaussian)
    sigma = dot_size / 2.355

    # Compute squared distances: (N, 1, 2) - (1, n_neurons, 2) = (N, n_neurons, 2)
    diffs = coords_pixels.unsqueeze(1) - neuron_positions_pixels.unsqueeze(0)
    dist_sq = torch.sum(diffs ** 2, dim=2)  # (N, n_neurons)

    # Gaussian kernel: exp(-dist^2 / (2*sigma^2))
    gaussian_weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # (N, n_neurons)

    # Step 4: Sum weighted activities (Gaussian splatting)
    # Each query point receives contribution from all neurons weighted by Gaussian
    splatted_activities = torch.sum(gaussian_weights * neuron_activities.unsqueeze(0), dim=1)  # (N,)

    return splatted_activities


def render_siren_activity_video(pretrained_activity_net, neuron_positions, affine_scale, affine_bias,
                                  nnr_f_T_period, n_frames, res, device, output_dir, fps=30):
    """Render a video of SIREN activity with learned affine transform using Gaussian splatting

    Args:
        pretrained_activity_net: SIREN network mapping t -> [activity_1, ..., activity_n]
        neuron_positions: (n_neurons, 2) tensor of neuron positions in [-0.5, 0.5]
        affine_scale: Learned scale parameter
        affine_bias: Learned bias parameter
        nnr_f_T_period: Temporal period for SIREN
        n_frames: Number of frames to render
        res: Image resolution (e.g., 512)
        device: torch device
        output_dir: Directory to save video
        fps: Frames per second for video (default 30)
    """
    # Create temporary directory for frames
    temp_dir = f"{output_dir}/siren_activity_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Create full 2D coordinate grid
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)  # (res*res, 2)

    print(f"rendering {n_frames} frames...")
    for frame_idx in tqdm(range(n_frames), ncols=100):
        t_normalized = frame_idx / (n_frames - 1)

        # Get activity at all pixel coordinates using learned affine transform
        with torch.no_grad():
            activity_img = get_activity_at_coords(
                coords_2d,
                t_normalized,
                pretrained_activity_net,
                neuron_positions,
                nnr_f_T_period,
                device,
                dot_size=32,
                image_size=res,
                affine_scale=affine_scale,
                affine_bias=affine_bias
            ).reshape(res, res).cpu().numpy()

        # Normalize to [0, 1] for visualization
        activity_min = activity_img.min()
        activity_max = activity_img.max()
        if activity_max > activity_min:
            activity_norm = (activity_img - activity_min) / (activity_max - activity_min)
        else:
            activity_norm = np.zeros_like(activity_img)

        # Convert to 8-bit RGB
        activity_uint8 = (np.clip(activity_norm, 0, 1) * 255).astype(np.uint8)
        activity_rgb = np.stack([activity_uint8, activity_uint8, activity_uint8], axis=2)

        # Add text overlay with frame info
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame {frame_idx}/{n_frames-1} | t={t_normalized:.3f}"
        cv2.putText(activity_rgb, text, (10, 30), font, 0.7, (255, 255, 255), 2)

        # Add scale/bias info
        text2 = f"scale={affine_scale.item():.2f}, bias={affine_bias.item():.1f}"
        cv2.putText(activity_rgb, text2, (10, 60), font, 0.6, (255, 255, 255), 2)

        # Save frame
        cv2.imwrite(f"{temp_dir}/frame_{frame_idx:06d}.png", activity_rgb)

    # Create video using ffmpeg
    video_path = f"{output_dir}/siren_activity_affine.mp4"
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%06d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 {video_path}"
    )

    subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

    # Clean up temporary frames
    shutil.rmtree(temp_dir)

    print(f"SIREN activity video saved to: {video_path}")


def apply_sinusoidal_warp(image, frame_idx, num_frames, motion_intensity=0.015):
    """Apply sinusoidal warping to an image, similar to pixel_NSTM.py"""

    h, w = image.shape

    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(
        np.linspace(0, 1, h),
        np.linspace(0, 1, w),
        indexing='ij'
    )

    # Create displacement fields with time-varying frequency
    t_norm = frame_idx / num_frames
    freq_x = 2 + t_norm * 2
    freq_y = 3 + t_norm * 1

    dx = motion_intensity * np.sin(freq_x * np.pi * y_grid) * np.cos(freq_y * np.pi * x_grid)
    dy = motion_intensity * np.cos(freq_x * np.pi * y_grid) * np.sin(freq_y * np.pi * x_grid)

    # Create source coordinates
    coords_y, coords_x = np.meshgrid(
        np.arange(h),
        np.arange(w),
        indexing='ij'
    )

    # Apply displacement
    sample_y = coords_y - dy * h
    sample_x = coords_x - dx * w

    # Ensure coordinates are within bounds
    sample_y = np.clip(sample_y, 0, h - 1)
    sample_x = np.clip(sample_x, 0, w - 1)

    # Warp the image
    warped = map_coordinates(image, [sample_y, sample_x], order=1, mode='reflect')

    return warped


def train_siren(x_list, device, output_dir, num_training_steps=20000,
                nnr_f_T_period=10, n_train_frames=10, n_neurons=100):
    """Pre-train SIREN network on discrete neuron time series (t -> [activity_1, ..., activity_n])

    Args:
        x_list: List of neuron data arrays (n_frames, n_neurons, features)
                x_list[frame][neuron, 6] = activity value
        device: torch device
        output_dir: Directory to save outputs
        num_training_steps: Number of training steps
        nnr_f_T_period: Period for temporal coordinate (t will be scaled by 2π/period)
        n_train_frames: Number of frames to train on
        n_neurons: Number of neurons (output dimension)

    Returns:
        siren_net: Pre-trained SIREN network mapping t -> [activity_1, ..., activity_n]
    """
    from NeuralGraph.models.Siren_Network import Siren

    print(f"pre-training SIREN time network on {n_neurons} neurons across {n_train_frames} frames...")

    # Extract activity data from all training frames
    all_t_coords = []
    all_activities = []

    for t_idx in range(n_train_frames):
        frame_data = x_list[0][t_idx]
        activities = frame_data[:, 6] # (n_neurons,) - activity values for all neurons

        # Normalize time coordinate
        t_value = t_idx / (n_train_frames - 1) if n_train_frames > 1 else 0.0
        t_normalized = t_value * (2 * np.pi / nnr_f_T_period)

        all_t_coords.append(t_normalized)
        all_activities.append(activities)

    # Print statistics
    all_activities_array = np.array(all_activities)
    print(f"n_neurons: {n_neurons}")
    print(f"n_frames: {n_train_frames}")
    print(f"time range: [0, {all_t_coords[-1]:.4f}] (normalized)")
    print(f"activity range: [{all_activities_array.min():.4f}, {all_activities_array.max():.4f}]")

    # Convert to tensors
    t_coords_tensor = torch.tensor(all_t_coords, dtype=torch.float32, device=device).reshape(-1, 1)  # (n_frames, 1)
    activities_tensor = torch.tensor(all_activities_array, dtype=torch.float32, device=device)  # (n_frames, n_neurons)

    # Create SIREN network: input = t (1D), output = n_neurons activities
    siren_net = Siren(
        in_features=1,  # Just time
        out_features=n_neurons,  # One output per neuron
        hidden_features=256,
        hidden_layers=4,
        outermost_linear=True,
        first_omega_0=60,
        hidden_omega_0=30
    ).to(device)

    optimizer = torch.optim.Adam(siren_net.parameters(), lr=5e-5)

    # Training loop
    loss_history = []
    siren_net.train()

    pbar = tqdm(range(num_training_steps), desc="training SIREN time network", ncols=100)
    for step in pbar:
        optimizer.zero_grad()

        # Forward pass: input is time, output is all neuron activities
        predicted_activities = siren_net(t_coords_tensor)  # (n_frames, n_neurons)

        # Loss: MSE between predicted and true activities
        loss = 100 * torch.nn.functional.mse_loss(predicted_activities, activities_tensor)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    siren_net.eval()
    with torch.no_grad():
        # Plot loss history
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.title('SIREN time pre-training loss', fontsize=12)
        plt.xlabel('step')
        plt.ylabel('MSE loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/pretrain_time_loss.png', dpi=150)
        plt.close()

        print(f"final loss: {loss_history[-1]:.6f}")

        # Test: scatter plot of all predictions vs ground truth
        predicted_all = siren_net(t_coords_tensor).cpu().numpy()  # (n_frames, n_neurons)
        ground_truth_all = activities_tensor.cpu().numpy()  # (n_frames, n_neurons)

        # Flatten to get all (frame, neuron) pairs
        pred_flat = predicted_all.flatten()
        gt_flat = ground_truth_all.flatten()

        # Calculate R²
        r2 = 1 - (np.sum((gt_flat - pred_flat) ** 2) / np.sum((gt_flat - gt_flat.mean()) ** 2))

        # Plot scatter comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.scatter(gt_flat, pred_flat, c='white', s=50, alpha=0.6, edgecolors='green', linewidths=1, label='SIREN vs GT')

        # Add diagonal reference line
        min_val = min(gt_flat.min(), pred_flat.min())
        max_val = max(gt_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.5, linewidth=2, label='Perfect Fit')

        # Add R² text
        ax.text(0.02, 0.98, f'R²={r2:.4f}', transform=ax.transAxes,
               fontsize=14, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
               color='white' if r2 > 0.9 else ('orange' if r2 > 0.7 else 'red'))

        ax.set_xlabel('Ground Truth Activity', fontsize=12)
        ax.set_ylabel('SIREN Predicted Activity', fontsize=12)
        ax.set_title(f'SIREN Time Model: {n_neurons} neurons × {n_train_frames} frames', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/siren_time_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"R² = {r2:.4f} ({n_neurons * n_train_frames} data points)")

    siren_net.train()
    return siren_net


def train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir, num_training_steps=3000,
               siren_config=None, pretrained_activity_net=None, x_list=None,
               use_siren=True, siren_lr=1e-4, nstm_lr=5e-4, siren_loss_weight=1.0,
               neural_renderer=None, neural_renderer_learnable=False, neural_renderer_lr=1e-6,
               neuron_positions=None, siren_net=None):
    """Train Neural Space-Time Model with fixed_scene + activity decomposition

    Args:
        siren_config: Dict with keys: hidden_dim_nnr_f, n_layers_nnr_f, omega_f,
                      nnr_f_xy_period, nnr_f_T_period, outermost_linear_nnr_f
        x_list: List of neuron data arrays for SIREN supervision (n_frames, n_neurons, features)
        use_siren: Boolean to use SIREN network for activity (True) or grid_sample (False)
        siren_lr: Learning rate for SIREN network
        nstm_lr: Learning rate for deformation and fixed_scene networks
        siren_loss_weight: Weight for SIREN supervision loss on discrete neurons
        neural_renderer: NeuralRenderer network from Stage 4 (optional)
        neural_renderer_learnable: If True, make neural_renderer learnable during training
        neural_renderer_lr: Learning rate for neural_renderer if learnable
        neuron_positions: Neuron positions (N, 2) needed for neural_renderer
        siren_net: SIREN network needed for neural_renderer
    """
    # Load motion frames
    print("loading motion frames...")
    motion_images = []
    for i in range(n_frames):
        img = imread(f"{motion_frames_dir}/frame_{i:06d}.tif")
        motion_images.append(img)

    # Load activity images (only if not using pretrained SIREN)
    activity_images = []
    if pretrained_activity_net is None:
        print("loading original activity images...")
        for i in range(n_frames):
            img = imread(f"{activity_dir}/frame_{i:06d}.tif")
            activity_images.append(img)
        print(f"loaded {n_frames} activity images")
    else:
        print("using pretrained SIREN for activity (no activity images loaded)")

    # Compute normalization statistics from motion images
    all_pixels = np.concatenate([img.flatten() for img in motion_images])
    data_min = all_pixels.min()
    data_max = all_pixels.max()
    print(f"input data range: [{data_min:.2f}, {data_max:.2f}]")

    # Normalize motion images to [0, 1] and convert to tensors
    motion_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float32).to(device) for img in motion_images]

    # Normalize activity images to [0, 1] and convert to tensors (if loaded)
    if activity_images:
        # When using neural renderer, compute separate normalization for activity images
        if neural_renderer is not None:
            activity_pixels = np.concatenate([img.flatten() for img in activity_images])
            activity_min = activity_pixels.min()
            activity_max = activity_pixels.max()
            print(f"activity data range: [{activity_min:.2f}, {activity_max:.2f}]")
            activity_tensors = [torch.tensor((img - activity_min) / (activity_max - activity_min), dtype=torch.float32).to(device) for img in activity_images]
        else:
            # Use motion normalization for consistency when not using neural renderer
            activity_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float32).to(device) for img in activity_images]
        print(f"loaded {n_frames} frames into gpu memory")
    else:
        activity_tensors = None
        activity_min = data_min
        activity_max = data_max
        print(f"loaded {n_frames} motion frames into gpu memory")

    # Create networks
    print("creating networks...")
    config = create_network_config()

    # Deformation network: (x, y, t) -> (δx, δy)
    deformation_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,  # x, y, t
        n_output_dims=2,  # δx, δy
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Fixed scene network: (x, y) -> raw mask value
    fixed_scene_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,  # x, y
        n_output_dims=1,  # mask value
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Note: Activity is handled by pretrained_activity_net via get_activity_at_coords()
    # No separate activity network needed in train_nstm

    # Create coordinate grid (use float16 for tinycudann, float32 for coords)
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Convert to float16 for tinycudann networks
    coords_2d_f16 = coords_2d.to(torch.float16)

    # Create learnable affine transform parameters for SIREN activity mapping
    # Maps SIREN output range (e.g., [-20, 20]) to activity image range (e.g., [0, 120])
    # Initialize with reasonable defaults: scale=3.0, bias=60.0 for [-20,20] -> [0,120]
    affine_scale = torch.nn.Parameter(torch.tensor(3.0, device=device, dtype=torch.float32))
    affine_bias = torch.nn.Parameter(torch.tensor(60.0, device=device, dtype=torch.float32))

    # Create optimizer for NSTM networks only (deformation + fixed_scene + affine params)
    nstm_params = (list(deformation_net.parameters()) +
                   list(fixed_scene_net.parameters()) +
                   [affine_scale, affine_bias])
    optimizer_nstm = torch.optim.Adam(nstm_params, lr=nstm_lr)

    # Create separate optimizer for neural_renderer if learnable
    optimizer_neural_renderer = None
    if neural_renderer is not None and neural_renderer_learnable:
        optimizer_neural_renderer = torch.optim.Adam(neural_renderer.parameters(), lr=neural_renderer_lr)
        print(f"neural renderer is learnable (lr={neural_renderer_lr})")

    # Convert neuron positions to torch tensor once if using neural_renderer
    neuron_positions_torch = None
    if neural_renderer is not None and neuron_positions is not None:
        neuron_positions_torch = torch.tensor(neuron_positions, dtype=torch.float32, device=device)
        neuron_positions_torch = (neuron_positions_torch + 0.5)  # [-0.5, 0.5] -> [0, 1]

    # Extract neuron positions for activity sampling (if using pretrained SIREN)
    neuron_positions = None
    if pretrained_activity_net is not None and x_list is not None:
        print("extracting neuron positions for activity sampling...")
        frame_data = x_list[0][0]
        neuron_positions = frame_data[:, 1:3].astype(np.float32)  # (n_neurons, 2) in [-0.5, 0.5]
        neuron_positions = torch.tensor(neuron_positions, dtype=torch.float32, device=device)
        print(f"using {neuron_positions.shape[0]} neuron positions")

    # Two-epoch training schedule (like Stage 4)
    print(f"training for {num_training_steps * 2} steps (2 epochs × {num_training_steps} steps)...")
    loss_history = []
    regularization_history = {'deformation': []}
    batch_size = min(16384, res*res)

    for epoch in range(1, 3):
        # Set learning rate for this epoch
        if epoch == 1:
            lr_nstm = nstm_lr
            lr_neural = neural_renderer_lr
        else:
            lr_nstm = nstm_lr / 10.0
            lr_neural = neural_renderer_lr / 10.0

        # Update optimizer learning rates
        for param_group in optimizer_nstm.param_groups:
            param_group['lr'] = lr_nstm
        if optimizer_neural_renderer is not None:
            for param_group in optimizer_neural_renderer.param_groups:
                param_group['lr'] = lr_neural

        print(f"epoch {epoch}: lr_nstm={lr_nstm:.6f}" +
              (f", lr_neural={lr_neural:.6f}" if optimizer_neural_renderer is not None else ""))

        pbar = trange(num_training_steps, desc=f"epoch {epoch}", ncols=150)
        for step in pbar:
            # Select random frame
            t_idx = np.random.randint(0, n_frames)
            t_normalized = t_idx / (n_frames - 1)

            # Select random batch of pixels
            indices = torch.randperm(res*res, device=device)[:batch_size]
            batch_coords = coords_2d[indices]

            # Target values from warped motion frames
            target = motion_tensors[t_idx].reshape(-1, 1)[indices]

            # Create 3D coordinates for deformation network (convert to float16 for tcnn)
            t_tensor = torch.full_like(batch_coords[:, 0:1], t_normalized)
            coords_3d = torch.cat([batch_coords, t_tensor], dim=1).to(torch.float16)

            # Forward pass - deformation network (backward warp)
            deformation = deformation_net(coords_3d).to(torch.float32)

            # Compute source coordinates (backward warp)
            source_coords = batch_coords - deformation  # Note: MINUS for backward warp
            source_coords = torch.clamp(source_coords, 0, 1)

            # Sample fixed scene mask at source coordinates (convert to float16 for tcnn)
            fixed_scene_mask = torch.sigmoid(fixed_scene_net(source_coords.to(torch.float16))).to(torch.float32)

            # Sample activity at source coordinates
            if neural_renderer is not None:
                # Use NeuralRenderer to query activity directly at source coordinates
                # Get SIREN activities for this frame (time t is fixed for this iteration)
                t_siren = torch.tensor([t_normalized * (2 * np.pi / siren_config['nnr_f_T_period'])],
                                       dtype=torch.float32, device=device).reshape(1, 1)
                with torch.set_grad_enabled(neural_renderer_learnable):
                    activities_siren = siren_net(t_siren).squeeze()  # (n_neurons,) in [0, 1]

                    # Query neural renderer ONLY at batch source coordinates (efficient!)
                    # source_coords: (batch_size, 2) in [0, 1]
                    # Neural renderer outputs activity values in [activity_min, activity_max] range
                    sampled_activity_unnormalized = neural_renderer(
                        neuron_positions_torch,  # (N, 2) neuron positions
                        activities_siren,         # (N,) activities in [0, 1]
                        source_coords            # (batch_size, 2) query points
                    )  # Returns: (batch_size,) in [activity_min, activity_max] range

                # CRITICAL: Normalize using same activity_min/activity_max as activity images!
                # This ensures neural renderer outputs are in same range as loaded activity images
                sampled_activity = ((sampled_activity_unnormalized - activity_min) / (activity_max - activity_min)).unsqueeze(1)  # (batch_size, 1)
            elif pretrained_activity_net is not None and neuron_positions is not None:
                # Use pretrained temporal SIREN to get activity at 2D coordinates
                sampled_activity = get_activity_at_coords(
                    source_coords,
                    t_normalized,
                    pretrained_activity_net,
                    neuron_positions,
                    siren_config['nnr_f_T_period'],
                    device,
                    affine_scale=affine_scale,
                    affine_bias=affine_bias
                ).unsqueeze(1)  # (batch_size, 1)
            else:
                # Fallback: use grid_sample on ground-truth activity images
                source_coords_normalized = source_coords * 2 - 1
                source_coords_grid = source_coords_normalized.reshape(1, 1, batch_size, 2)
                activity_frame = activity_tensors[t_idx].reshape(1, 1, res, res)
                sampled_activity = torch.nn.functional.grid_sample(
                    activity_frame,
                    source_coords_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                ).reshape(-1, 1)

            # Reconstruction: fixed_scene × activity
            reconstructed = fixed_scene_mask * sampled_activity

            # Main reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(reconstructed, target)

            # Regularization: Deformation smoothness
            deformation_smoothness = torch.mean(torch.abs(deformation))

            # Combined loss with regularization
            total_loss = recon_loss + 0.01 * deformation_smoothness

            # Optimize NSTM networks (deformation + fixed_scene only)
            optimizer_nstm.zero_grad()
            if optimizer_neural_renderer is not None:
                optimizer_neural_renderer.zero_grad()

            total_loss.backward()

            optimizer_nstm.step()
            if optimizer_neural_renderer is not None:
                optimizer_neural_renderer.step()

            # Record losses
            loss_history.append(recon_loss.item())
            regularization_history['deformation'].append(deformation_smoothness.item())

            # Update progress bar
            postfix_dict = {
                'recon': f'{recon_loss.item():.6f}',
                'def': f'{deformation_smoothness.item():.4f}'
            }
            # Add affine parameters if using pretrained SIREN
            if pretrained_activity_net is not None:
                postfix_dict['scale'] = f'{affine_scale.item():.2f}'
                postfix_dict['bias'] = f'{affine_bias.item():.1f}'

            if step % 100 == 0:
                pbar.set_postfix(postfix_dict)

    print("training complete!")

    # Print learned affine parameters if using pretrained SIREN
    if pretrained_activity_net is not None:
        print(f"learned affine transform: scale={affine_scale.item():.3f}, bias={affine_bias.item():.3f}")

    # Extract learned fixed_scene mask
    print("extracting learned fixed_scene mask...")
    with torch.no_grad():
        fixed_scene_mask = torch.sigmoid(fixed_scene_net(coords_2d))
        fixed_scene_mask_img = fixed_scene_mask.reshape(res, res).cpu().numpy()

    # Compute evaluation metrics (only if activity images were loaded)
    if activity_images:
        # Compute average of original activity images
        activity_average = np.mean(activity_images, axis=0)

        # Create ground-truth mask from activity average (threshold at mean)
        threshold = activity_average.mean()
        ground_truth_mask = (activity_average > threshold).astype(np.float32)
        n_gt_pixels = ground_truth_mask.sum()

        # Normalize fixed_scene mask for comparison
        fixed_scene_mask_norm = (fixed_scene_mask_img - fixed_scene_mask_img.min()) / (fixed_scene_mask_img.max() - fixed_scene_mask_img.min() + 1e-8)

        # Threshold fixed_scene to match ground truth coverage (top N pixels)
        flat_fixed_scene = fixed_scene_mask_norm.flatten()
        n_pixels_to_select = int(n_gt_pixels)
        sorted_indices = np.argsort(flat_fixed_scene)[::-1]  # Sort descending
        fixed_scene_binary_flat = np.zeros_like(flat_fixed_scene, dtype=np.float32)
        fixed_scene_binary_flat[sorted_indices[:n_pixels_to_select]] = 1.0
        fixed_scene_binary = fixed_scene_binary_flat.reshape(fixed_scene_mask_norm.shape)

        # Compute DICE score and IoU between learned fixed_scene and ground-truth mask
        intersection = np.sum(fixed_scene_binary * ground_truth_mask)
        union = np.sum(fixed_scene_binary) + np.sum(ground_truth_mask)
        dice_score = 2 * intersection / (union + 1e-8)
        iou_score = intersection / (union - intersection + 1e-8)

        # Compute median of motion frames (warped frames) as baseline
        motion_median = np.median(motion_images, axis=0)

        # Reconstruct fixed scene: fixed_scene × activity_average
        fixed_scene_denorm = fixed_scene_mask_img * activity_average

        # Compute RMSE between fixed scene and activity average
        rmse_activity = np.sqrt(np.mean((fixed_scene_denorm - activity_average) ** 2))

        # Compute SSIM between fixed scene and activity average
        # Normalize both to [0, 1] for SSIM computation
        fixed_scene_norm = (fixed_scene_denorm - fixed_scene_denorm.min()) / (fixed_scene_denorm.max() - fixed_scene_denorm.min() + 1e-8)
        activity_avg_norm = (activity_average - activity_average.min()) / (activity_average.max() - activity_average.min() + 1e-8)
        ssim_activity = ssim(activity_avg_norm, fixed_scene_norm, data_range=1.0)

        # Compute RMSE between motion median and activity average (baseline)
        rmse_baseline = np.sqrt(np.mean((motion_median - activity_average) ** 2))

        # Compute SSIM between motion median and activity average (baseline)
        motion_median_norm = (motion_median - motion_median.min()) / (motion_median.max() - motion_median.min() + 1e-8)
        ssim_baseline = ssim(activity_avg_norm, motion_median_norm, data_range=1.0)

        # Print simplified metrics (5 lines)
        print(f"fixed_scene: range=[{fixed_scene_mask_img.min():.3f}, {fixed_scene_mask_img.max():.3f}] | dice={dice_score:.3f} | iou={iou_score:.3f}")
        print(f"reconstruction: rmse={rmse_activity:.4f} | ssim={ssim_activity:.4f}")
        print(f"baseline:       rmse={rmse_baseline:.4f} | ssim={ssim_baseline:.4f}")
        print(f"improvement:    rmse={((rmse_baseline - rmse_activity) / rmse_baseline * 100):+.1f}% | ssim={((ssim_activity - ssim_baseline) / (1.0 - ssim_baseline) * 100):+.1f}%")
    else:
        # Using pretrained SIREN - skip metrics that require activity images
        print(f"fixed_scene: range=[{fixed_scene_mask_img.min():.3f}, {fixed_scene_mask_img.max():.3f}]")
        print("(skipping DICE/IoU/SSIM metrics - using pretrained SIREN without activity images)")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save learned fixed scene mask
    imwrite(f'{output_dir}/fixed_scene_learned.tif', fixed_scene_mask_img.astype(np.float32))

    # Save comparison figures (only if activity images were loaded)
    if activity_images:
        # Removed: imwrite(f'{output_dir}/fixed_scene_binary.tif', fixed_scene_binary.astype(np.float32))

        # Create fixed scene comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(fixed_scene_mask_norm, cmap='viridis')
        axes[0, 0].set_title('Learned Fixed Scene (Normalized)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(ground_truth_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(fixed_scene_binary, cmap='gray')
        axes[1, 0].set_title(f'Learned Fixed Scene (Binary)\nDICE: {dice_score:.3f}, IoU: {iou_score:.3f}')
        axes[1, 0].axis('off')

        # Difference map
        diff_map = np.abs(fixed_scene_binary - ground_truth_mask)
        axes[1, 1].imshow(diff_map, cmap='hot')
        axes[1, 1].set_title('Difference (Error Map)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        # Removed: plt.savefig(f'{output_dir}/fixed_scene_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Create fixed scene distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of raw learned fixed scene values
    axes[0].hist(fixed_scene_mask_img.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Fixed Scene Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Learned Fixed Scene Distribution (Raw)', fontsize=14)
    axes[0].axvline(fixed_scene_mask_img.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {fixed_scene_mask_img.mean():.3f}')
    axes[0].axvline(np.median(fixed_scene_mask_img), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(fixed_scene_mask_img):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of normalized fixed_scene values
    fixed_scene_mask_norm_hist = (fixed_scene_mask_img - fixed_scene_mask_img.min()) / (fixed_scene_mask_img.max() - fixed_scene_mask_img.min() + 1e-8)
    axes[1].hist(fixed_scene_mask_norm_hist.flatten(), bins=100, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Normalized Fixed Scene Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Learned Fixed Scene Distribution (Normalized)', fontsize=14)
    axes[1].axvline(fixed_scene_mask_norm_hist.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {fixed_scene_mask_norm_hist.mean():.3f}')
    axes[1].axvline(np.median(fixed_scene_mask_norm_hist), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(fixed_scene_mask_norm_hist):.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Removed: plt.savefig(f'{output_dir}/fixed_scene_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics to file
    near_zero_count = np.sum(np.abs(fixed_scene_mask_img) < 0.01)
    sparsity = near_zero_count / fixed_scene_mask_img.size * 100

    # Plot and save loss history with regularization terms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Main reconstruction loss
    axes[0].plot(loss_history)
    axes[0].set_title('NSTM Reconstruction Loss', fontsize=12)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(True, alpha=0.3)

    # Deformation regularization
    axes[1].plot(regularization_history['deformation'], color='orange')
    axes[1].set_title('Deformation Smoothness', fontsize=12)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('L1 Deformation')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_history.png', dpi=150)
    plt.close()

    # Generate SIREN activity video with learned affine transform
    if pretrained_activity_net is not None and neuron_positions is not None:
        print("generating SIREN activity video with learned affine transform...")
        render_siren_activity_video(
            pretrained_activity_net,
            neuron_positions,
            affine_scale,
            affine_bias,
            siren_config['nnr_f_T_period'],
            n_frames,
            res,
            device,
            output_dir
        )

    return deformation_net, fixed_scene_net, pretrained_activity_net, loss_history


def create_motion_field_visualization(base_image, motion_x, motion_y, res, step_size=16, black_background=False):
    """Create visualization of motion field with arrows"""
    if black_background:
        # Black background
        vis = np.zeros((res, res, 3), dtype=np.uint8)
    else:
        # Convert base image to RGB
        base_uint8 = (np.clip(base_image, 0, 1) * 255).astype(np.uint8)
        vis = np.stack([base_uint8, base_uint8, base_uint8], axis=2).copy()

    # Draw arrows
    for y in range(0, res, step_size):
        for x in range(0, res, step_size):
            dx = motion_x[y, x] * res * 5  # Scale for visibility (reduced from 10 to 5)
            dy = motion_y[y, x] * res * 5  # Scale for visibility (reduced from 10 to 5)

            if abs(dx) > 0.5 or abs(dy) > 0.5:  # Only draw significant motion
                pt1 = (int(x), int(y))
                pt2 = (int(x + dx), int(y + dy))
                cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

    return vis


def create_quad_panel_video(deformation_net, fixed_scene_net, activity_images, motion_images,
                            data_min, data_max, res, device, output_dir, num_frames=90,
                            boat_fixed_scene=None, boat_downsample_factor=None, activity_images_original=None,
                            use_neural_renderer=False):
    """Create an 8-panel comparison video (4 columns x 2 rows)

    Top row (Training Data):
    - Col 1: Activity
    - Col 2: Activity × Boat fixed_scene
    - Col 3: Ground truth motion field (arrows on black)
    - Col 4: Target (warped motion frames)

    Bottom row (Learned):
    - Col 1: Learned fixed_scene
    - Col 2: Neural Renderer (if use_neural_renderer=True) or Fixed Scene × Activity
    - Col 3: Learned motion field (arrows on black)
    - Col 4: NSTM Reconstruction
    """
    # Create temporary directory for frames
    temp_dir = f"{output_dir}/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Create coordinate grid
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Extract fixed_scene mask once
    with torch.no_grad():
        fixed_scene_mask = torch.sigmoid(fixed_scene_net(coords_2d))
        fixed_scene_mask_img = fixed_scene_mask.reshape(res, res).cpu().numpy()

    # Normalize fixed_scene for visualization
    fixed_scene_norm = (fixed_scene_mask_img - fixed_scene_mask_img.min()) / (fixed_scene_mask_img.max() - fixed_scene_mask_img.min() + 1e-8)

    # Prepare boat fixed_scene for visualization (if provided)
    if boat_fixed_scene is not None:
        boat_norm = (boat_fixed_scene - boat_fixed_scene.min()) / (boat_fixed_scene.max() - boat_fixed_scene.min() + 1e-8)
        boat_uint8 = (np.clip(boat_norm, 0, 1) * 255).astype(np.uint8)
        boat_rgb = np.stack([boat_uint8, boat_uint8, boat_uint8], axis=2)
    else:
        boat_rgb = np.zeros((res, res, 3), dtype=np.uint8)

    # Find original frames at evenly spaced time points
    n_activity_frames = len(activity_images)
    original_indices = [min(int(round(i * (n_activity_frames - 1) / (num_frames - 1))), n_activity_frames - 1)
                        for i in range(num_frames)]

    # Generate frames for each time point
    for i in trange(num_frames, desc="creating video frames", ncols=100):
        # Get normalized time
        t = i / (num_frames - 1)
        t_idx = original_indices[i]

        # === TOP ROW: Training Data ===

        # Top-1: Activity (always use matplotlib activity, not neural renderer)
        if activity_images_original is not None:
            # Use original matplotlib activity images
            activity_frame = activity_images_original[t_idx]
        else:
            # Fallback to activity_images (may be neural renderer if use_neural_renderer=True)
            activity_frame = activity_images[t_idx]

        activity_norm = (activity_frame - data_min) / (data_max - data_min)
        activity_uint8 = (np.clip(activity_norm, 0, 1) * 255).astype(np.uint8)
        activity_rgb = np.stack([activity_uint8, activity_uint8, activity_uint8], axis=2)

        # Top-2: Activity × Boat fixed_scene (before warping)
        if boat_fixed_scene is not None:
            activity_times_boat = activity_frame * boat_fixed_scene
            activity_boat_norm = (activity_times_boat - activity_times_boat.min()) / (activity_times_boat.max() - activity_times_boat.min() + 1e-8)
            activity_boat_uint8 = (np.clip(activity_boat_norm, 0, 1) * 255).astype(np.uint8)
            activity_boat_rgb = np.stack([activity_boat_uint8, activity_boat_uint8, activity_boat_uint8], axis=2)
        else:
            activity_boat_rgb = activity_rgb.copy()

        # Top-3: Ground truth motion field (arrows on black background)
        # Compute GT deformation at current frame
        gt_dx, gt_dy = compute_gt_deformation_field(t_idx, len(activity_images), res, motion_intensity=0.015)
        gt_motion_vis = create_motion_field_visualization(None, gt_dx, gt_dy, res, step_size=16, black_background=True)

        # Top-4: Target (warped motion frame)
        motion_frame = motion_images[t_idx]
        motion_norm = (motion_frame - data_min) / (data_max - data_min)
        motion_uint8 = (np.clip(motion_norm, 0, 1) * 255).astype(np.uint8)
        target_rgb = np.stack([motion_uint8, motion_uint8, motion_uint8], axis=2)

        # === BOTTOM ROW: Learned Components ===

        # Compute learned components
        with torch.no_grad():
            # Create 3D coordinates for deformation
            t_tensor = torch.full((coords_2d.shape[0], 1), t, device=device, dtype=torch.float16)
            coords_3d = torch.cat([coords_2d, t_tensor], dim=1)

            # Get deformation
            deformation = deformation_net(coords_3d)

            # Backward warp
            source_coords = coords_2d - deformation
            source_coords = torch.clamp(source_coords, 0, 1)

            # Sample fixed_scene at source coords
            fixed_scene_at_source = torch.sigmoid(fixed_scene_net(source_coords))

            # Sample activity at source coords
            activity_tensor = torch.tensor((activity_frame - data_min) / (data_max - data_min),
                                          dtype=torch.float16, device=device).reshape(1, 1, res, res)
            source_coords_normalized = source_coords * 2 - 1
            source_coords_grid = source_coords_normalized.reshape(1, 1, -1, 2)

            sampled_activity = torch.nn.functional.grid_sample(
                activity_tensor, source_coords_grid,
                mode='bilinear', padding_mode='border', align_corners=True
            ).reshape(-1, 1)

            # Reconstruction
            recon = (fixed_scene_at_source * sampled_activity).reshape(res, res).cpu().numpy()
            sampled_activity_img = sampled_activity.reshape(res, res).cpu().numpy()

        # Bottom-1: Learned fixed_scene
        fixed_scene_uint8 = (np.clip(fixed_scene_norm, 0, 1) * 255).astype(np.uint8)
        learned_fixed_scene_rgb = np.stack([fixed_scene_uint8, fixed_scene_uint8, fixed_scene_uint8], axis=2)

        # Bottom-2: Neural Renderer or Fixed Scene × Activity
        if use_neural_renderer:
            # Show neural renderer output directly (constant activity if threshold_activity=True)
            # Use activity_images (from activity_neural/) instead of activity_frame (which may be matplotlib)
            neural_renderer_frame = activity_images[t_idx]
            neural_renderer_norm = (neural_renderer_frame - data_min) / (data_max - data_min + 1e-8)
            neural_renderer_uint8 = (np.clip(neural_renderer_norm, 0, 1) * 255).astype(np.uint8)
            fixed_scene_activity_rgb = np.stack([neural_renderer_uint8, neural_renderer_uint8, neural_renderer_uint8], axis=2)
        else:
            # Fixed Scene × Activity (use global data range to prevent flickering)
            fixed_scene_times_activity = fixed_scene_mask_img * activity_frame
            fixed_scene_act_norm = (fixed_scene_times_activity - data_min) / (data_max - data_min + 1e-8)
            fixed_scene_act_uint8 = (np.clip(fixed_scene_act_norm, 0, 1) * 255).astype(np.uint8)
            fixed_scene_activity_rgb = np.stack([fixed_scene_act_uint8, fixed_scene_act_uint8, fixed_scene_act_uint8], axis=2)

        # Bottom-3: Learned motion field (arrows on black background)
        deformation_2d = deformation.reshape(res, res, 2).cpu().numpy()
        motion_x = deformation_2d[:, :, 0]
        motion_y = deformation_2d[:, :, 1]
        learned_motion_vis = create_motion_field_visualization(fixed_scene_norm, motion_x, motion_y, res, step_size=16, black_background=True)

        # Bottom-4: NSTM Reconstruction
        recon_denorm = recon * (data_max - data_min) + data_min
        recon_norm = (recon_denorm - data_min) / (data_max - data_min)
        recon_uint8 = (np.clip(recon_norm, 0, 1) * 255).astype(np.uint8)
        recon_rgb = np.stack([recon_uint8, recon_uint8, recon_uint8], axis=2)

        # Add text labels - top-left position with larger font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        margin = 10
        y_pos = margin + 25

        # Top row labels
        cv2.putText(activity_rgb, "Activity", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(activity_boat_rgb, "Activity x Boat", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(gt_motion_vis, "GT Motion", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(target_rgb, "Target", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Bottom row labels
        cv2.putText(learned_fixed_scene_rgb, "Learned Fixed Scene", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)
        panel2_title = "Neural Renderer" if use_neural_renderer else "Fixed Scene x Activity"
        cv2.putText(fixed_scene_activity_rgb, panel2_title, (margin, y_pos), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(learned_motion_vis, "Learned Motion", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(recon_rgb, "Reconstruction", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Create 4x2 grid layout
        top_row = np.hstack([activity_rgb, activity_boat_rgb, gt_motion_vis, target_rgb])
        bottom_row = np.hstack([learned_fixed_scene_rgb, fixed_scene_activity_rgb, learned_motion_vis, recon_rgb])
        combined = np.vstack([top_row, bottom_row])

        # Save frame
        cv2.imwrite(f"{temp_dir}/frame_{i:04d}.png", combined)

    # Create video with ffmpeg
    video_path = f'{output_dir}/stage5_nstm.mp4'
    fps = 30

    print("creating video from frames...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-pattern_type", "glob", "-i", f"{temp_dir}/frame_*.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "22",
        video_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"video saved to {video_path}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"error creating video with ffmpeg: {e}")
        print("falling back to opencv video writer...")

        # Fallback to OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (res*4, res*2))

        for j in range(num_frames):
            frame_path = f"{temp_dir}/frame_{j:04d}.png"
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
        print(f"video saved to {video_path}")

    # Clean up temporary files
    print("cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    return video_path


def create_video(frames_list, output_path, fps=30, layout='1panel', panel_titles=None):
    """
    Universal video creation function for 1-panel, 2-panel, or 8-panel layouts

    Args:
        frames_list: List of numpy arrays or dict with keys for different panels
                     - For 1panel: single list of (H, W) or (H, W, 3) arrays
                     - For 2panel: dict like {'left': [...], 'right': [...]}
                     - For 8panel: dict like {'row0_col0': [...], 'row0_col1': [...], ...}
        output_path: Path to save MP4 file
        fps: Frames per second
        layout: '1panel', '2panel', or '8panel'
        panel_titles: List of titles for each panel (optional)
    """
    temp_dir = f"{os.path.dirname(output_path)}/temp_video_frames"
    os.makedirs(temp_dir, exist_ok=True)

    if layout == '1panel':
        n_frames = len(frames_list)

        # Compute global min/max across all frames for consistent normalization
        all_values = np.concatenate([frame.flatten() for frame in frames_list])
        global_min = all_values.min()
        global_max = all_values.max()

        for i, frame in enumerate(tqdm(frames_list, desc="creating video", ncols=100)):
            # Convert to RGB uint8 with global normalization
            frame_rgb = _to_rgb_uint8(frame, global_min, global_max)

            # Add title if provided
            if panel_titles and len(panel_titles) > 0:
                cv2.putText(frame_rgb, panel_titles[0], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imwrite(f"{temp_dir}/frame_{i:06d}.png", frame_rgb)

    elif layout == '2panel':
        # Assume frames_list is dict with 'left' and 'right' keys
        left_frames = frames_list['left']
        right_frames = frames_list['right']
        n_frames = len(left_frames)

        # Compute global min/max for left and right separately
        all_left = np.concatenate([frame.flatten() for frame in left_frames])
        all_right = np.concatenate([frame.flatten() for frame in right_frames])
        left_min, left_max = all_left.min(), all_left.max()
        right_min, right_max = all_right.min(), all_right.max()

        for i in tqdm(range(n_frames), desc="creating video", ncols=100):
            left = left_frames[i]
            right = right_frames[i]

            # Convert to RGB uint8 with global normalization per panel
            left_rgb = _to_rgb_uint8(left, left_min, left_max)
            right_rgb = _to_rgb_uint8(right, right_min, right_max)

            # Add titles if provided
            if panel_titles:
                cv2.putText(left_rgb, panel_titles[0], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(right_rgb, panel_titles[1], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Horizontal concatenation
            combined = np.hstack([left_rgb, right_rgb])
            cv2.imwrite(f"{temp_dir}/frame_{i:06d}.png", combined)

    elif layout == '8panel':
        # Assume frames_list is dict with grid structure
        # Expected keys: 'row0_col0', 'row0_col1', ..., 'row1_col3'
        n_frames = len(frames_list['row0_col0'])

        for i in tqdm(range(n_frames), desc="creating video", ncols=100):
            # Top row (4 panels)
            top_panels = []
            for col in range(4):
                key = f'row0_col{col}'
                panel = _to_rgb_uint8(frames_list[key][i])
                if panel_titles and col < len(panel_titles):
                    cv2.putText(panel, panel_titles[col], (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                top_panels.append(panel)

            # Bottom row (4 panels)
            bottom_panels = []
            for col in range(4):
                key = f'row1_col{col}'
                panel = _to_rgb_uint8(frames_list[key][i])
                if panel_titles and (4 + col) < len(panel_titles):
                    cv2.putText(panel, panel_titles[4 + col], (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                bottom_panels.append(panel)

            # Concatenate
            top_row = np.hstack(top_panels)
            bottom_row = np.hstack(bottom_panels)
            combined = np.vstack([top_row, bottom_row])

            cv2.imwrite(f"{temp_dir}/frame_{i:06d}.png", combined)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Create video with ffmpeg
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%06d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 {output_path}"
    )
    subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

    # Clean up
    shutil.rmtree(temp_dir)


def _to_rgb_uint8(frame, global_min=None, global_max=None):
    """Convert frame to RGB uint8 format

    Args:
        frame: Input frame (grayscale or RGB)
        global_min: Global minimum value for normalization (if None, use frame min)
        global_max: Global maximum value for normalization (if None, use frame max)
    """
    # Ensure RGB
    if len(frame.shape) == 2:
        frame_rgb = np.stack([frame, frame, frame], axis=2)
    else:
        frame_rgb = frame

    # Ensure uint8
    if frame_rgb.dtype != np.uint8:
        # Use global min/max if provided, otherwise per-frame normalization
        frame_min = global_min if global_min is not None else frame_rgb.min()
        frame_max = global_max if global_max is not None else frame_rgb.max()

        if frame_max > frame_min:
            frame_norm = (frame_rgb - frame_min) / (frame_max - frame_min)
        else:
            frame_norm = np.zeros_like(frame_rgb)
        frame_rgb = (np.clip(frame_norm, 0, 1) * 255).astype(np.uint8)

    return frame_rgb


def stage1_train_siren(x_list, device, output_dir, config, use_constant_activity=False):
    """
    Stage 1: Train SIREN on discrete neuron activities

    Args:
        x_list: List of neuron data arrays (n_frames, n_neurons, features)
        device: torch device
        output_dir: Directory for outputs
        config: Configuration dict with SIREN parameters
        use_constant_activity: If True, train SIREN with constant activity value of 5
                               (demonstrates that INR decomposition requires time-dependent activity)

    Returns:
        siren_net: Trained SIREN network
    """
    if use_constant_activity:
        print("stage 1: training siren with CONSTANT activity (value=5)")
        print("  -> this demonstrates that INR decomposition requires time-dependent activity knowledge")

        # Create modified x_list with constant activity value of 5
        x_list_constant = []
        for run_data in x_list:
            run_constant = []
            for frame_data in run_data:
                # frame_data shape: (n_neurons, features)
                # Set activity column (index 6) to constant value 5
                frame_modified = frame_data.copy()
                frame_modified[:, 6] = 5.0  # Set all activities to constant 5
                run_constant.append(frame_modified)
            x_list_constant.append(np.array(run_constant))

        siren_net = train_siren(
            x_list=x_list_constant,
            device=device,
            output_dir=output_dir,
            num_training_steps=config.get('num_training_steps', 20000),
            nnr_f_T_period=config.get('nnr_f_T_period', 10),
            n_train_frames=config.get('n_train_frames', 256),
            n_neurons=config.get('n_neurons', 100)
        )
    else:
        print("stage 1: training siren")

        siren_net = train_siren(
            x_list=x_list,
            device=device,
            output_dir=output_dir,
            num_training_steps=config.get('num_training_steps', 20000),
            nnr_f_T_period=config.get('nnr_f_T_period', 10),
            n_train_frames=config.get('n_train_frames', 256),
            n_neurons=config.get('n_neurons', 100)
        )

    return siren_net


def stage2_generate_activity_images(x_list, neuron_positions, n_frames, res, device,
                                     output_dir, activity_dir, activity_original_dir=None,
                                     threshold_activity=False, run=0):
    """
    Stage 2: Generate activity images using matplotlib scatter rendering

    Args:
        x_list: List of neuron data arrays (n_frames, n_neurons, features)
        neuron_positions: (n_neurons, 2) array of positions in [-0.5, 0.5]
        n_frames: Number of frames to generate
        res: Image resolution
        device: torch device
        output_dir: Directory for outputs
        activity_dir: Directory to save activity images
        activity_original_dir: Not used (kept for compatibility)
        threshold_activity: Not used (kept for compatibility - constant activity is set in stage 1)
        run: Run index (default 0)

    Returns:
        activity_images: List of generated activity images
        activity_images_original: Always None (no longer used)
    """
    print("stage 2: generating activity images with matplotlib")

    os.makedirs(activity_dir, exist_ok=True)

    activity_images = []
    activity_images_original = None

    # Convert neuron positions to tensor for plotting
    X1 = torch.tensor(neuron_positions, dtype=torch.float32, device=device)
    for frame_idx in tqdm(range(n_frames), ncols=100):
        x = torch.tensor(x_list[run][frame_idx], dtype=torch.float32, device=device)
        # Create figure and render to get pixel data
        # Flip y-coordinates to match neural renderer coordinate system
        fig = plt.figure(figsize=(512/80, 512/80), dpi=80)  # 512x512 pixels
        plt.scatter(
            to_numpy(X1[:, 0]),
            -to_numpy(X1[:, 1]),  # Flip y-axis: y -> -y
            s=700,
            c=to_numpy(x[:, 3]),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        # Render to canvas and extract grayscale data
        fig.canvas.draw()
        img_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_rgba = img_rgba[:, :, :3]  # Convert RGBA to RGB

        # Convert RGB to grayscale
        img_gray = np.dot(img_rgba[...,:3], [0.2989, 0.5870, 0.1140])

        # Resize to exactly 512x512 if needed
        from scipy.ndimage import zoom
        if img_gray.shape != (512, 512):
            zoom_factors = (512 / img_gray.shape[0], 512 / img_gray.shape[1])
            img_gray = zoom(img_gray, zoom_factors, order=1)

        img_activity_32bit = img_gray.astype(np.float32)
        activity_images.append(img_activity_32bit)

        # Save as TIFF (float32, matching original)
        imwrite(
            f"{activity_dir}/frame_{frame_idx:06d}.tif",
            img_activity_32bit,
            photometric='minisblack',
            dtype=np.float32
        )

        plt.close(fig)


    print(f"activity: [{np.min(img_activity_32bit):.2f}, {np.max(img_activity_32bit):.2f}]")

    video_path = f"{output_dir}/stage2_activity.mp4"
    create_video(
        frames_list=activity_images,
        output_path=video_path,
        fps=30,
        layout='1panel',
        panel_titles=['activity']
    )

    return activity_images, activity_images_original


def stage3_generate_warped_motion_frames(activity_images, boat_fixed_scene, n_frames,
                                         output_dir, motion_frames_dir, target_downsample_factor=1,
                                         motion_intensity=0.015):
    """
    Stage 3: Generate warped motion frames (activity × boat + sinusoidal warp)

    Args:
        activity_images: List of activity images
        boat_fixed_scene: Boat anatomy/fixed scene image
        n_frames: Number of frames
        output_dir: Directory for outputs
        motion_frames_dir: Directory to save motion frames
        target_downsample_factor: Downsample targets for super-resolution test
        motion_intensity: Sinusoidal warp intensity

    Returns:
        motion_images: List of warped motion frames
    """
    print("stage 3: generating motion frames")
    if target_downsample_factor > 1:
        print(f"targets will be downsampled by {target_downsample_factor}x for super-resolution test")

    os.makedirs(motion_frames_dir, exist_ok=True)

    motion_images = []
    from scipy.ndimage import zoom

    for frame_idx in tqdm(range(n_frames), ncols=100):
        activity_frame = activity_images[frame_idx]

        # Element-wise multiplication: activity × boat_fixed_scene
        img_with_fixed_scene = activity_frame * boat_fixed_scene

        # Apply sinusoidal warping
        img_warped = apply_sinusoidal_warp(img_with_fixed_scene, frame_idx, n_frames,
                                           motion_intensity=motion_intensity)

        # Downsample target if requested (for super-resolution test)
        if target_downsample_factor > 1:
            res = img_warped.shape[0]
            downsampled_size = res // target_downsample_factor
            # Downsample using nearest neighbor
            zoom_factor_down = downsampled_size / res
            img_downsampled = zoom(img_warped, zoom_factor_down, order=0)
            # Upsample back to original resolution
            zoom_factor_up = res / downsampled_size
            img_warped = zoom(img_downsampled, zoom_factor_up, order=0)

        motion_images.append(img_warped)

        # Save as TIFF
        imwrite(f"{motion_frames_dir}/frame_{frame_idx:06d}.tif", img_warped.astype(np.float32))

    print(f"boat: [{boat_fixed_scene.min():.2f}, {boat_fixed_scene.max():.2f}]")
    print(f"motion: [{np.min([img.min() for img in motion_images]):.2f}, {np.max([img.max() for img in motion_images]):.2f}]")

    # Create 1-panel MP4 video (warped target only)
    video_path = f"{output_dir}/stage3_motion.mp4"
    create_video(
        frames_list=motion_images,
        output_path=video_path,
        fps=30,
        layout='1panel',
        panel_titles=['motion']
    )

    return motion_images


def stage4_train_neural_renderer(siren_net, x_list, neuron_positions, n_frames, res, device,
                                  output_dir, activity_dir, activity_neural_dir, nnr_f_T_period=10, run=0):
    """
    Stage 4: Train NeuralRenderer (Soft Voronoi + MLP) to mimic matplotlib rendering

    ARCHITECTURE:
    - Soft Voronoi: Differentiable Voronoi tessellation with smooth bump kernels
    - MLP: Refines scalar field to match matplotlib style
    - Learnable params: affine transform (4), σ (1), β (1), MLP weights (~4k)

    Training schedule:
        - Epoch 1: 2000 iterations, LR=1e-3
        - Epoch 2: 2000 iterations, LR=5e-5

    Args:
        siren_net: Trained SIREN network from Stage 1
        x_list: List of neuron data arrays (for ground truth comparison)
        neuron_positions: (n_neurons, 2) array of neuron positions in [-0.5, 0.5]
        n_frames: Number of frames to train on
        res: Image resolution (512)
        device: torch device
        output_dir: Directory for outputs
        activity_dir: Directory with true activity images (from Stage 2)
        activity_neural_dir: Directory to save neural renderer activity images
        nnr_f_T_period: Temporal period for SIREN
        run: Run index for x_list

    Returns:
        neural_renderer: Trained NeuralRenderer network
        activity_neural_images: List of neural renderer activity images
    """
    import matplotlib.pyplot as plt
    from NeuralGraph.models.NeuralRenderer import NeuralRenderer
    from tifffile import imread
    import torch
    import numpy as np

    print("stage 4: training neural renderer (soft voronoi + mlp)")

    # Preload all true activity images from Stage 2 (targets for training)
    print(f"loading true activity images from {activity_dir}...")
    activity_images_target = []
    for frame_idx in range(n_frames):
        img = imread(f"{activity_dir}/frame_{frame_idx:06d}.tif")
        activity_images_target.append(torch.tensor(img, dtype=torch.float32, device=device))
    print(f"loaded {n_frames} activity images")

    print(f"reloaded last saved image: [{img.min():.2f}, {img.max():.2f}]")

    # Create neural renderer with Soft Voronoi architecture
    # - Soft Voronoi: affine (4 params) + σ (1 param) + β (1 param)
    # - MLP: 3 layers with 64 hidden units (~4k params)
    neural_renderer = NeuralRenderer(
        resolution=res,
        sigma_init=-2,   # ~30 px
        beta_init=500,   # sharp-ish edge
        hidden_dim=64,
    ).to(device)

    # Convert neuron positions to torch tensor (normalize from [-0.5, 0.5] to [0, 1])
    neuron_positions_torch = torch.tensor(neuron_positions, dtype=torch.float32, device=device)
    neuron_positions_torch = (neuron_positions_torch + 0.5)  # [-0.5, 0.5] -> [0, 1]
    n_neurons = neuron_positions.shape[0]

    # Create query grid for training (all pixel coordinates)
    query_grid = neural_renderer.create_grid(res, device=device)  # (res*res, 2)

    # Training loop
    loss_history = []

    siren_net.eval()  # SIREN is frozen, only train renderer
    neural_renderer.train()

    fig_dir = f"{output_dir}/Fig"
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    # Two-stage training schedule
    training_schedule = [
        {'lr': 5e-4, 'steps': 5000, 'epoch': 1},
        {'lr': 5e-5, 'steps': 5000, 'epoch': 2}
    ]

    global_step = 0  # Track global step across epochs
    for schedule in training_schedule:
        optimizer = torch.optim.Adam(neural_renderer.parameters(), lr=schedule['lr'])
        epoch_losses = []

        pbar = trange(schedule['steps'], desc=f"epoch {schedule['epoch']}", ncols=150)
        for step in pbar:
            # Sample random frame
            frame_idx = np.random.randint(0, n_frames)

            # Get SIREN-predicted activities for this frame
            t_normalized = frame_idx / (n_frames - 1) if n_frames > 1 else 0.0
            t_input = torch.tensor(t_normalized * (2 * np.pi / nnr_f_T_period),
                                  dtype=torch.float32, device=device).reshape(1, 1)

            with torch.no_grad():
                activities_siren = siren_net(t_input).squeeze()  # (n_neurons,) in [0, 40] (pre-shifted)

            # Use preloaded true activity image as target (from Stage 2)
            target_image = activity_images_target[frame_idx].flatten()  # (res*res,)

            # Predict with neural renderer (Gaussian splatting + MLP)
            # Use normalized positions [0, 1] for neural renderer
            predicted_flat = neural_renderer(neuron_positions_torch, activities_siren, query_grid)  # (res*res,)

            loss_mse = torch.nn.functional.mse_loss(predicted_flat, target_image)
            loss_l1  = torch.nn.functional.l1_loss(predicted_flat, target_image)
            loss     = loss_mse + 0.1 * loss_l1


            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            loss_history.append(loss.item())

            if step % 100 == 0:
                params = neural_renderer.get_learnable_params()
                import torch.nn.functional as F
                # Show actual radius R (after softplus), not raw parameter
                R_actual = params['sigma'] + 1e-6
                beta_actual = F.softplus(torch.tensor(params['beta'])).item() + 1e-6
                pbar.set_postfix({
                    'loss': f'{loss.item():.2f}',
                    'σ': f"{R_actual:.3f}",
                    'β': f"{beta_actual:.1f}",
                    'ax': f"{params['affine_ax']:.2f}",
                    'tx': f"{params['affine_tx']:.3f}"
                })


            # Save visualization every 200 iterations
            if global_step % 200 == 0:

                neural_renderer.eval()
                with torch.no_grad():
                    # Use current frame for visualization
                    # 1. True activity image (target)
                    true_img = activity_images_target[frame_idx].cpu().numpy()

                    # 2. Gaussian splatting only
                    gaussian_only_flat = neural_renderer.forward_splatting_only(
                        neuron_positions_torch, activities_siren, query_grid
                    )
                    gaussian_only_img = gaussian_only_flat.reshape(res, res).cpu().numpy()

                    # 3. Full neural renderer (gaussian + MLP)
                    neural_flat = neural_renderer(neuron_positions_torch, activities_siren, query_grid)
                    neural_img = neural_flat.reshape(res, res).cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # Panel 1: True Activity (vmin=0, vmax=100)
                im0 = axes[0].imshow(true_img, vmin=0, vmax=255, cmap='gray')
                axes[0].set_title('true activity', fontsize=12)
                axes[0].axis('off')
                plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                # Panel 2: Soft Voronoi
                im1 = axes[1].imshow(gaussian_only_img, cmap='coolwarm')
                axes[1].set_title('soft voronoi', fontsize=12)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                # Panel 3: Soft Voronoi + MLP
                im2 = axes[2].imshow(neural_img, vmin=0, vmax=255, cmap='gray')
                axes[2].set_title('soft voronoi + MLP', fontsize=12)
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                # Add parameter text below panels
                param_text = f"σ={params['sigma']:.2f}, β={params['beta']:.3f}"
                fig.suptitle(param_text, fontsize=10, y=0.02)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(f"{fig_dir}/step_{global_step:06d}.png", dpi=150)
                plt.close(fig)

                neural_renderer.train()

            global_step += 1

    print("neural renderer training complete!")

    # Summary of network architecture
    total_params = neural_renderer.get_num_parameters()
    learnable_params = neural_renderer.get_learnable_params()
    voronoi_params = 6  # affine (4) + σ (1) + β (1)
    mlp_params = total_params - voronoi_params

    print(f"  total parameters: {total_params:,}")
    print(f"    - soft voronoi: {voronoi_params} params")
    print(f"      affine: ax={learnable_params['affine_ax']:.3f}, ay={learnable_params['affine_ay']:.3f}, "
          f"tx={learnable_params['affine_tx']:.3f}, ty={learnable_params['affine_ty']:.3f}")
    print(f"      kernel: σ={learnable_params['sigma']:.4f}, β={learnable_params['beta']:.2f}")
    print(f"    - mlp: {mlp_params:,} params")

    # Plot loss history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title('neural renderer training loss', fontsize=12)
    plt.xlabel('step')
    plt.ylabel('mse loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/neural_renderer_loss.png', dpi=150)
    plt.close()

    # Generate comparison video: 3 panels (target | gaussian splatting | gaussian + MLP)
    print("generating 3-panel comparison video...")
    neural_renderer.eval()

    matplotlib_frames = []
    gaussian_only_frames = []
    neural_frames = []

    with torch.no_grad():
        for frame_idx in tqdm(range(n_frames), desc="rendering comparison frames", ncols=100):
            # Get SIREN activities (already shifted to [0, 40])
            t_normalized = frame_idx / (n_frames - 1) if n_frames > 1 else 0.0
            t_input = torch.tensor(t_normalized * (2 * np.pi / nnr_f_T_period),
                                  dtype=torch.float32, device=device).reshape(1, 1)
            activities = siren_net(t_input).squeeze()  # (n_neurons,) in [0, 40]

            # 1. True activity image (target from Stage 2)
            matplotlib_img = activity_images_target[frame_idx].cpu().numpy()
            matplotlib_frames.append(matplotlib_img)

            # 2. Gaussian splatting only (WITHOUT MLP refinement)
            gaussian_only_flat = neural_renderer.forward_splatting_only(
                neuron_positions_torch, activities, query_grid
            )
            gaussian_only_img = gaussian_only_flat.reshape(res, res).cpu().numpy()
            gaussian_only_frames.append(gaussian_only_img)

            # 3. Neural renderer (Gaussian splatting + MLP)
            neural_flat = neural_renderer(neuron_positions_torch, activities, query_grid)
            neural_img = neural_flat.reshape(res, res).cpu().numpy()
            neural_frames.append(neural_img)

    # Create 3-panel video with matplotlib (matching training visualization)
    video_path = f"{output_dir}/stage4_renderer_comparison.mp4"

    # Compute global min/max for consistent colorbar ranges
    all_matplotlib = np.concatenate([f.flatten() for f in matplotlib_frames])
    all_gaussian_only = np.concatenate([f.flatten() for f in gaussian_only_frames])
    all_neural = np.concatenate([f.flatten() for f in neural_frames])

    matplotlib_vmin, matplotlib_vmax = all_matplotlib.min(), all_matplotlib.max()
    gaussian_vmin, gaussian_vmax = all_gaussian_only.min(), all_gaussian_only.max()
    neural_vmin, neural_vmax = all_neural.min(), all_neural.max()

    # Create video frames using matplotlib
    temp_dir = f"{output_dir}/temp_renderer_frames"
    os.makedirs(temp_dir, exist_ok=True)

    params = neural_renderer.get_learnable_params()
    import torch.nn.functional as F
    R_actual = F.softplus(torch.tensor(params['sigma'])).item() + 1e-6
    beta_actual = F.softplus(torch.tensor(params['beta'])).item() + 1e-6

    for i in tqdm(range(len(matplotlib_frames)), desc="saving frames", ncols=100):
        # Use figure size that produces even dimensions: 15x5 @ 80 dpi = 1200x400 (both even)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: True Activity
        im0 = axes[0].imshow(matplotlib_frames[i], vmin=matplotlib_vmin, vmax=matplotlib_vmax, cmap='gray')
        axes[0].set_title('true activity', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: Soft Voronoi (use coolwarm for signed values)
        im1 = axes[1].imshow(gaussian_only_frames[i], vmin=gaussian_vmin, vmax=gaussian_vmax, cmap='coolwarm')
        axes[1].set_title('soft voronoi', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Soft Voronoi + MLP
        im2 = axes[2].imshow(neural_frames[i], vmin=neural_vmin, vmax=neural_vmax, cmap='gray')
        axes[2].set_title('soft voronoi + MLP', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Add parameter text below panels
        param_text = f"σ={R_actual:.3f}, β={beta_actual:.1f}, ax={params['affine_ax']:.2f}, tx={params['affine_tx']:.3f}"
        fig.suptitle(param_text, fontsize=10, y=0.02)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Use dpi=80 to get 1200x400 pixels (both divisible by 2)
        plt.savefig(f"{temp_dir}/frame_{i:06d}.png", dpi=80)
        plt.close(fig)

    # Create video with ffmpeg
    # Use vf pad filter to ensure dimensions are divisible by 2 for libx264
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate 30 -i {temp_dir}/frame_%06d.png "
        f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 {video_path}"
    )
    subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True, text=True)

    # Clean up temp frames
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print(f"comparison video saved to {video_path}")

    # Save neural renderer activity images (reuse from comparison video generation)
    print("saving neural renderer activity images for all frames")
    os.makedirs(activity_neural_dir, exist_ok=True)

    from tifffile import imwrite
    for frame_idx, neural_img in enumerate(tqdm(neural_frames, desc="saving activity images", ncols=100)):
        # Save as TIFF
        imwrite(
            f"{activity_neural_dir}/frame_{frame_idx:06d}.tif",
            neural_img.astype(np.float32),
            photometric='minisblack',
            dtype=np.float32
        )

    activity_neural_images = neural_frames
    print(f"neural renderer activity images saved to {activity_neural_dir}")

    # Calculate R² between matplotlib (true) and neural renderer (inferred) images
    r2_scores = []
    for matplotlib_img, neural_img in zip(matplotlib_frames, neural_frames):
        # Flatten images to 1D arrays
        y_true = matplotlib_img.flatten()
        y_pred = neural_img.flatten()

        # Calculate R²: R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        r2_scores.append(r2)

    avg_r2 = np.mean(r2_scores)
    print(f"  avg R² (matplotlib vs neural renderer): {avg_r2:.4f}")

    # Create scatter plot to visualize pixel intensity correlation
    print("creating scatter plot of pixel intensities...")
    import matplotlib.pyplot as plt

    # Sample pixels from all frames for scatter plot (downsample to avoid too many points)
    sample_step = 10  # Sample every 10th pixel
    all_matplotlib_pixels = []
    all_neural_pixels = []

    for matplotlib_img, neural_img in zip(matplotlib_frames, neural_frames):
        matplotlib_flat = matplotlib_img.flatten()[::sample_step]
        neural_flat = neural_img.flatten()[::sample_step]
        all_matplotlib_pixels.append(matplotlib_flat)
        all_neural_pixels.append(neural_flat)

    all_matplotlib_pixels = np.concatenate(all_matplotlib_pixels)
    all_neural_pixels = np.concatenate(all_neural_pixels)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with transparency and downsampling for visibility
    ax.hexbin(all_matplotlib_pixels, all_neural_pixels, gridsize=50, cmap='Blues', mincnt=1)

    # Plot diagonal line (perfect correlation)
    min_val = min(all_matplotlib_pixels.min(), all_neural_pixels.min())
    max_val = max(all_matplotlib_pixels.max(), all_neural_pixels.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect correlation')

    # Labels and statistics
    ax.set_xlabel('Matplotlib pixel intensity', fontsize=12)
    ax.set_ylabel('Neural renderer pixel intensity', fontsize=12)
    ax.set_title(f'Pixel Intensity Correlation (R²={avg_r2:.4f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Matplotlib: [{all_matplotlib_pixels.min():.1f}, {all_matplotlib_pixels.max():.1f}]\n'
    stats_text += f'Neural renderer: [{all_neural_pixels.min():.1f}, {all_neural_pixels.max():.1f}]'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/stage4_intensity_correlation.png', dpi=150)
    plt.close()
    print(f"scatter plot saved to {output_dir}/stage4_intensity_correlation.png")

    neural_renderer.train()
    return neural_renderer, activity_neural_images


def stage5_train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir,
                      siren_config, pretrained_activity_net, x_list, boat_fixed_scene=None,
                      boat_downsample_factor=1, activity_images_original=None,
                      neural_renderer=None, neural_renderer_learnable=False,
                      neuron_positions=None, siren_net=None):
    """
    Stage 5: Train NSTM (deformation + fixed_scene) on warped frames

    Args:
        motion_frames_dir: Directory with warped motion frames
        activity_dir: Directory with activity images
        n_frames: Number of frames
        res: Image resolution
        device: torch device
        output_dir: Directory for outputs
        siren_config: SIREN configuration dict
        pretrained_activity_net: Trained SIREN network (optional)
        x_list: Neuron data for SIREN supervision (optional)
        boat_fixed_scene: Boat fixed scene image (optional)
        boat_downsample_factor: Downsampling factor for boat image (optional)
        activity_images_original: Original activity images before thresholding (optional)
        neural_renderer: Trained NeuralRenderer from Stage 4 (optional)
        neural_renderer_learnable: If True and neural_renderer provided, make it learnable (LR=1e-6)
        neuron_positions: Neuron positions needed if using neural_renderer
        siren_net: SIREN network needed if using neural_renderer

    Returns:
        Trained networks and loss history
    """
    if neural_renderer is not None:
        if neural_renderer_learnable:
            print("stage 5: training nstm with learnable neural renderer (lr=1e-6)")
        else:
            print("stage 5: training nstm with fixed neural renderer")
    else:
        print("stage 5: training nstm with matplotlib activity")

    deformation_net, fixed_scene_net, activity_net, loss_history = train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        num_training_steps=20000,
        siren_config=siren_config,
        pretrained_activity_net=pretrained_activity_net,
        x_list=x_list,
        use_siren=False,  # Use ground truth activity images, not SIREN
        siren_lr=1e-4,
        nstm_lr=2.5e-4,
        siren_loss_weight=1.0,
        neural_renderer=neural_renderer,
        neural_renderer_learnable=neural_renderer_learnable,
        neural_renderer_lr=1e-6,
        neuron_positions=neuron_positions,
        siren_net=siren_net
    )

    # Create 8-panel video after training
    # Note: create_quad_panel_video is defined in this file (line 819)
    from tifffile import imread

    # Load all frames for video creation
    motion_images_list = []
    activity_images_list = []
    for i in range(n_frames):
        motion_images_list.append(imread(f"{motion_frames_dir}/frame_{i:06d}.tif"))
        activity_images_list.append(imread(f"{activity_dir}/frame_{i:06d}.tif"))

    # Compute data range for normalization
    all_pixels = np.concatenate([img.flatten() for img in motion_images_list])
    data_min = all_pixels.min()
    data_max = all_pixels.max()

    # Load matplotlib activity images for top-left panel if using neural renderer
    activity_images_original_list = None
    if neural_renderer is not None:
        # When using neural renderer, activity_dir contains neural renderer images
        # Load original matplotlib activity images for top-left panel display
        matplotlib_activity_dir = activity_dir.replace('activity_neural', 'activity')
        activity_images_original_list = []
        for i in range(n_frames):
            activity_images_original_list.append(imread(f"{matplotlib_activity_dir}/frame_{i:06d}.tif"))
    elif activity_images_original is not None:
        # Load original activity images if thresholding was used
        activity_images_original_list = activity_images_original  # Use all frames

    # Create 8-panel comparison video
    video_path = create_quad_panel_video(
        deformation_net=deformation_net,
        fixed_scene_net=fixed_scene_net,
        activity_images=activity_images_list,
        motion_images=motion_images_list,
        data_min=data_min,
        data_max=data_max,
        res=res,
        device=device,
        output_dir=output_dir,
        num_frames=n_frames,
        boat_fixed_scene=boat_fixed_scene,
        boat_downsample_factor=boat_downsample_factor,
        activity_images_original=activity_images_original_list,
        use_neural_renderer=(neural_renderer is not None)
    )
    print("stage4_nstm.mp4")

    return deformation_net, fixed_scene_net, activity_net, loss_history


def data_train_NGP(config=None, device=None):
    """
    Main pipeline: Sequential execution of 5 stages

    Args:
        config: NeuralGraphConfig object (optional)
        device: torch device (optional)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    if config is not None:
        dataset_name = config.dataset
    else:
        dataset_name = "signal_N11_2_1"
    base_dir = f"/groups/saalfeld/home/allierc/Py/NeuralGraph/log/{dataset_name}"
    output_dir = f"{base_dir}/nstm_output"
    activity_dir = f"{base_dir}/activity"
    activity_original_dir = f"{base_dir}/activity_original"  # For saving original before thresholding
    activity_neural_dir = f"{base_dir}/activity_neural"  # For NeuralRenderer-generated activity images
    motion_frames_dir = f"{base_dir}/motion_frames"


    # Parameters
    n_frames = 256
    res = 512
    target_downsample_factor = 1  # Downsample motion frame targets for super-resolution test (1 = no downsampling, 4 = quarter resolution, etc.)
    motion_intensity = 0.015  # Sinusoidal warp intensity (higher = more motion, better sub-pixel sampling for super-resolution)
    threshold_activity = False  # If True, train SIREN with constant activity value of 5 (demonstrates INR decomposition requires time-dependent activity)

    # Stage 5 options: Choose activity source for NSTM training
    # Option 1: use_neural_renderer_for_nstm = False, neural_renderer_learnable = False  --> Use matplotlib activity (Stage 2)
    # Option 2: use_neural_renderer_for_nstm = True,  neural_renderer_learnable = False  --> Use fixed NeuralRenderer (Stage 4)
    # Option 3: use_neural_renderer_for_nstm = True,  neural_renderer_learnable = True   --> Use learnable NeuralRenderer (Stage 4, LR=1e-6)
    use_neural_renderer_for_nstm = True  # If True, use NeuralRenderer (Stage 4). If False, use matplotlib (Stage 2)
    neural_renderer_learnable = True  # Only used if use_neural_renderer_for_nstm=True. If True, NeuralRenderer is learnable during NSTM training (LR=1e-6)

    os.makedirs(output_dir, exist_ok=True)
    if threshold_activity:
        print("NOTE: SIREN will be trained with CONSTANT activity")
        os.makedirs(activity_original_dir, exist_ok=True)

    # Load neuron data
    x_list = []
    y_list = []
    n_runs = 1
    for run in range(n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x[:,:,6] = np.clip((x[:,:,6] + 7.5) * 20, 0, 255) / 255
        x_list.append(x)
        y_list.append(y)

    run = 0
    x = x_list[0][n_frames - 10]
    n_neurons = x.shape[0]

    print(f"Activity range after shift: [{x[:, 3].min():.1f}, {x[:, 3].max():.1f}]")

    # SIREN config
    siren_config = {
        'num_training_steps': 10000,
        'nnr_f_T_period': 10,
        'n_train_frames': n_frames,
        'n_neurons': n_neurons
    }

    # Extract neuron positions
    neuron_positions = x_list[0][0][:, 1:3]

    # Load boat fixed scene
    import os as os_module
    current_dir = os_module.path.dirname(os_module.path.abspath(__file__))
    boat_fixed_scene_path = os_module.path.join(current_dir, 'pics_boat_512.tif')

    if os.path.exists(boat_fixed_scene_path):
        boat_fixed_scene = imread(boat_fixed_scene_path).astype(np.float32)
        print("boat: loaded high-res")
    else:
        boat_fixed_scene = np.ones((res, res), dtype=np.float32)
        print("boat: using default")

    print(f"boat: [{boat_fixed_scene.min():.2f}, {boat_fixed_scene.max():.2f}]")

    # Set matplotlib style for black background
    plt.style.use("dark_background")
    matplotlib.rcParams["savefig.pad_inches"] = 0

    # Clear existing files
    import glob
    for f in glob.glob(f'{motion_frames_dir}/*'):
        os.remove(f)
    for f in glob.glob(f'{activity_dir}/*'):
        os.remove(f)

    # Stage 1: Train SIREN
    siren_net = stage1_train_siren(x_list, device, output_dir, siren_config,
                                    use_constant_activity=threshold_activity)

    # Stage 2: Generate Activity Images
    activity_images, activity_images_original = stage2_generate_activity_images(
        x_list=x_list,
        neuron_positions=neuron_positions,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        activity_dir=activity_dir,
        activity_original_dir=activity_original_dir if threshold_activity else None,
        threshold_activity=threshold_activity,
        run=0
    )

    # Stage 3: Generate Warped Motion Frames
    # IMPORTANT: Use original (non-thresholded) activity for target generation
    # This creates a mismatch when training uses thresholded activity
    motion_input_activity = activity_images_original if activity_images_original is not None else activity_images
    motion_images = stage3_generate_warped_motion_frames(
        activity_images=motion_input_activity,
        boat_fixed_scene=boat_fixed_scene,
        n_frames=n_frames,
        output_dir=output_dir,
        motion_frames_dir=motion_frames_dir,
        target_downsample_factor=target_downsample_factor,
        motion_intensity=motion_intensity
    )

    # Stage 4: Train NeuralRenderer
    if use_neural_renderer_for_nstm:
        neural_renderer, activity_neural_images = stage4_train_neural_renderer(
        siren_net=siren_net,
        x_list=x_list,
        neuron_positions=neuron_positions,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        activity_dir=activity_dir,
        activity_neural_dir=activity_neural_dir,
        nnr_f_T_period=siren_config['nnr_f_T_period'],
        run=0
    )

    # Stage 5: Train NSTM
    # Choose activity source and neural_renderer based on flags
    if use_neural_renderer_for_nstm:
        nstm_activity_dir = activity_neural_dir
        nstm_neural_renderer = neural_renderer
        nstm_siren_net = siren_net
        nstm_neuron_positions = neuron_positions
    else:
        nstm_activity_dir = activity_dir
        nstm_neural_renderer = None
        nstm_siren_net = None
        nstm_neuron_positions = None

    deformation_net, fixed_scene_net, activity_net, loss_history = stage5_train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=nstm_activity_dir,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        siren_config=siren_config,
        pretrained_activity_net=None,
        x_list=None,
        boat_fixed_scene=boat_fixed_scene,
        boat_downsample_factor=1,  # No longer downsampling boat - it's high-res
        activity_images_original=activity_images_original,
        neural_renderer=nstm_neural_renderer,
        neural_renderer_learnable=neural_renderer_learnable,
        neuron_positions=nstm_neuron_positions,
        siren_net=nstm_siren_net
    )

