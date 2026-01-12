import torch
import torch.nn as nn
import numpy as np
from NeuralGraph.models.Siren_Network import Siren

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("Warning: tinycudann not available, falling back to standard PyTorch networks")


class Signal_Activity(nn.Module):
    """
    Neural Space-Time Model for Activity Decomposition

    This class encapsulates all learnable components for the instant-NGP based
    neural space-time model (NSTM) that decomposes video into:
    - Temporal activity (via SIREN network)
    - Spatial fixed scene (via tinycudann network)
    - Spatial-temporal deformation field (via tinycudann network)

    The model supports:
    1. SIREN network: t -> [activity_1, ..., activity_n] for discrete neuron activities
    2. Affine transform: learnable scale/bias to map SIREN outputs to image intensities
    3. Activity rendering: Gaussian splatting from discrete neurons to continuous image
    4. Fixed scene network: (x, y) -> scene_mask
    5. Deformation network: (x, y, t) -> (δx, δy)
    """

    def __init__(self, config=None, device=None):
        super(Signal_Activity, self).__init__()

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract configuration or use defaults
        if config is not None and hasattr(config, 'graph_model'):
            model_config = config.graph_model
            self._init_from_config(model_config)
        else:
            self._init_defaults()

        # Build networks
        self._build_siren()
        self._build_affine_transform()
        self._build_deformation_network()
        self._build_fixed_scene_network()

    def _init_from_config(self, model_config):
        """Initialize parameters from config"""
        # SIREN parameters
        self.input_size_nnr_f = getattr(model_config, 'input_size_nnr_f', 1)
        self.output_size_nnr_f = getattr(model_config, 'output_size_nnr_f', 100)
        self.hidden_dim_nnr_f = getattr(model_config, 'hidden_dim_nnr_f', 256)
        self.n_layers_nnr_f = getattr(model_config, 'n_layers_nnr_f', 4)
        self.omega_f = getattr(model_config, 'omega_f', 30)
        self.outermost_linear_nnr_f = getattr(model_config, 'outermost_linear_nnr_f', True)
        self.nnr_f_T_period = getattr(model_config, 'nnr_f_T_period', 10)

        # Affine transform initialization
        self.affine_scale_init = getattr(model_config, 'affine_scale_init', 3.0)
        self.affine_bias_init = getattr(model_config, 'affine_bias_init', 60.0)

        # Activity rendering parameters
        self.dot_size = getattr(model_config, 'dot_size', 32)
        self.image_size = getattr(model_config, 'image_size', 512)

        # InstantNGP encoding/network configs (for deformation and fixed_scene)
        self.encoding_config = getattr(model_config, 'encoding_config', self._default_encoding_config())
        self.network_config = getattr(model_config, 'network_config', self._default_network_config())

    def _init_defaults(self):
        """Initialize with hardcoded defaults"""
        # SIREN defaults (temporal activity network)
        self.input_size_nnr_f = 1  # time only
        self.output_size_nnr_f = 100  # 100 neurons
        self.hidden_dim_nnr_f = 256
        self.n_layers_nnr_f = 4
        self.omega_f = 30  # lower frequency for temporal signals
        self.outermost_linear_nnr_f = True
        self.nnr_f_T_period = 10

        # Affine transform defaults
        self.affine_scale_init = 3.0
        self.affine_bias_init = 60.0

        # Activity rendering defaults
        self.dot_size = 32  # pixels
        self.image_size = 512

        # InstantNGP defaults
        self.encoding_config = self._default_encoding_config()
        self.network_config = self._default_network_config()

    def _default_encoding_config(self):
        """Default hash encoding for tinycudann"""
        return {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5
        }

    def _default_network_config(self):
        """Default MLP config for tinycudann"""
        return {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        }

    def _build_siren(self):
        """Build SIREN network for temporal activity: t -> [activity_1, ..., activity_n]"""
        self.siren_net = Siren(
            in_features=self.input_size_nnr_f,
            out_features=self.output_size_nnr_f,
            hidden_features=self.hidden_dim_nnr_f,
            hidden_layers=self.n_layers_nnr_f,
            outermost_linear=self.outermost_linear_nnr_f,
            first_omega_0=60,
            hidden_omega_0=self.omega_f
        )
        self.siren_net.to(self.device)

    def _build_affine_transform(self):
        """Build learnable affine transform: y = scale * x + bias"""
        self.affine_scale = nn.Parameter(
            torch.tensor(self.affine_scale_init, device=self.device, dtype=torch.float32)
        )
        self.affine_bias = nn.Parameter(
            torch.tensor(self.affine_bias_init, device=self.device, dtype=torch.float32)
        )

    def _build_deformation_network(self):
        """Build deformation network: (x, y, t) -> (δx, δy)"""
        if TCNN_AVAILABLE:
            self.deformation_net = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,  # x, y, t
                n_output_dims=2,  # δx, δy
                encoding_config=self.encoding_config,
                network_config=self.network_config
            ).to(self.device)
        else:
            # Fallback to standard MLP if tinycudann not available
            from NeuralGraph.models.MLP import MLP
            self.deformation_net = MLP(
                input_size=3,
                output_size=2,
                nlayers=3,
                hidden_size=64,
                activation='relu',
                device=self.device
            )

    def _build_fixed_scene_network(self):
        """Build fixed scene network: (x, y) -> mask_value"""
        if TCNN_AVAILABLE:
            self.fixed_scene_net = tcnn.NetworkWithInputEncoding(
                n_input_dims=2,  # x, y
                n_output_dims=1,  # mask value
                encoding_config=self.encoding_config,
                network_config=self.network_config
            ).to(self.device)
        else:
            # Fallback to standard MLP if tinycudann not available
            from NeuralGraph.models.MLP import MLP
            self.fixed_scene_net = MLP(
                input_size=2,
                output_size=1,
                nlayers=3,
                hidden_size=64,
                activation='relu',
                device=self.device
            )

    # ========== Forward Methods ==========

    def forward_siren(self, t):
        """
        Query SIREN network for neuron activities at time t

        Args:
            t: (batch, 1) tensor of normalized time values in [0, 1]

        Returns:
            (batch, n_neurons) tensor of activity values
        """
        # Convert normalized time to SIREN input (with period scaling)
        t_siren = t * (2 * np.pi / self.nnr_f_T_period)
        activities = self.siren_net(t_siren)

        # Apply affine transform
        activities = self.affine_scale * activities + self.affine_bias

        return activities

    def forward_deformation(self, coords_3d):
        """
        Query deformation network for motion field

        Args:
            coords_3d: (N, 3) tensor of [x, y, t] coordinates in [0, 1]

        Returns:
            (N, 2) tensor of deformation vectors [δx, δy]
        """
        if TCNN_AVAILABLE:
            # tinycudann requires float16
            coords_f16 = coords_3d.to(torch.float16)
            deformation = self.deformation_net(coords_f16).to(torch.float32)
        else:
            deformation = self.deformation_net(coords_3d)

        return deformation

    def forward_fixed_scene(self, coords_2d):
        """
        Query fixed scene network for scene mask

        Args:
            coords_2d: (N, 2) tensor of [x, y] coordinates in [0, 1]

        Returns:
            (N, 1) tensor of mask values (apply sigmoid for [0,1] range)
        """
        if TCNN_AVAILABLE:
            # tinycudann requires float16
            coords_f16 = coords_2d.to(torch.float16)
            mask = self.fixed_scene_net(coords_f16).to(torch.float32)
        else:
            mask = self.fixed_scene_net(coords_2d)

        return mask

    # ========== Activity Rendering ==========

    def render_activity_at_coords(self, coords_2d, t_normalized, neuron_positions,
                                    use_uniform_kernel=False, kernel_width=0.05):
        """
        Render activity at arbitrary 2D coordinates using kernel splatting

        Args:
            coords_2d: (N, 2) tensor of 2D coordinates in [0, 1]
            t_normalized: scalar, normalized time value in [0, 1]
            neuron_positions: (n_neurons, 2) tensor of neuron positions in [-0.5, 0.5]
            use_uniform_kernel: if True, use uniform kernel instead of Gaussian
            kernel_width: width parameter for uniform kernel (in [0,1] space)

        Returns:
            (N,) tensor of activity values at the queried coordinates
        """
        # Query SIREN for neuron activities at this time
        t_tensor = torch.tensor([[t_normalized]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            neuron_activities = self.forward_siren(t_tensor)[0]  # (n_neurons,)

        if use_uniform_kernel:
            # Uniform kernel: denser and more uniform than Gaussian
            splatted = self._splat_uniform(coords_2d, neuron_positions, neuron_activities, kernel_width)
        else:
            # Gaussian kernel (original implementation)
            splatted = self._splat_gaussian(coords_2d, neuron_positions, neuron_activities)

        return splatted

    def _splat_gaussian(self, coords_2d, neuron_positions, neuron_activities):
        """Gaussian splatting (original method)"""
        # Convert to pixel space
        coords_pixels = coords_2d * (self.image_size - 1)
        neuron_positions_pixels = (neuron_positions + 0.5) * (self.image_size - 1)

        # Gaussian kernel
        sigma = self.dot_size / 2.355
        diffs = coords_pixels.unsqueeze(1) - neuron_positions_pixels.unsqueeze(0)
        dist_sq = torch.sum(diffs ** 2, dim=2)
        gaussian_weights = torch.exp(-dist_sq / (2 * sigma ** 2))

        # Weighted sum
        splatted = torch.sum(gaussian_weights * neuron_activities.unsqueeze(0), dim=1)
        return splatted

    def _splat_uniform(self, coords_2d, neuron_positions, neuron_activities, kernel_width):
        """
        Uniform kernel splatting: denser and more uniform than Gaussian

        Uses a uniform kernel: weight = 1 if distance < threshold, else 0
        This creates denser, more uniform coverage compared to Gaussian decay
        """
        # Compute distances in [0,1] normalized space
        diffs = coords_2d.unsqueeze(1) - neuron_positions.unsqueeze(0)  # (N, n_neurons, 2)
        distances = torch.norm(diffs, dim=2)  # (N, n_neurons)

        # Uniform kernel: 1 inside radius, 0 outside
        weights = (distances < kernel_width).float()

        # Normalize weights (so each point gets weighted average)
        weight_sum = weights.sum(dim=1, keepdim=True) + 1e-8
        weights = weights / weight_sum

        # Weighted sum
        splatted = torch.sum(weights * neuron_activities.unsqueeze(0), dim=1)
        return splatted

    def render_activity_image(self, t_normalized, neuron_positions, res=None,
                              use_uniform_kernel=False, kernel_width=0.05):
        """
        Render full activity image at time t

        Args:
            t_normalized: scalar, normalized time value in [0, 1]
            neuron_positions: (n_neurons, 2) tensor of neuron positions in [-0.5, 0.5]
            res: image resolution (default: self.image_size)
            use_uniform_kernel: if True, use uniform kernel instead of Gaussian
            kernel_width: width parameter for uniform kernel

        Returns:
            (res, res) numpy array of rendered activity image
        """
        if res is None:
            res = self.image_size

        # Create coordinate grid
        y_coords = torch.linspace(0, 1, res, device=self.device, dtype=torch.float32)
        x_coords = torch.linspace(0, 1, res, device=self.device, dtype=torch.float32)
        yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

        # Render activity
        with torch.no_grad():
            activity = self.render_activity_at_coords(
                coords_2d, t_normalized, neuron_positions,
                use_uniform_kernel=use_uniform_kernel,
                kernel_width=kernel_width
            )

        return activity.reshape(res, res).cpu().numpy()

    # ========== Utility Methods ==========

    def get_siren_parameters(self):
        """Return SIREN network parameters for separate optimization"""
        return list(self.siren_net.parameters())

    def get_nstm_parameters(self):
        """Return NSTM parameters (deformation + fixed_scene + affine)"""
        params = (list(self.deformation_net.parameters()) +
                 list(self.fixed_scene_net.parameters()) +
                 [self.affine_scale, self.affine_bias])
        return params

    def get_affine_values(self):
        """Return current affine transform values"""
        return {
            'scale': self.affine_scale.item(),
            'bias': self.affine_bias.item()
        }

    def freeze_siren(self):
        """Freeze SIREN network (for NSTM training)"""
        for param in self.siren_net.parameters():
            param.requires_grad = False

    def unfreeze_siren(self):
        """Unfreeze SIREN network"""
        for param in self.siren_net.parameters():
            param.requires_grad = True
