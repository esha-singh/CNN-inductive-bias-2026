"""
Network Architectures

Contains all network architectures used in experiments:
- SharedCNN: Parameter sharing across views (main experiments)
- SyntheticModel: For synthetic experiments with known ground truth
- Baseline architectures: FullyConnected, UnsharedCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Main Architecture: SharedCNN
# ============================================================================

class SharedCNN(nn.Module):
    """
    CNN with parameter sharing across local views.

    Architecture:
    - Shared filters applied to all views
    - Global average pooling across views
    - Linear readout layer

    Used in main experiments.
    """

    def __init__(self, d: int, m: int, K: int, stride: int):
        """
        Args:
            d: Ambient dimension
            m: Patch size
            K: Number of filters (width)
            stride: Stride for sliding window
        """
        super().__init__()
        self.d = d
        self.m = m
        self.K = K
        self.L = (d - m) // stride + 1
        self.stride = stride

        # Shared parameters across all views
        self.filter_weights = nn.Linear(m, K)
        self.biases = nn.Parameter(torch.randn(K))
        self.readout = nn.Linear(K, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, d)

        Returns:
            Output of shape (batch_size,)
        """
        batch_size = x.shape[0]
        view_outputs = []

        for ℓ in range(self.L):
            start = ℓ * self.stride
            end = start + self.m
            patch = x[:, start:end]

            # Same filters for all views (parameter sharing)
            h = self.filter_weights(patch) - self.biases
            h = F.relu(h)
            view_outputs.append(h)

        # Global average pooling
        pooled = torch.stack(view_outputs, dim=1).mean(dim=1)
        return self.readout(pooled).squeeze(-1)


# ============================================================================
# Synthetic Model (with arbitrary views)
# ============================================================================

class SyntheticModel(nn.Module):
    """
    Model network that learns the ground truth function.

    Same architecture as ground truth but with learnable parameters.
    Supports arbitrary view indices (not just sliding window).

    Used in synthetic experiments.
    """

    def __init__(self, d: int, K: int, H: np.ndarray):
        """
        Args:
            d: Ambient dimension
            K: Number of filters (width)
            H: View indices, shape (L, m)
        """
        super().__init__()
        self.d = d
        self.K = K
        self.H = torch.from_numpy(H).long()
        self.L = H.shape[0]
        self.m = H.shape[1]

        # Shared parameters
        # OLD INIT (had redundant bias in Linear + randn biases):
        # self.filter_weights = nn.Linear(self.m, K)  # Has bias=True by default
        # self.biases = nn.Parameter(torch.randn(K))  # N(0,1) - too large variance

        # NEW INIT (proper Kaiming He for weights, zero for biases):
        self.filter_weights = nn.Linear(self.m, K, bias=False)  # No redundant bias
        self.biases = nn.Parameter(torch.zeros(K))  # Standard: init bias to 0
        self.readout = nn.Linear(K, 1)

        # Apply Kaiming He initialization for ReLU
        nn.init.kaiming_normal_(self.filter_weights.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, d)

        Returns:
            Output of shape (batch_size,)
        """
        batch_size = x.shape[0]
        H_device = self.H.to(x.device)

        # Extract views: (batch_size, L, m)
        x_views = x[:, H_device]

        # Reshape for batch processing: (batch_size * L, m)
        x_flat = x_views.reshape(batch_size * self.L, self.m)

        # Apply filters: (batch_size * L, K)
        h_flat = self.filter_weights(x_flat) - self.biases
        h_flat = F.relu(h_flat)

        # Reshape back: (batch_size, L, K)
        h = h_flat.reshape(batch_size, self.L, self.K)

        # Global average pooling: (batch_size, K)
        pooled = h.mean(dim=1)

        # Readout: (batch_size,)
        return self.readout(pooled).squeeze(-1)


# ============================================================================
# Baseline Architectures (for comparison)
# ============================================================================

class FullyConnectedBaseline(nn.Module):
    """
    Fully-connected network for comparison.
    Suffers from curse of dimensionality.
    """

    def __init__(self, d: int, K: int):
        """
        Args:
            d: Ambient dimension
            K: Hidden layer width
        """
        super().__init__()
        self.d = d
        self.K = K
        self.fc1 = nn.Linear(d, K)
        self.fc2 = nn.Linear(K, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, d)

        Returns:
            Output of shape (batch_size,)
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class UnsharedCNN(nn.Module):
    """
    CNN with independent filters per view (no parameter sharing).
    For comparison with SharedCNN.
    """

    def __init__(self, d: int, m: int, K: int, stride: int):
        """
        Args:
            d: Ambient dimension
            m: Patch size
            K: Number of filters per view
            stride: Stride for sliding window
        """
        super().__init__()
        self.d = d
        self.m = m
        self.K = K
        self.L = (d - m) // stride + 1
        self.stride = stride

        # Separate parameters for each view!
        self.filters = nn.ModuleList([
            nn.Linear(m, K) for _ in range(self.L)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.randn(K)) for _ in range(self.L)
        ])
        self.readout = nn.Linear(K, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, d)

        Returns:
            Output of shape (batch_size,)
        """
        batch_size = x.shape[0]
        view_outputs = []

        for ℓ in range(self.L):
            start = ℓ * self.stride
            end = start + self.m
            patch = x[:, start:end]

            # Independent filters for this view
            h = self.filters[ℓ](patch) - self.biases[ℓ]
            h = F.relu(h)
            view_outputs.append(h)

        # Global average pooling
        pooled = torch.stack(view_outputs, dim=1).mean(dim=1)
        return self.readout(pooled).squeeze(-1)


class RandomCNN(nn.Module):
    """
    CNN with random local views (no structure, just random subsets).

    Each view samples m random coordinates from the d-dimensional input.
    Uses parameter sharing like SyntheticModel but views are random not structured.

    Architecture matches SyntheticModel/LocalViewNetwork exactly:
        f(x) = sum_k v_k * (1/L) * sum_l ReLU(W @ x_l - b_k) + beta

    For comparison: tests if structured views matter or if random views work.
    """

    def __init__(self, d: int, m: int, K: int, L: int, seed: int = 0):
        """
        Args:
            d: Ambient dimension
            m: Patch size (number of coordinates per view)
            K: Number of filters (width)
            L: Number of random views
            seed: Random seed for view generation
        """
        super().__init__()
        self.d = d
        self.m = m
        self.K = K
        self.L = L

        # Generate random views
        rng = np.random.default_rng(seed)
        H = []
        for _ in range(L):
            indices = np.sort(rng.choice(d, size=m, replace=False))
            H.append(indices)
        self.register_buffer('H', torch.from_numpy(np.array(H)).long())

        # Shared parameters - matching SyntheticModel/LocalViewNetwork exactly
        self.W = nn.Linear(m, K, bias=False)  # No bias in linear layer
        self.b = nn.Parameter(torch.zeros(K))  # Bias initialized to zeros
        self.v = nn.Parameter(torch.randn(K) / np.sqrt(K))  # Readout weights
        self.beta = nn.Parameter(torch.zeros(1))  # Global bias

        # Initialize W with Kaiming
        nn.init.kaiming_normal_(self.W.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, d)

        Returns:
            Output of shape (batch_size,)
        """
        batch_size = x.shape[0]

        # Extract random views: (batch_size, L, m)
        x_views = x[:, self.H]

        # Reshape for batch processing: (batch_size * L, m)
        x_flat = x_views.reshape(batch_size * self.L, self.m)

        # Apply shared filters: (batch_size * L, K)
        h_flat = self.W(x_flat) - self.b
        h_flat = F.relu(h_flat)

        # Reshape back: (batch_size, L, K)
        h = h_flat.reshape(batch_size, self.L, self.K)

        # Global average pooling: (batch_size, K)
        pooled = h.mean(dim=1)

        # Readout: (batch_size,)
        return (pooled * self.v).sum(dim=1) + self.beta


class UnsharedCNNWithH(nn.Module):
    """
    CNN with INDEPENDENT filters per view (NO parameter sharing).
    Takes arbitrary view indices H (like SyntheticModel).

    Architecture:
        f(x) = Σ_k v_k * (1/L) * Σ_l ReLU(W_l @ x_l - b_l_k) + β

    Key difference from SyntheticModel: Each view l has its OWN W_l and b_l
    This tests whether parameter sharing (weight tying) helps generalization.
    """

    def __init__(self, d: int, K: int, H: np.ndarray):
        """
        Args:
            d: Ambient dimension
            K: Number of filters (width)
            H: Local view indices, numpy array of shape (L, m)
        """
        super().__init__()
        self.d = d
        self.K = K
        self.L = H.shape[0]
        self.m = H.shape[1]

        # Register H as buffer
        self.register_buffer('H', torch.from_numpy(H).long())

        # INDEPENDENT parameters for each view (no sharing!)
        self.W_list = nn.ModuleList([
            nn.Linear(self.m, K, bias=False) for _ in range(self.L)
        ])
        self.b_list = nn.ParameterList([
            nn.Parameter(torch.zeros(K)) for _ in range(self.L)
        ])

        # Shared readout (same as SyntheticModel)
        self.v = nn.Parameter(torch.randn(K) / np.sqrt(K))
        self.beta = nn.Parameter(torch.zeros(1))

        # Initialize with Kaiming
        for W in self.W_list:
            nn.init.kaiming_normal_(W.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Input (batch_size, d)

        Returns:
            Output (batch_size,)
        """
        N = X.shape[0]

        # Process each view with its own parameters
        view_outputs = []
        for l in range(self.L):
            # Extract view l: (N, m)
            x_view = X[:, self.H[l]]

            # Apply view-specific filters: (N, K)
            h = self.W_list[l](x_view) - self.b_list[l]
            h = F.relu(h)
            view_outputs.append(h)

        # Stack and average pool: (N, K)
        pooled = torch.stack(view_outputs, dim=1).mean(dim=1)

        # Output: (N,)
        return (pooled * self.v).sum(dim=1) + self.beta


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_effective_rank(model: nn.Module, threshold: float = 0.01) -> int:
    """
    Estimate effective rank of learned filters.

    Args:
        model: Network model
        threshold: Singular value threshold (relative to max)

    Returns:
        Effective rank
    """
    if isinstance(model, (SharedCNN, SyntheticModel)):
        W = model.filter_weights.weight.data.cpu().numpy()
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        # Count singular values above threshold
        rank = np.sum(S > threshold * S[0])
        return int(rank)
    else:
        return -1  # Not applicable
