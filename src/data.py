"""
Data Generation Module

Handles all data generation for experiments:
- Sphere sampling
- Label generation (smooth, binary)
- Ground truth functions
- Synthetic datasets
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Callable

# Detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Sphere Sampling
# ============================================================================

def generate_sphere_data(n: int, d: int, seed: int = None) -> torch.Tensor:
    """
    Generate n points uniformly on unit sphere S^(d-1).

    Args:
        n: Number of points
        d: Ambient dimension
        seed: Random seed

    Returns:
        X: Tensor of shape (n, d) on unit sphere
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(n, d)
    return F.normalize(x, dim=1)


def generate_sphere_data_fast(n: int, d: int, seed: int = None, device='cuda') -> torch.Tensor:
    """
    Generate directly on GPU - much faster!

    Args:
        n: Number of points
        d: Ambient dimension
        seed: Random seed
        device: Device to generate on

    Returns:
        X: Tensor of shape (n, d) on unit sphere
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(n, d, device=device)
    return F.normalize(x, dim=1)


# ============================================================================
# Label Generation (Main Experiments)
# ============================================================================

def generate_labels_smooth(x: torch.Tensor, freq: int = 3, sigma: float = 0.0) -> torch.Tensor:
    """
    Generate smooth labels on sphere for regression with optional Gaussian noise.

    Target function: y = tanh(sum(x * weights)) + sigma * epsilon
    where weights = cos(linspace(0, freq*pi, d))

    Args:
        x: Input points on sphere (n, d)
        freq: Frequency parameter for the target function
        sigma: Noise level (standard deviation). Default 0.0 = no noise.

    Returns:
        y: Labels in [-1, 1] with added Gaussian noise if sigma > 0
    """
    d = x.shape[1]
    # Create weights on same device as x
    weights = torch.cos(torch.linspace(0, freq * np.pi, d, device=x.device))
    scores = (x * weights).sum(dim=1)
    clean_labels = torch.tanh(scores)

    # Add Gaussian noise if requested
    if sigma > 0:
        noise = torch.randn_like(clean_labels) * sigma
        noisy_labels = clean_labels + noise
        return noisy_labels
    else:
        return clean_labels


def generate_labels_binary(x: torch.Tensor) -> torch.Tensor:
    """
    Binary classification based on hemisphere.

    Args:
        x: Input points on sphere (n, d)

    Returns:
        y: Labels in {-1, +1}
    """
    return (x[:, 0] > 0).float() * 2 - 1


# ============================================================================
# Ground Truth Function (Synthetic Experiments)
# ============================================================================

def create_ground_truth_function(d: int, K: int, H: np.ndarray, seed: int) -> Callable:
    """
    Create the ground truth function (complex CNN).

    The function has the form:
    f(x) = v^T · (1/L) · Σ_l ReLU(W · π_l(x) - b) + β

    Args:
        d: Ambient dimension
        K: Ground truth network width (complexity)
        H: Local view indices, shape (L, m)
        seed: Random seed for parameter initialization

    Returns:
        Callable that maps torch.Tensor (N, d) → torch.Tensor (N,)
    """
    rng = np.random.default_rng(seed)
    m = H.shape[1]
    L = H.shape[0]

    # Initialize ground truth parameters
    v = torch.from_numpy(rng.normal(0, 1, K)).float()
    W = torch.from_numpy(rng.normal(0, 1, (K, m))).float()
    b = torch.from_numpy(rng.uniform(-0.5, 0.5, K)).float()
    beta = torch.tensor(rng.uniform(-1, 1)).float()
    H_tensor = torch.from_numpy(H).long()

    def groundtruth_function(X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate ground truth on batch X.

        Args:
            X: Input tensor of shape (N, d)

        Returns:
            Output tensor of shape (N,)
        """
        device = X.device
        v_dev = v.to(device)
        W_dev = W.to(device)
        b_dev = b.to(device)
        beta_dev = beta.to(device)
        H_dev = H_tensor.to(device)

        N = X.shape[0]

        # Extract local views: (N, L, m)
        X_views = X[:, H_dev]

        # Reshape for batch computation
        X_flat = X_views.reshape(N * L, m)

        # Apply filters: (N*L, m) @ (m, K) -> (N*L, K)
        hidden_flat = X_flat @ W_dev.T - b_dev
        hidden_flat = F.relu(hidden_flat)

        # Reshape back: (N, L, K)
        hidden = hidden_flat.reshape(N, L, K)

        # Global average pooling: (N, K)
        pooled = hidden.mean(dim=1)

        # Final linear layer: (N, K) @ (K,) -> (N,)
        output = (pooled * v_dev).sum(dim=1) + beta_dev

        return output

    return groundtruth_function


# ============================================================================
# Dataset Generation (Synthetic Experiments)
# ============================================================================

def generate_synthetic_dataset(
    d: int,
    K_true: int,
    H: np.ndarray,
    N_train: int,
    N_test: int,
    sigma: float,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Callable]:
    """
    Generate complete synthetic dataset with known ground truth.

    Args:
        d: Ambient dimension
        K_true: Ground truth network width
        H: View indices, shape (L, m)
        N_train: Training samples
        N_test: Test samples
        sigma: Noise level
        seed: Random seed

    Returns:
        X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean, groundtruth_fn
    """
    rng = np.random.default_rng(seed)

    # Generate training data
    X_train_np = rng.normal(0, 1, (N_train, d))
    X_train_np = X_train_np / np.linalg.norm(X_train_np, axis=1, keepdims=True)
    X_train = torch.from_numpy(X_train_np).float()

    # Generate test data
    X_test_np = rng.normal(0, 1, (N_test, d))
    X_test_np = X_test_np / np.linalg.norm(X_test_np, axis=1, keepdims=True)
    X_test = torch.from_numpy(X_test_np).float()

    # Create ground truth function
    groundtruth_fn = create_ground_truth_function(d, K_true, H, seed)

    # Generate clean labels
    with torch.no_grad():
        y_train_clean = groundtruth_fn(X_train)
        y_test_clean = groundtruth_fn(X_test)

    # Add noise
    noise_train = torch.from_numpy(rng.normal(0, sigma, N_train)).float()
    noise_test = torch.from_numpy(rng.normal(0, sigma, N_test)).float()

    y_train_noisy = y_train_clean + noise_train
    y_test_noisy = y_test_clean + noise_test

    return X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean, groundtruth_fn


def generate_identical_patches_data(
    m: int,
    L: int,
    K_true: int,
    N: int,
    sigma: float,
    seed: int,
    H: np.ndarray = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable, np.ndarray]:
    """
    Generate data where all L patches are IDENTICAL.

    Input structure: X = [x_patch, x_patch, ..., x_patch]
    where x_patch ∈ S^(m-1)

    Args:
        m: Patch size
        L: Number of repetitions
        K_true: Ground truth width
        N: Number of samples
        sigma: Noise level
        seed: Random seed
        H: View indices (if None, use disjoint blocks)

    Returns:
        X, y_clean, y_noisy, groundtruth_fn, H
    """
    d = L * m
    rng = np.random.default_rng(seed)

    # Generate disjoint views if not provided
    if H is None:
        H = np.array([np.arange(i*m, (i+1)*m) for i in range(L)])

    # Create ground truth
    groundtruth_fn = create_ground_truth_function(d, K_true, H, seed)

    # Generate X with identical patches
    X_list = []
    for i in range(N):
        # Generate ONE patch
        patch = rng.normal(0, 1, m)
        patch = patch / np.linalg.norm(patch)

        # Replicate L times
        x = np.tile(patch, L)
        X_list.append(x)

    X_np = np.array(X_list)
    X = torch.from_numpy(X_np).float()

    # Generate labels
    with torch.no_grad():
        y_clean = groundtruth_fn(X)

    noise = torch.from_numpy(rng.normal(0, sigma, N)).float()
    y_noisy = y_clean + noise

    return X, y_clean, y_noisy, groundtruth_fn, H


def generate_augmented_views(
    d: int,
    m: int,
    L: int,
    R: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate augmented views: L disjoint + R random.

    Args:
        d: Ambient dimension
        m: Patch size
        L: Number of disjoint blocks
        R: Number of random views to add
        seed: Random seed

    Returns:
        H_combined, H_disjoint, H_random
    """
    rng = np.random.default_rng(seed)

    # Generate L disjoint blocks
    H_disjoint = np.array([np.arange(i*m, (i+1)*m) for i in range(L)])

    # Generate R random views
    H_random = []
    for _ in range(R):
        indices = np.sort(rng.choice(d, size=m, replace=False))
        H_random.append(indices)
    H_random = np.array(H_random) if R > 0 else np.zeros((0, m), dtype=int)

    # Combine
    if R > 0:
        H_combined = np.vstack([H_disjoint, H_random])
    else:
        H_combined = H_disjoint

    return H_combined, H_disjoint, H_random
