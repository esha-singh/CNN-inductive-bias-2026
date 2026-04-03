"""
Synthetic Experiments with Known Ground Truth

This module implements experiments where:
- Ground truth is a known CNN-like function
- Model network tries to learn it
- Multiple view generation strategies are compared
- Compatible with the main experimental framework

Key differences from main experiments:
- Target function is a complex CNN (not simple tanh)
- Ground truth parameters are known
- Realizable learning problem
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Try to import from experimental framework
try:
    from experimental_framework import (
        check_convergence,
        check_boundary_condition,
        compute_theoretical_alpha,
        DEVICE
    )
except ImportError:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def check_convergence(losses, window=10, threshold=0.001):
        if len(losses) < window:
            return False, None
        recent = losses[-window:]
        mean = np.mean(recent)
        std = np.std(recent)
        if mean == 0:
            return False, None
        rel_std = std / abs(mean)
        if rel_std < threshold:
            return True, len(losses) - window
        return False, None


# ============================================================================
# View Generation Strategies
# ============================================================================

class ViewStrategy(Enum):
    """Different strategies for generating local views"""
    SLIDING_WINDOW = "sliding_window"      # Your original: overlapping stride-based
    RANDOM_SUBSETS = "random_subsets"      # Toy code: random m indices
    DISJOINT_BLOCKS = "disjoint_blocks"    # Non-overlapping consecutive blocks
    RANDOM_OVERLAPPING = "random_overlap"  # Random positions with guaranteed overlap
    HIERARCHICAL = "hierarchical"          # Multi-scale patches
    DENSE_SLIDING = "dense_sliding"        # Stride=1 (maximum overlap)


def generate_views_sliding_window(d: int, m: int, stride: int, seed: int = None) -> np.ndarray:
    """
    Sliding window views (like your SharedCNN).

    Creates overlapping views by sliding a window of size m with given stride.

    Args:
        d: Ambient dimension
        m: Patch size
        stride: Step size between consecutive patches
        seed: Random seed (unused, for API consistency)

    Returns:
        H: Array of shape (L, m) with consecutive indices
    """
    H = []
    start = 0
    while start + m <= d:
        H.append(np.arange(start, start + m))
        start += stride

    if len(H) == 0:
        raise ValueError(f"Cannot create views: d={d}, m={m}, stride={stride}")

    return np.array(H)


def generate_views_random_subsets(d: int, m: int, L: int, seed: int) -> np.ndarray:
    """
    Random subset views (like toy_experiment_7.py).

    Each view selects m random coordinates from d dimensions.
    Views can overlap arbitrarily.

    Args:
        d: Ambient dimension
        m: Patch size
        L: Number of views to generate
        seed: Random seed

    Returns:
        H: Array of shape (L, m) with random sorted indices
    """
    rng = np.random.default_rng(seed)
    H = [np.sort(rng.choice(d, size=m, replace=False)) for _ in range(L)]
    return np.array(H)


def generate_views_disjoint_blocks(d: int, m: int, seed: int = None) -> np.ndarray:
    """
    Disjoint consecutive blocks with no overlap.

    Divides the input into L = d//m non-overlapping blocks.

    Args:
        d: Ambient dimension (should be divisible by m ideally)
        m: Patch size
        seed: Random seed (unused, for API consistency)

    Returns:
        H: Array of shape (L, m) with disjoint consecutive blocks
    """
    L = d // m
    if L == 0:
        raise ValueError(f"Cannot create disjoint blocks: d={d}, m={m}")

    H = [np.arange(i * m, (i + 1) * m) for i in range(L)]
    return np.array(H)


def generate_views_random_overlapping(d: int, m: int, L: int, overlap_ratio: float, seed: int) -> np.ndarray:
    """
    Random starting positions with controlled overlap.

    Args:
        d: Ambient dimension
        m: Patch size
        L: Number of views
        overlap_ratio: Target overlap ratio (0=disjoint, 1=fully overlapping)
        seed: Random seed

    Returns:
        H: Array of shape (L, m)
    """
    rng = np.random.default_rng(seed)

    # Compute stride to achieve target overlap
    effective_stride = max(1, int(m * (1 - overlap_ratio)))
    max_start = d - m

    H = []
    for _ in range(L):
        start = rng.integers(0, max_start + 1)
        H.append(np.arange(start, start + m))

    return np.array(H)


def generate_views_hierarchical(d: int, m: int, num_scales: int, seed: int) -> np.ndarray:
    """
    Multi-scale hierarchical views.

    Creates views at different scales (m, 2m, 4m, ...).

    Args:
        d: Ambient dimension
        m: Base patch size
        num_scales: Number of scales to use
        seed: Random seed

    Returns:
        H: Array of shape (L, m_max) padded with -1 for smaller patches
    """
    rng = np.random.default_rng(seed)
    H = []

    for scale in range(num_scales):
        patch_size = m * (2 ** scale)
        if patch_size > d:
            break

        # Create a few views at this scale
        num_views = max(1, (d - patch_size) // patch_size + 1)
        for i in range(num_views):
            start = i * patch_size
            if start + patch_size <= d:
                indices = np.arange(start, start + patch_size)
                # Pad to consistent size
                padded = np.full(d, -1)
                padded[:len(indices)] = indices
                H.append(padded[:m * (2 ** (num_scales - 1))])

    return np.array(H) if H else generate_views_sliding_window(d, m, m, seed)


def generate_views_dense_sliding(d: int, m: int, seed: int = None) -> np.ndarray:
    """
    Dense sliding window with stride=1 (maximum overlap).

    Args:
        d: Ambient dimension
        m: Patch size
        seed: Random seed (unused)

    Returns:
        H: Array of shape (L, m) with maximum overlap
    """
    return generate_views_sliding_window(d, m, stride=1, seed=seed)


def generate_views(d: int, m: int, strategy: ViewStrategy,
                   L: Optional[int] = None, stride: Optional[int] = None,
                   overlap_ratio: float = 0.5, num_scales: int = 3,
                   seed: int = 42) -> np.ndarray:
    """
    Unified interface for generating views with different strategies.

    Args:
        d: Ambient dimension
        m: Patch size
        strategy: View generation strategy
        L: Number of views (for strategies that need it)
        stride: Stride (for sliding window strategies)
        overlap_ratio: Overlap ratio (for random_overlapping)
        num_scales: Number of scales (for hierarchical)
        seed: Random seed

    Returns:
        H: Array of shape (L, m) containing view indices
    """
    if strategy == ViewStrategy.SLIDING_WINDOW:
        if stride is None:
            stride = max(1, m // 2)
        return generate_views_sliding_window(d, m, stride, seed)

    elif strategy == ViewStrategy.RANDOM_SUBSETS:
        if L is None:
            L = (d - m) // max(1, m // 2) + 1  # Default to similar count as sliding
        return generate_views_random_subsets(d, m, L, seed)

    elif strategy == ViewStrategy.DISJOINT_BLOCKS:
        return generate_views_disjoint_blocks(d, m, seed)

    elif strategy == ViewStrategy.RANDOM_OVERLAPPING:
        if L is None:
            L = (d - m) // max(1, m // 2) + 1
        return generate_views_random_overlapping(d, m, L, overlap_ratio, seed)

    elif strategy == ViewStrategy.HIERARCHICAL:
        return generate_views_hierarchical(d, m, num_scales, seed)

    elif strategy == ViewStrategy.DENSE_SLIDING:
        return generate_views_dense_sliding(d, m, seed)

    else:
        raise ValueError(f"Unknown view strategy: {strategy}")


# ============================================================================
# Ground Truth Function
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

        # Reshape and pool: (N, L, K) -> (N, K)
        hidden = hidden_flat.reshape(N, L, K)
        pooled = hidden.mean(dim=1)

        # Final linear layer: (N, K) @ (K,) -> (N,)
        output = (pooled * v_dev).sum(dim=1) + beta_dev

        return output

    return groundtruth_function


# ============================================================================
# Data Generation
# ============================================================================

def generate_synthetic_dataset(
    d: int,
    K_true: int,
    m: int,
    H: np.ndarray,
    N: int,
    sigma: float,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]:
    """
    Generate synthetic dataset with teacher-student setup.

    Args:
        d: Ambient dimension
        K_true: Teacher network width
        m: Patch size
        H: Local view indices
        N: Number of samples
        sigma: Noise level
        seed: Random seed

    Returns:
        X: Input features (N, d)
        y_clean: Clean labels from teacher (N,)
        y_noisy: Noisy labels for training (N,)
        groundtruth_fn: The teacher function
    """
    rng = np.random.default_rng(seed)

    # Generate inputs on unit sphere (same as your main experiments)
    X_np = rng.normal(0, 1, (N, d))
    X_np = X_np / np.linalg.norm(X_np, axis=1, keepdims=True)
    X = torch.from_numpy(X_np).float()

    # Create teacher function
    groundtruth_fn = create_ground_truth_function(d, K_true, H, seed)

    # Generate clean labels
    with torch.no_grad():
        y_clean = groundtruth_fn(X)

    # Add noise
    noise = torch.from_numpy(rng.normal(0, sigma, N)).float()
    y_noisy = y_clean + noise

    return X, y_clean, y_noisy, groundtruth_fn


# ============================================================================
# Model Network Architecture
# ============================================================================

class SyntheticModel(nn.Module):
    """
    Model network that learns the ground truth function.

    Same architecture as ground truth but with learnable parameters.
    """

    def __init__(self, d: int, K: int, H: np.ndarray):
        super().__init__()

        self.d = d
        self.K = K
        self.m = H.shape[1]
        self.L = H.shape[0]

        # Register H as buffer
        self.register_buffer('H', torch.from_numpy(H).long())

        # Learnable parameters
        self.W = nn.Linear(self.m, K, bias=False)
        self.b = nn.Parameter(torch.zeros(K))
        self.v = nn.Parameter(torch.randn(K) / np.sqrt(K))
        self.beta = nn.Parameter(torch.zeros(1))

        # Initialize with Kaiming
        nn.init.kaiming_normal_(self.W.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Input (batch_size, d)

        Returns:
            Output (batch_size,)
        """
        N = X.shape[0]

        # Extract views: (N, L, m)
        X_views = X[:, self.H]

        # Reshape: (N*L, m)
        X_flat = X_views.reshape(N * self.L, self.m)

        # Apply filters: (N*L, K)
        hidden_flat = self.W(X_flat) - self.b
        hidden_flat = F.relu(hidden_flat)

        # Reshape and pool: (N, K)
        hidden = hidden_flat.reshape(N, self.L, self.K)
        pooled = hidden.mean(dim=1)

        # Output: (N,)
        output = (pooled * self.v).sum(dim=1) + self.beta

        return output


# ============================================================================
# Training Utilities
# ============================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test_noisy: torch.Tensor,
    y_test_clean: torch.Tensor,
    y_train_clean: torch.Tensor = None,
    lr: float = 0.01,
    epochs: int = 1000,
    device: str = 'cuda',
    log_interval: int = 50,
    verbose: bool = True,
    gradient_clip: float = 100.0,
    early_stopping: bool = False,
    grad_norm_threshold: float = 1e-6
) -> Dict[str, List[float]]:
    """
    Train model network.

    Args:
        model: Model network
        X_train: Training inputs
        y_train: Training labels (noisy)
        X_test: Test inputs
        y_test_noisy: Test labels (noisy)
        y_test_clean: Test labels (clean, from ground truth)
        y_train_clean: Training labels (clean), optional for tracking train MSE clean
        lr: Learning rate
        epochs: Number of epochs
        device: Device to train on
        log_interval: Print logs every N epochs (0 to disable)
        verbose: Whether to print epoch logs
        gradient_clip: Max gradient norm for clipping (prevents exploding gradients)
        early_stopping: If True, stop when gradient norm < grad_norm_threshold
        grad_norm_threshold: Threshold for early stopping (default: 1e-6)

    Returns:
        Dictionary with training history including gradient norms
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test_noisy = y_test_noisy.to(device)
    y_test_clean = y_test_clean.to(device)
    if y_train_clean is not None:
        y_train_clean = y_train_clean.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'train_loss_clean': [],
        'test_loss_noisy': [],
        'test_loss_clean': [],
        'gen_gap': [],
        'grad_norm': [],
        'convergence_epoch': None  # Epoch where grad_norm first drops below threshold
    }

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train)
        loss_train = criterion(pred_train, y_train)
        loss_train.backward()

        # Compute gradient norm (every epoch for tracking)
        total_grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm_sq += param.grad.norm().item() ** 2
        current_grad_norm = np.sqrt(total_grad_norm_sq)
        history['grad_norm'].append(current_grad_norm)

        # Track convergence epoch (first time grad_norm drops below threshold)
        if history['convergence_epoch'] is None and current_grad_norm < grad_norm_threshold:
            history['convergence_epoch'] = epoch
            if verbose:
                print(f"    [CONVERGENCE] Gradient norm ({current_grad_norm:.2e}) < threshold ({grad_norm_threshold:.2e}) at epoch {epoch}")

        # Debug: Check gradients on first epoch
        if epoch == 0 and verbose:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"    [DEBUG] {name}: grad_norm={grad_norm:.6f}, requires_grad={param.requires_grad}")
                else:
                    print(f"    [DEBUG] {name}: grad=None, requires_grad={param.requires_grad}")
            print(f"    [DEBUG] Total grad norm: {current_grad_norm:.6f}")
            # Save param values before step
            if hasattr(model, 'beta'):
                beta_before = model.beta.item()

        # Gradient clipping to prevent exploding gradients
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Debug: Check if parameters actually changed
        if epoch == 0 and verbose and hasattr(model, 'beta'):
            beta_after = model.beta.item()
            print(f"    [DEBUG] beta before step: {beta_before:.6f}, after step: {beta_after:.6f}")
            print(f"    [DEBUG] lr={lr}")
            # Verify loss changes with new params
            with torch.no_grad():
                pred_new = model(X_train)
                loss_new = criterion(pred_new, y_train)
            print(f"    [DEBUG] loss BEFORE step: {loss_train.item():.6f}, loss AFTER step: {loss_new.item():.6f}")

        # Early stopping check (only if enabled)
        if early_stopping and history['convergence_epoch'] is not None:
            if verbose:
                print(f"    [EARLY STOP] Stopping at epoch {epoch} (converged at {history['convergence_epoch']})")
            break

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test)
            loss_test_noisy = criterion(pred_test, y_test_noisy)
            loss_test_clean = criterion(pred_test, y_test_clean)

            # Compute train MSE (clean) if y_train_clean provided
            if y_train_clean is not None:
                loss_train_clean = criterion(pred_train, y_train_clean)
            else:
                loss_train_clean = loss_train  # fallback to noisy

        history['train_loss'].append(loss_train.item())
        history['train_loss_clean'].append(loss_train_clean.item())
        history['test_loss_noisy'].append(loss_test_noisy.item())
        history['test_loss_clean'].append(loss_test_clean.item())
        history['gen_gap'].append(abs(loss_test_noisy.item() - loss_train.item()))

        # Logging
        if verbose and log_interval > 0 and (epoch + 1) % log_interval == 0:
            log_msg = f"    Epoch {epoch+1:5d}/{epochs}: Train Loss(noisy)={loss_train.item():.6f}"
            if y_train_clean is not None:
                log_msg += f", Train MSE(clean)={loss_train_clean.item():.6f}"
            log_msg += f", Test MSE(clean)={loss_test_clean.item():.6f}"
            print(log_msg)

        # Debug: Check ReLU health on epoch 50 (only for models with W attribute)
        if verbose and epoch == 49 and hasattr(model, 'W'):
            model.eval()
            with torch.no_grad():
                # Check activations manually
                X_views = X_train[:, model.H]
                X_flat = X_views.reshape(-1, model.m)
                pre_relu = model.W(X_flat) - model.b
                post_relu = F.relu(pre_relu)
                alive_frac = (post_relu > 0).float().mean().item()
                print(f"    [DEBUG] ReLU alive fraction: {alive_frac:.4f} (0 means dead ReLUs!)")

    return history


# ============================================================================
# EXPERIMENT 1: View Strategy Comparison
# ============================================================================

def experiment_view_strategy_comparison(
    d: int = 100,
    m: int = 10,
    K_true: int = 50,
    K_model: int = 100,
    N_train: int = 1000,
    N_test: int = 500,
    sigma: float = 0.5,
    lr: float = 0.01,
    epochs: int = 1000,
    num_seeds: int = 5,
    strategies: List[ViewStrategy] = None
) -> Dict:
    """
    Compare different view generation strategies.

    Tests how different ways of creating local views affect learning.
    """
    if strategies is None:
        strategies = [
            ViewStrategy.SLIDING_WINDOW,
            ViewStrategy.RANDOM_SUBSETS,
            ViewStrategy.DISJOINT_BLOCKS,
            ViewStrategy.DENSE_SLIDING
        ]

    print("\n" + "="*80)
    print("EXPERIMENT: View Strategy Comparison")
    print("="*80)
    print(f"Config: d={d}, m={m}, K_true={K_true}, K_model={K_model}")
    print(f"N_train={N_train}, N_test={N_test}, sigma={sigma}")
    print(f"Strategies: {[s.value for s in strategies]}")
    print()

    results = {strategy.value: {
        'gaps': [], 'clean_loss': [], 'noisy_loss': [],
        'histories': []  # Store full training histories for plotting
    } for strategy in strategies}

    for strategy in strategies:
        print(f"\n--- Testing {strategy.value} ---")

        for seed in range(num_seeds):
            print(f"  Seed {seed+1}/{num_seeds}...", end=' ')

            # Generate views
            H = generate_views(d, m, strategy, seed=seed)
            L = H.shape[0]

            # Generate data
            X_train, y_train_clean, y_train_noisy, groundtruth_fn = generate_synthetic_dataset(
                d, K_true, m, H, N_train, sigma, seed
            )
            X_test, y_test_clean, y_test_noisy, _ = generate_synthetic_dataset(
                d, K_true, m, H, N_test, sigma, seed + 1000
            )

            # Train student
            model = SyntheticModel(d, K_model, H)
            history = train_model(
                model, X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean,
                lr=lr, epochs=epochs, device=DEVICE,
                y_train_clean=y_train_clean  # Pass clean labels for train_loss_clean
            )

            # Record final metrics
            results[strategy.value]['gaps'].append(history['gen_gap'][-1])
            results[strategy.value]['clean_loss'].append(history['test_loss_clean'][-1])
            results[strategy.value]['noisy_loss'].append(history['test_loss_noisy'][-1])
            # Store full history for convergence plots
            results[strategy.value]['histories'].append(history)

            print(f"Gap={history['gen_gap'][-1]:.4f}, Clean Loss={history['test_loss_clean'][-1]:.4f}")

        # Print summary
        print(f"\n  Summary for {strategy.value}:")
        print(f"    Generalization Gap: {np.mean(results[strategy.value]['gaps']):.4f} ± {np.std(results[strategy.value]['gaps']):.4f}")
        print(f"    Clean Test Loss: {np.mean(results[strategy.value]['clean_loss']):.4f} ± {np.std(results[strategy.value]['clean_loss']):.4f}")

    return results


# ============================================================================
# EXPERIMENT 2: Sample Complexity with Synthetic Teacher
# ============================================================================

def experiment_sample_complexity_synthetic(
    d: int = 100,
    m: int = 10,
    K_true: int = 50,
    K_model: int = 100,
    n_values: List[int] = None,
    sigma: float = 0.5,
    strategy: ViewStrategy = ViewStrategy.SLIDING_WINDOW,
    lr: float = 0.01,
    epochs: int = 1000,
    num_seeds: int = 5
) -> Dict:
    """
    Test sample complexity with synthetic teacher.

    How many samples needed to learn the teacher function?
    """
    if n_values is None:
        n_values = [100, 500, 1000, 2000, 5000]

    print("\n" + "="*80)
    print("EXPERIMENT: Sample Complexity (Synthetic Teacher)")
    print("="*80)
    print(f"Config: d={d}, m={m}, K_true={K_true}, K_model={K_model}")
    print(f"Strategy: {strategy.value}, sigma={sigma}")
    print(f"Testing n values: {n_values}")
    print()

    results = {n: {'gaps': [], 'clean_loss': [], 'noisy_loss': [], 'histories': []} for n in n_values}

    for n in n_values:
        print(f"\n--- N = {n} ---")

        for seed in range(num_seeds):
            print(f"  Seed {seed+1}/{num_seeds}...", end=' ')

            # Generate views (same for all n)
            H = generate_views(d, m, strategy, seed=seed)

            # Generate data
            X_train, y_train_clean, y_train_noisy, groundtruth_fn = generate_synthetic_dataset(
                d, K_true, m, H, n, sigma, seed
            )
            X_test, y_test_clean, y_test_noisy, _ = generate_synthetic_dataset(
                d, K_true, m, H, 1000, sigma, seed + 1000
            )

            # Train
            model = SyntheticModel(d, K_model, H)
            history = train_model(
                model, X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean,
                lr=lr, epochs=epochs, device=DEVICE,
                y_train_clean=y_train_clean  # Pass clean labels for train_loss_clean
            )

            results[n]['gaps'].append(history['gen_gap'][-1])
            results[n]['clean_loss'].append(history['test_loss_clean'][-1])
            results[n]['noisy_loss'].append(history['train_loss'][-1])
            results[n]['histories'].append(history)

            print(f"Gap={history['gen_gap'][-1]:.4f}, Clean={history['test_loss_clean'][-1]:.4f}")

        print(f"  Summary: Gap={np.mean(results[n]['gaps']):.4f}±{np.std(results[n]['gaps']):.4f}")

    return results


# ============================================================================
# EXPERIMENT 3: Teacher-Student Width Gap
# ============================================================================

def experiment_width_gap(
    d: int = 100,
    m: int = 10,
    K_true: int = 50,
    K_model_values: List[int] = None,
    N_train: int = 1000,
    sigma: float = 0.5,
    strategy: ViewStrategy = ViewStrategy.SLIDING_WINDOW,
    lr: float = 0.01,
    epochs: int = 2000,
    num_seeds: int = 5
) -> Dict:
    """
    Test effect of student width relative to teacher.

    Can underparameterized student learn? How much overparameterization helps?
    """
    if K_model_values is None:
        K_model_values = [25, 50, 100, 200, 500]  # Relative to K_true=50

    print("\n" + "="*80)
    print("EXPERIMENT: Teacher-Student Width Gap")
    print("="*80)
    print(f"Config: d={d}, m={m}, K_true={K_true}")
    print(f"Testing student widths: {K_model_values}")
    print()

    results = {K: {'gaps': [], 'clean_loss': [], 'noisy_loss': [], 'histories': []} for K in K_model_values}

    for K_model in K_model_values:
        print(f"\n--- K_model = {K_model} (teacher={K_true}) ---")

        for seed in range(num_seeds):
            print(f"  Seed {seed+1}/{num_seeds}...", end=' ')

            H = generate_views(d, m, strategy, seed=seed)

            X_train, y_train_clean, y_train_noisy, _ = generate_synthetic_dataset(
                d, K_true, m, H, N_train, sigma, seed
            )
            X_test, y_test_clean, y_test_noisy, _ = generate_synthetic_dataset(
                d, K_true, m, H, 500, sigma, seed + 1000
            )

            model = SyntheticModel(d, K_model, H)
            history = train_model(
                model, X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean,
                lr=lr, epochs=epochs, device=DEVICE,
                y_train_clean=y_train_clean  # Pass clean labels for train_loss_clean
            )

            results[K_model]['gaps'].append(history['gen_gap'][-1])
            results[K_model]['clean_loss'].append(history['test_loss_clean'][-1])
            results[K_model]['noisy_loss'].append(history['train_loss'][-1])
            results[K_model]['histories'].append(history)

            print(f"Clean Loss={history['test_loss_clean'][-1]:.4f}")

        print(f"  Avg Clean Loss: {np.mean(results[K_model]['clean_loss']):.4f}±{np.std(results[K_model]['clean_loss']):.4f}")

    return results


# ============================================================================
# EXPERIMENT 4: Patch Size Effect (Synthetic)
# ============================================================================

def experiment_patch_size_synthetic(
    d: int = 100,
    m_values: List[int] = None,
    K_true: int = 50,
    K_model: int = 100,
    N_train: int = 1000,
    sigma: float = 0.5,
    strategy: ViewStrategy = ViewStrategy.SLIDING_WINDOW,
    lr: float = 0.01,
    epochs: int = 1000,
    num_seeds: int = 5
) -> Dict:
    """
    Effect of patch size with synthetic teacher.
    """
    if m_values is None:
        m_values = [5, 10, 15, 20]

    print("\n" + "="*80)
    print("EXPERIMENT: Patch Size Effect (Synthetic Teacher)")
    print("="*80)
    print(f"Config: d={d}, K_true={K_true}, K_model={K_model}")
    print(f"Testing m values: {m_values}")
    print()

    results = {m: {'gaps': [], 'clean_loss': [], 'noisy_loss': [], 'L': None, 'histories': []} for m in m_values}

    for m in m_values:
        print(f"\n--- m = {m} ---")

        for seed in range(num_seeds):
            print(f"  Seed {seed+1}/{num_seeds}...", end=' ')

            H = generate_views(d, m, strategy, seed=seed)
            L = H.shape[0]

            if results[m]['L'] is None:
                results[m]['L'] = L

            X_train, y_train_clean, y_train_noisy, _ = generate_synthetic_dataset(
                d, K_true, m, H, N_train, sigma, seed
            )
            X_test, y_test_clean, y_test_noisy, _ = generate_synthetic_dataset(
                d, K_true, m, H, 500, sigma, seed + 1000
            )

            model = SyntheticModel(d, K_model, H)
            history = train_model(
                model, X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean,
                lr=lr, epochs=epochs, device=DEVICE,
                y_train_clean=y_train_clean  # Pass clean labels for train_loss_clean
            )

            results[m]['gaps'].append(history['gen_gap'][-1])
            results[m]['clean_loss'].append(history['test_loss_clean'][-1])
            results[m]['noisy_loss'].append(history['train_loss'][-1])
            results[m]['histories'].append(history)

            print(f"L={L}, Clean Loss={history['test_loss_clean'][-1]:.4f}")

        print(f"  Summary: L={L}, Clean Loss={np.mean(results[m]['clean_loss']):.4f}±{np.std(results[m]['clean_loss']):.4f}")

    return results


# ============================================================================
# Identical Patches Experiments
# ============================================================================

def generate_identical_patches_dataset(
    m: int,
    L: int,
    K_true: int,
    N: int,
    sigma: float,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Generate dataset where ALL L patches are IDENTICAL copies of the same m-dim vector.

    Input structure: X = [patch, patch, ..., patch] (L times)
    - Each patch is m-dimensional, sampled from unit sphere S^(m-1)
    - Total input dimension: d = L × m

    Args:
        m: Patch size (dimension of each local view)
        L: Number of patches (all identical)
        K_true: Width of ground truth network
        N: Number of samples
        sigma: Noise level
        seed: Random seed

    Returns:
        X: Input tensor (N, d) where d = L×m
        y_clean: Clean labels (N,)
        y_noisy: Noisy labels (N,)
        H: Local view indices (L, m) - disjoint consecutive blocks
    """
    d = L * m
    rng = np.random.default_rng(seed)

    # Generate disjoint consecutive views: H[i] = [i*m, i*m+1, ..., (i+1)*m-1]
    H = np.array([np.arange(i * m, (i + 1) * m) for i in range(L)])

    # Create ground truth function
    gt_seed = seed + 999
    gt_rng = np.random.default_rng(gt_seed)
    v = gt_rng.normal(0, 1, K_true)
    w = gt_rng.normal(0, 1, (K_true, m))
    b = gt_rng.uniform(-0.5, 0.5, K_true)
    beta = gt_rng.uniform(-1, 1)

    def ground_truth(X):
        """Ground truth function for identical patches."""
        N_samples = X.shape[0]
        X_H = X[:, H]  # (N, L, m)
        X_H_flat = X_H.reshape(N_samples * L, m)
        hidden_flat = X_H_flat @ w.T - b  # (N*L, K)
        hidden_flat = np.maximum(0, hidden_flat)  # ReLU
        hidden = hidden_flat.reshape(N_samples, L, K_true)
        pooled = np.mean(hidden, axis=1)  # (N, K)
        return (pooled * v).sum(axis=1) + beta

    # Generate N samples with IDENTICAL patches
    X = np.zeros((N, d))
    for i in range(N):
        # Sample one patch from unit sphere S^(m-1)
        patch = rng.normal(0, 1, m)
        patch = patch / np.linalg.norm(patch)
        # Replicate L times
        for j in range(L):
            X[i, j * m:(j + 1) * m] = patch

    # Generate labels
    y_clean = ground_truth(X)
    noise = rng.normal(0, sigma, N)
    y_noisy = y_clean + noise

    # Convert to tensors
    X = torch.FloatTensor(X)
    y_clean = torch.FloatTensor(y_clean)
    y_noisy = torch.FloatTensor(y_noisy)

    return X, y_clean, y_noisy, H


def experiment_identical_patches(
    m: int = 9,
    L: int = 10,
    K_true: int = 50,
    K_model: int = 1024,
    N_train: int = 512,
    N_test: int = 500,
    sigma: float = 1.0,
    lr: float = 0.1,
    epochs: int = 4000,
    num_seeds: int = 3
) -> dict:
    """
    Experiment with identical patches input structure.

    Tests how the CNN handles periodic/redundant input where all patches are identical.
    Input has only m degrees of freedom but ambient dimension is d = L×m.

    Args:
        m: Patch size
        L: Number of identical patches
        K_true: Ground truth network width
        K_model: Model network width
        N_train: Training samples
        N_test: Test samples
        sigma: Noise level
        lr: Learning rate
        epochs: Training epochs
        num_seeds: Number of random seeds

    Returns:
        Dictionary with results and histories
    """
    d = L * m

    # Calculate model parameters: w (K×m) + b (K) + v (K) + beta (1)
    num_params_teacher = K_true * m + K_true + K_true + 1
    num_params_model = K_model * m + K_model + K_model + 1

    print(f"\n{'='*70}")
    print(f"IDENTICAL PATCHES EXPERIMENT")
    print(f"{'='*70}")
    print(f"\n📊 SETUP:")
    print(f"  Input Structure:")
    print(f"    - Ambient dimension (d): {d}")
    print(f"    - Patch size (m): {m}")
    print(f"    - Number of patches (L): {L}")
    print(f"    - Effective DoF: {m} (all patches identical)")
    print(f"\n  Ground Truth (Teacher):")
    print(f"    - Width (K_true): {K_true}")
    print(f"    - Parameters: {num_params_teacher:,}")
    print(f"\n  Model (Student):")
    print(f"    - Width (K_model): {K_model}")
    print(f"    - Parameters: {num_params_model:,}")
    print(f"    - Overparameterization ratio: {K_model/K_true:.1f}x")
    print(f"\n  Training:")
    print(f"    - N_train: {N_train}")
    print(f"    - N_test: {N_test}")
    print(f"    - Noise (σ): {sigma} → σ²={sigma**2}")
    print(f"    - Learning rate: {lr}")
    print(f"    - Epochs: {epochs}")
    print(f"    - Seeds: {num_seeds}")
    print(f"\n  Rank Restriction:")
    print(f"    - Max effective rank: L×m = {L*m}")
    print(f"{'='*70}")

    results = {
        'clean_loss': [],
        'noisy_loss': [],
        'gaps': [],
        'histories': []
    }

    for seed in range(num_seeds):
        print(f"\n  Seed {seed+1}/{num_seeds}...")

        # Generate data
        X_train, y_train_clean, y_train_noisy, H = generate_identical_patches_dataset(
            m, L, K_true, N_train, sigma, seed
        )
        X_test, y_test_clean, y_test_noisy, _ = generate_identical_patches_dataset(
            m, L, K_true, N_test, sigma, seed + 1000
        )

        # Create and train model
        model = SyntheticModel(d, K_model, H)
        history = train_model(
            model, X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean,
            y_train_clean=y_train_clean, lr=lr, epochs=epochs, device=DEVICE
        )

        results['clean_loss'].append(history['test_loss_clean'][-1])
        results['noisy_loss'].append(history['test_loss_noisy'][-1])
        results['gaps'].append(history['gen_gap'][-1])
        results['histories'].append(history)

        print(f"    Clean Loss={history['test_loss_clean'][-1]:.4f}, "
              f"Noisy Loss={history['test_loss_noisy'][-1]:.4f}")

    print(f"\n  Summary:")
    print(f"    Clean Loss: {np.mean(results['clean_loss']):.4f} ± {np.std(results['clean_loss']):.4f}")
    print(f"    Noisy Loss: {np.mean(results['noisy_loss']):.4f} ± {np.std(results['noisy_loss']):.4f}")

    return results


# ============================================================================
# Augmented Views Experiments
# ============================================================================

def generate_augmented_views_dataset(
    m: int,
    L: int,
    R: int,
    K_true: int,
    N: int,
    sigma: float,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Generate dataset with:
    - Input: Periodic identical patches (dimension L×m)
    - Views: H = H_disjoint(m,L) ∪ H_random(R)

    Args:
        m: Patch size
        L: Number of disjoint views (and periodic repetitions in input)
        R: Number of additional random views
        K_true: Ground truth network width
        N: Number of samples
        sigma: Noise level
        seed: Random seed

    Returns:
        X, y_clean, y_noisy, H (augmented views)
    """
    d = L * m
    rng = np.random.default_rng(seed)

    # Part 1: Disjoint consecutive blocks
    H_disjoint = np.array([np.arange(i * m, (i + 1) * m) for i in range(L)])

    # Part 2: R random views (handle R=0 case)
    if R > 0:
        H_random = np.array([np.sort(rng.choice(d, size=m, replace=False)) for _ in range(R)])
        # Combine: H = H_disjoint ∪ H_random
        H = np.vstack([H_disjoint, H_random])
    else:
        H = H_disjoint

    total_views = L + R

    # Create ground truth function with augmented views
    gt_seed = seed + 999
    gt_rng = np.random.default_rng(gt_seed)
    v = gt_rng.normal(0, 1, K_true)
    w = gt_rng.normal(0, 1, (K_true, m))
    b = gt_rng.uniform(-0.5, 0.5, K_true)
    beta = gt_rng.uniform(-1, 1)

    def ground_truth(X):
        N_samples = X.shape[0]
        X_H = X[:, H]  # (N, total_views, m)
        X_H_flat = X_H.reshape(N_samples * total_views, m)
        hidden_flat = X_H_flat @ w.T - b
        hidden_flat = np.maximum(0, hidden_flat)
        hidden = hidden_flat.reshape(N_samples, total_views, K_true)
        pooled = np.mean(hidden, axis=1)
        return (pooled * v).sum(axis=1) + beta

    # Generate N samples with IDENTICAL periodic patches
    X = np.zeros((N, d))
    for i in range(N):
        patch = rng.normal(0, 1, m)
        patch = patch / np.linalg.norm(patch)
        for j in range(L):
            X[i, j * m:(j + 1) * m] = patch

    # Generate labels
    y_clean = ground_truth(X)
    noise = rng.normal(0, sigma, N)
    y_noisy = y_clean + noise

    X = torch.FloatTensor(X)
    y_clean = torch.FloatTensor(y_clean)
    y_noisy = torch.FloatTensor(y_noisy)

    return X, y_clean, y_noisy, H


def experiment_augmented_views(
    m: int = 9,
    L: int = 10,
    R_values: list = [0, 10, 20, 40],
    K_true: int = 50,
    K_model: int = 1024,
    N_train: int = 512,
    N_test: int = 500,
    sigma: float = 1.0,
    lr: float = 0.1,
    epochs: int = 4000,
    num_seeds: int = 3
) -> dict:
    """
    Experiment with augmented views: H = H_disjoint ∪ H_random.

    Tests whether adding random views to structured disjoint views helps learning
    when input has periodic identical patches.

    Args:
        m: Patch size
        L: Number of disjoint views (and periodic repetitions)
        R_values: List of random view counts to test
        K_true: Ground truth network width
        K_model: Model network width
        N_train: Training samples
        N_test: Test samples
        sigma: Noise level
        lr: Learning rate
        epochs: Training epochs
        num_seeds: Number of random seeds

    Returns:
        Dictionary with results for each R value
    """
    d = L * m

    # Calculate model parameters: w (K×m) + b (K) + v (K) + beta (1)
    num_params_teacher = K_true * m + K_true + K_true + 1
    num_params_model = K_model * m + K_model + K_model + 1

    print(f"\n{'='*70}")
    print(f"AUGMENTED VIEWS EXPERIMENT")
    print(f"{'='*70}")
    print(f"\n📊 SETUP:")
    print(f"  Input Structure:")
    print(f"    - Ambient dimension (d): {d}")
    print(f"    - Patch size (m): {m}")
    print(f"    - Number of disjoint views (L): {L}")
    print(f"    - Input type: Periodic identical patches")
    print(f"    - Effective DoF: {m} (all L patches identical)")
    print(f"\n  Ground Truth (Teacher):")
    print(f"    - Width (K_true): {K_true}")
    print(f"    - Parameters: {num_params_teacher:,}")
    print(f"\n  Model (Student):")
    print(f"    - Width (K_model): {K_model}")
    print(f"    - Parameters: {num_params_model:,}")
    print(f"    - Overparameterization ratio: {K_model/K_true:.1f}x")
    print(f"\n  Training:")
    print(f"    - N_train: {N_train}")
    print(f"    - N_test: {N_test}")
    print(f"    - Noise (σ): {sigma} → σ²={sigma**2}")
    print(f"    - Learning rate: {lr}")
    print(f"    - Epochs: {epochs}")
    print(f"    - Seeds: {num_seeds}")
    print(f"\n  Augmented Views:")
    print(f"    - H = H_disjoint(L={L}) ∪ H_random(R)")
    print(f"    - Testing R values: {R_values}")
    print(f"    - Total views for each R: {[L + R for R in R_values]}")
    print(f"\n  Rank Restriction:")
    print(f"    - With L disjoint views: max rank = L×m = {L*m}")
    print(f"    - With R random views: rank increases (more coverage)")
    print(f"{'='*70}")

    results = {R: {'clean_loss': [], 'noisy_loss': [], 'gaps': [], 'histories': []}
               for R in R_values}

    for R in R_values:
        total_views = L + R
        print(f"\n--- R={R} (total views: {total_views}) ---")

        for seed in range(num_seeds):
            print(f"  Seed {seed+1}/{num_seeds}...", end=' ')

            X_train, y_train_clean, y_train_noisy, H = generate_augmented_views_dataset(
                m, L, R, K_true, N_train, sigma, seed
            )
            X_test, y_test_clean, y_test_noisy, _ = generate_augmented_views_dataset(
                m, L, R, K_true, N_test, sigma, seed + 1000
            )

            model = SyntheticModel(d, K_model, H)
            history = train_model(
                model, X_train, y_train_noisy, X_test, y_test_noisy, y_test_clean,
                y_train_clean=y_train_clean, lr=lr, epochs=epochs, device=DEVICE
            )

            results[R]['clean_loss'].append(history['test_loss_clean'][-1])
            results[R]['noisy_loss'].append(history['test_loss_noisy'][-1])
            results[R]['gaps'].append(history['gen_gap'][-1])
            results[R]['histories'].append(history)

            print(f"Clean Loss={history['test_loss_clean'][-1]:.4f}")

        print(f"  Summary: Clean Loss={np.mean(results[R]['clean_loss']):.4f} "
              f"± {np.std(results[R]['clean_loss']):.4f}")

    return results


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SYNTHETIC TEACHER-STUDENT EXPERIMENTS")
    print("="*80)
    print(f"Device: {DEVICE}")
    print()

    # Run quick demo of each experiment
    print("\n>>> Quick Demo: View Strategy Comparison")
    results1 = experiment_view_strategy_comparison(
        d=80, m=8, K_true=30, K_model=60,
        N_train=500, epochs=500, num_seeds=2
    )

    print("\n>>> Quick Demo: Sample Complexity")
    results2 = experiment_sample_complexity_synthetic(
        d=80, m=8, n_values=[200, 500, 1000, 2000],
        epochs=500, num_seeds=2
    )

    print("\n>>> Quick Demo: Width Gap")
    results3 = experiment_width_gap(
        d=80, m=8, K_true=50,
        K_model_values=[25, 50, 100, 200],
        epochs=1000, num_seeds=2
    )

    print("\n>>> Quick Demo: Patch Size Effect")
    results4 = experiment_patch_size_synthetic(
        d=100, m_values=[5, 10, 15],
        epochs=500, num_seeds=2
    )

    print("\n" + "="*80)
    print("ALL QUICK DEMOS COMPLETED")
    print("="*80)
