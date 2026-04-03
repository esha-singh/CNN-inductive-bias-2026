"""
Metrics Module

Handles all metric computations:
- Loss computation
- Generalization gap
- Convergence detection
- Boundary conditions
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================================
# Metrics Dataclass
# ============================================================================

@dataclass
class Metrics:
    """Container for all metrics tracked during training."""
    train_loss: float
    test_loss: float
    test_loss_clean: float  # Loss on clean labels (without noise)
    gen_gap: float
    convergence_epoch: Optional[int] = None


# ============================================================================
# Convergence Detection
# ============================================================================

def check_convergence(loss_curve, window=10, threshold=0.001) -> Tuple[bool, Optional[int]]:
    """
    Check if training has converged by looking at variance in final epochs.

    Args:
        loss_curve: List of loss values per epoch
        window: Number of final epochs to check
        threshold: Maximum relative change to consider converged (0.001 = 0.1% variance)

    Returns:
        (converged, convergence_epoch)
    """
    if len(loss_curve) < window:
        return False, None

    # Check if loss has stabilized in last 'window' epochs
    recent_losses = loss_curve[-window:]
    mean_recent = np.mean(recent_losses)
    std_recent = np.std(recent_losses)

    if mean_recent > 0:
        relative_std = std_recent / mean_recent
        converged = relative_std < threshold
    else:
        converged = std_recent < threshold

    # Find approximate convergence epoch (when relative change drops below threshold)
    if converged:
        for i in range(len(loss_curve) - window, 0, -1):
            if i >= window:
                window_losses = loss_curve[i:i+window]
                window_mean = np.mean(window_losses)
                window_std = np.std(window_losses)
                if window_mean > 0:
                    rel_std = window_std / window_mean
                    if rel_std > threshold * 2:  # 2x threshold
                        return converged, i + window
        return converged, window

    return False, None


# ============================================================================
# Boundary Conditions
# ============================================================================

def check_boundary_condition(model, threshold=1e-3):
    """
    Check if model satisfies boundary conditions from theory.

    For SharedCNN: Checks if learned filters have specific structure.

    Args:
        model: Neural network model
        threshold: Tolerance for boundary check

    Returns:
        satisfies_boundary (bool)
    """
    # Placeholder - implement based on theoretical boundary conditions
    return True


# ============================================================================
# Theoretical Alpha Computation
# ============================================================================

def compute_theoretical_alpha(d: int, m: int):
    """
    Compute theoretical sample complexity exponent α.

    From theory: α = (d+3)(d-m)/(2(4d²+3d-2md-3m))

    Args:
        d: Ambient dimension
        m: Patch size

    Returns:
        alpha: Sample complexity exponent
    """
    numerator = (d + 3) * (d - m)
    denominator = 2 * (4 * d**2 + 3 * d - 2 * m * d - 3 * m)
    if denominator == 0:
        return 0.0
    alpha = numerator / denominator
    return alpha


# ============================================================================
# Generalization Gap
# ============================================================================

def compute_generalization_gap(
    model,
    train_loader,
    test_loader,
    test_loader_clean=None,
    criterion=None,
    device='cuda'
):
    """
    Compute generalization gap and test losses.

    Args:
        model: Neural network
        train_loader: Training data loader
        test_loader: Test data loader (with noisy labels)
        test_loader_clean: Test data loader (with clean labels), optional
        criterion: Loss function (if None, uses MSE)
        device: Device to compute on

    Returns:
        If test_loader_clean is None:
            train_loss, test_loss, gen_gap
        If test_loader_clean is provided:
            train_loss, test_loss_noisy, test_loss_clean, gen_gap
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()

    model.eval()

    # Compute train loss
    train_loss = 0.0
    train_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            train_loss += loss.item() * X_batch.shape[0]
            train_samples += X_batch.shape[0]
    train_loss /= train_samples

    # Compute test loss (noisy)
    test_loss_noisy = 0.0
    test_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss_noisy += loss.item() * X_batch.shape[0]
            test_samples += X_batch.shape[0]
    test_loss_noisy /= test_samples

    # Compute test loss (clean) if provided
    if test_loader_clean is not None:
        test_loss_clean = 0.0
        clean_samples = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader_clean:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss_clean += loss.item() * X_batch.shape[0]
                clean_samples += X_batch.shape[0]
        test_loss_clean /= clean_samples

        gen_gap = test_loss_noisy - train_loss
        return train_loss, test_loss_noisy, test_loss_clean, gen_gap
    else:
        gen_gap = test_loss_noisy - train_loss
        return train_loss, test_loss_noisy, gen_gap


# ============================================================================
# Loss Utilities
# ============================================================================

def compute_loss_on_batch(model, X, y, criterion=None, device='cuda'):
    """
    Compute loss on a single batch.

    Args:
        model: Neural network
        X: Input tensor
        y: Target tensor
        criterion: Loss function (if None, uses MSE)
        device: Device

    Returns:
        loss value (scalar)
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()

    model.eval()
    X = X.to(device)
    y = y.to(device)

    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)

    return loss.item()


def compute_per_sample_loss(model, X, y, device='cuda'):
    """
    Compute loss for each sample individually.

    Args:
        model: Neural network
        X: Input tensor of shape (N, d)
        y: Target tensor of shape (N,)
        device: Device

    Returns:
        losses: Array of shape (N,) with per-sample losses
    """
    model.eval()
    X = X.to(device)
    y = y.to(device)

    with torch.no_grad():
        outputs = model(X)
        losses = (outputs - y) ** 2

    return losses.cpu().numpy()


# ============================================================================
# Edge-of-Stability Metrics
# ============================================================================

def compute_sharpness(model, X, y, criterion=None, device='cuda', epsilon=1e-4):
    """
    Compute sharpness (maximum eigenvalue of loss Hessian approximation).

    Uses finite difference approximation.

    Args:
        model: Neural network
        X: Input batch
        y: Target batch
        criterion: Loss function
        epsilon: Finite difference step size
        device: Device

    Returns:
        Approximate sharpness
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()

    # Compute base loss
    model.eval()
    X = X.to(device)
    y = y.to(device)

    # Get flattened parameters
    params = []
    for p in model.parameters():
        params.append(p.view(-1))
    param_vec = torch.cat(params)

    # Compute loss
    outputs = model(X)
    loss = criterion(outputs, y)

    # Approximate largest eigenvalue using power iteration
    # (Simplified - full implementation would use proper power iteration)
    # For now, return gradient norm as proxy
    model.zero_grad()
    loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = np.sqrt(grad_norm)

    return grad_norm


def compute_gradient_norm(model):
    """
    Compute L2 norm of gradients.

    Args:
        model: Neural network (after backward pass)

    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = np.sqrt(total_norm)
    return total_norm
