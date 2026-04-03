"""
Training Module

Handles all training loops and optimization:
- Standard training for main experiments
- Synthetic training with clean labels tracking
- Various optimizers
"""

import torch
import torch.nn as nn
from typing import Dict, List
from .metrics import check_convergence


# Detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test_noisy: torch.Tensor,
    y_test_clean: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 1000,
    device: str = 'cuda',
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Train model network with clean label tracking.

    Used in synthetic experiments.

    Args:
        model: Model network
        X_train: Training inputs
        y_train: Training labels (noisy)
        X_test: Test inputs
        y_test_noisy: Test labels (noisy)
        y_test_clean: Test labels (clean, from ground truth)
        lr: Learning rate
        epochs: Number of epochs
        device: Device to train on
        verbose: Print progress

    Returns:
        Dictionary with training history:
        - train_loss: List of training losses
        - test_loss_noisy: List of test losses (noisy labels)
        - test_loss_clean: List of test losses (clean labels)
        - gen_gap: List of generalization gaps
        - convergence_epoch: Epoch when converged (or None)
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test_noisy = y_test_noisy.to(device)
    y_test_clean = y_test_clean.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'test_loss_noisy': [],
        'test_loss_clean': [],
        'gen_gap': [],
        'convergence_epoch': None
    }

    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss_noisy = criterion(test_outputs, y_test_noisy).item()
            test_loss_clean = criterion(test_outputs, y_test_clean).item()

        gen_gap = test_loss_noisy - train_loss

        # Store history
        history['train_loss'].append(train_loss)
        history['test_loss_noisy'].append(test_loss_noisy)
        history['test_loss_clean'].append(test_loss_clean)
        history['gen_gap'].append(gen_gap)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.4f}, "
                  f"Test(noisy)={test_loss_noisy:.4f}, "
                  f"Test(clean)={test_loss_clean:.4f}")

    # Check convergence
    converged, conv_epoch = check_convergence(history['train_loss'])
    history['convergence_epoch'] = conv_epoch

    return history


# ============================================================================
# Simple Training Loop (Main Experiments)
# ============================================================================

def train_model_simple(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 1000,
    device: str = 'cuda',
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Simple training loop without clean label tracking.

    Used in main experiments.

    Args:
        model: Neural network
        X_train: Training inputs
        y_train: Training labels
        X_test: Test inputs
        y_test: Test labels
        lr: Learning rate
        epochs: Number of epochs
        device: Device to train on
        verbose: Print progress

    Returns:
        Dictionary with training history:
        - train_loss: List of training losses
        - test_loss: List of test losses
        - gen_gap: List of generalization gaps
        - convergence_epoch: Epoch when converged
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'test_loss': [],
        'gen_gap': [],
        'convergence_epoch': None
    }

    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()

        gen_gap = test_loss - train_loss

        # Store history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['gen_gap'].append(gen_gap)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Test={test_loss:.4f}")

    # Check convergence
    converged, conv_epoch = check_convergence(history['train_loss'])
    history['convergence_epoch'] = conv_epoch

    return history


# ============================================================================
# Training with Early Stopping
# ============================================================================

def train_with_early_stopping(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 0.01,
    max_epochs: int = 5000,
    patience: int = 50,
    device: str = 'cuda',
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Training with early stopping based on validation loss.

    Args:
        model: Neural network
        X_train: Training inputs
        y_train: Training labels
        X_val: Validation inputs
        y_val: Validation labels
        lr: Learning rate
        max_epochs: Maximum epochs
        patience: Epochs to wait for improvement
        device: Device
        verbose: Print progress

    Returns:
        Training history
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'stopped_epoch': None
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            history['stopped_epoch'] = epoch
            # Restore best model
            model.load_state_dict(best_model_state)
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    return history


# ============================================================================
# Advanced: Training with Learning Rate Schedule
# ============================================================================

def train_with_schedule(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lr_start: float = 0.1,
    lr_end: float = 0.001,
    epochs: int = 1000,
    schedule: str = 'cosine',
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """
    Training with learning rate schedule.

    Args:
        model: Neural network
        X_train, y_train: Training data
        X_test, y_test: Test data
        lr_start: Initial learning rate
        lr_end: Final learning rate
        epochs: Number of epochs
        schedule: 'cosine', 'linear', or 'exponential'
        device: Device

    Returns:
        Training history
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_start)
    criterion = nn.MSELoss()

    if schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_end
        )
    elif schedule == 'exponential':
        gamma = (lr_end / lr_start) ** (1 / epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:  # linear
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=lr_end/lr_start, total_iters=epochs
        )

    history = {
        'train_loss': [],
        'test_loss': [],
        'lr': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()

        # Update learning rate
        scheduler.step()

        # Store
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

    return history
