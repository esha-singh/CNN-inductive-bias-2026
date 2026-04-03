"""
Core functionality for CNN experiments.

Import everything needed for experiments in one line:
    from src import *
"""

# Data generation
from .data import (
    generate_sphere_data,
    generate_sphere_data_fast,
    generate_labels_smooth,
    generate_labels_binary,
    create_ground_truth_function,
    generate_synthetic_dataset,
    generate_identical_patches_data,
    generate_augmented_views,
    DEVICE
)

# View strategies
from .views import (
    ViewStrategy,
    generate_views,
    generate_views_sliding_window,
    generate_views_random_subsets,
    generate_views_disjoint_blocks,
    generate_views_random_overlapping,
    generate_views_hierarchical,
    generate_views_dense_sliding,
    compute_view_overlap,
    compute_view_coverage
)

# Network architectures
from .networks import (
    SharedCNN,
    SyntheticModel,
    FullyConnectedBaseline,
    UnsharedCNN,
    count_parameters,
    get_effective_rank
)

# Training
from .training import (
    train_model,
    train_model_simple,
    train_with_early_stopping,
    train_with_schedule
)

# Metrics
from .metrics import (
    Metrics,
    check_convergence,
    check_boundary_condition,
    compute_theoretical_alpha,
    compute_generalization_gap,
    compute_loss_on_batch,
    compute_per_sample_loss,
    compute_sharpness,
    compute_gradient_norm
)

__all__ = [
    # Data
    'generate_sphere_data',
    'generate_sphere_data_fast',
    'generate_labels_smooth',
    'generate_labels_binary',
    'create_ground_truth_function',
    'generate_synthetic_dataset',
    'generate_identical_patches_data',
    'generate_augmented_views',
    'DEVICE',

    # Views
    'ViewStrategy',
    'generate_views',
    'generate_views_sliding_window',
    'generate_views_random_subsets',
    'generate_views_disjoint_blocks',
    'generate_views_random_overlapping',
    'generate_views_hierarchical',
    'generate_views_dense_sliding',
    'compute_view_overlap',
    'compute_view_coverage',

    # Networks
    'SharedCNN',
    'SyntheticModel',
    'FullyConnectedBaseline',
    'UnsharedCNN',
    'count_parameters',
    'get_effective_rank',

    # Training
    'train_model',
    'train_model_simple',
    'train_with_early_stopping',
    'train_with_schedule',

    # Metrics
    'Metrics',
    'check_convergence',
    'check_boundary_condition',
    'compute_theoretical_alpha',
    'compute_generalization_gap',
    'compute_loss_on_batch',
    'compute_per_sample_loss',
    'compute_sharpness',
    'compute_gradient_norm',
]
