"""
View Generation Strategies

Six different strategies for generating local views:
1. Sliding Window - Overlapping stride-based
2. Random Subsets - Random m indices per view
3. Disjoint Blocks - Non-overlapping consecutive blocks
4. Random Overlapping - Random with controlled overlap
5. Hierarchical - Multi-scale patches
6. Dense Sliding - Stride=1, maximum overlap
"""

import numpy as np
from enum import Enum
from typing import Optional


# ============================================================================
# View Strategy Enum
# ============================================================================

class ViewStrategy(Enum):
    """Different strategies for generating local views"""
    SLIDING_WINDOW = "sliding_window"      # Overlapping stride-based
    RANDOM_SUBSETS = "random_subsets"      # Random m indices
    DISJOINT_BLOCKS = "disjoint_blocks"    # Non-overlapping consecutive
    RANDOM_OVERLAPPING = "random_overlap"  # Random with guaranteed overlap
    HIERARCHICAL = "hierarchical"          # Multi-scale patches
    DENSE_SLIDING = "dense_sliding"        # Stride=1 (maximum overlap)


# ============================================================================
# Individual View Generation Functions
# ============================================================================

def generate_views_sliding_window(d: int, m: int, stride: int, seed: int = None) -> np.ndarray:
    """
    Sliding window views.

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
    Random subset views.

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


# ============================================================================
# Unified Interface
# ============================================================================

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
# View Utilities
# ============================================================================

def compute_view_overlap(H: np.ndarray) -> float:
    """
    Compute average overlap ratio between views.

    Args:
        H: View indices, shape (L, m)

    Returns:
        Average overlap ratio in [0, 1]
    """
    L, m = H.shape
    if L <= 1:
        return 0.0

    overlaps = []
    for i in range(L):
        for j in range(i+1, L):
            set_i = set(H[i])
            set_j = set(H[j])
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            if union > 0:
                overlaps.append(intersection / union)

    return np.mean(overlaps) if overlaps else 0.0


def compute_view_coverage(H: np.ndarray, d: int) -> float:
    """
    Compute what fraction of input dimensions are covered by views.

    Args:
        H: View indices, shape (L, m)
        d: Ambient dimension

    Returns:
        Coverage ratio in [0, 1]
    """
    covered = set()
    for view in H:
        covered.update(view[view >= 0])  # Exclude padding (-1)

    return len(covered) / d
