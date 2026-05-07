# Does Sparse Connectivity Improve Generalization? Convolutional Networks Below the Edge of Stability

This repository contains the code and experiments accompanying the paper. We study how sparse connectivity changes generalization below the Edge of Stability in two-layer ReLU networks, showing that sparse architectures process low-dimensional patch collections whose geometry governs the effective stability constraint — yielding non-vacuous generalization bounds in regimes where fully connected networks provably fail.

---

## Repository Structure

```
.
├── notebooks/
│   ├── clean_patch_vs_image_depth.ipynb    # Patch geometry evaluation across 12 connectivity schemes
│   ├── flat_interpolation_experiment.ipynb # Interpolation without generalization (Experiment 2)
│   ├── cifar_regression_experiment.ipynb   # FCN vs. LCN-WS on CIFAR-10 regression
│   ├── parameter_sharing_experiment.ipynb  # High-dimensional vocabulary learning experiment
│   ├── unshared_scn_sweep.ipynb          # Generalization gap scaling for unshared SCN (n and d sweep)
│   └── Toy_Experiment.ipynb               # Synthetic theory experiment with local view networks
├── src/
│   ├── data.py        # Sphere sampling, label generation, ground-truth functions
│   ├── networks.py    # SharedCNN, SyntheticModel, FullyConnected, UnsharedCNN
│   ├── training.py    # Training loops and optimization
│   ├── metrics.py     # Loss, generalization gap, convergence detection
│   ├── views.py       # Six view generation strategies (sliding window, disjoint blocks, etc.)
│   └── __init__.py
├── synthetic_teacher_experiments.py # Core model classes and data generation used by notebooks
└── LICENSE
```

---

## Key Experiments

### 1. Patchification Geometry Search (`notebooks/clean_patch_vs_image_depth.ipynb`)

Evaluates 12 patch connectivity schemes on CIFAR-10 using two geometry metrics:

| Metric | Interpretation |
|---|---|
| PCA elbow@90% (lower is better) | Effective intrinsic dimensionality of the patch distribution |
| Tukey half-space depth area (higher is better) | Concentration of the patch point cloud; harder to shatter |

**Schemes tested**: Conv (baseline), NonOverlap, JitteredGrid, RandomLocal, GlobalShuffle (negative control), Hex, Voronoi, Radial, LargeKernelSparse, MultiScale, Cross, RandomPlacement.

**Key result**: Five non-convolutional schemes (MultiScale, Voronoi, NonOverlap, Radial, JitteredGrid) match or exceed Conv on both metrics. GlobalShuffle collapses to image-space geometry. Locality — sampling spatially contiguous pixels — is the decisive factor.

---

### 2. Flat Interpolation Experiment (`notebooks/flat_interpolation_experiment.ipynb`)

Demonstrates that architecture alone is not sufficient for generalization.

- **Data**: Each patch `x^(j) ~ S^{m-1}` independently (patch-wise sphere); patches carry no shared spatial structure.
- **Model**: LCN-WS (K=1024, J=8 disjoint patches, m=10).
- **Finding**: LCN-WS interpolates (train loss → 0) while remaining at the edge of stability (λ_max(∇²L) ≈ 2/η), yet excess risk remains high. Without distributional alignment between architecture and data, interpolation does not imply generalization.

---

### 3. CIFAR-10 Regression: FCN vs. LCN-WS (`notebooks/cifar_regression_experiment.ipynb`)

Regression on CIFAR-10 with a ground-truth function that has local spatial structure.

- **Ground truth**: Two-layer SCN with K_TRUE=20, 3×3 patches, stride=1 (L=900 positions).
- **Finding**: LCN-WS achieves 3.7× lower excess risk with 106× fewer parameters. FCN interpolates the noisy training labels but fails to recover the underlying local structure. Hessian sharpness (λ_max) is tracked throughout; both models approach 2/η.

---

### 4. Patch Geometry vs. Image Space (`notebooks/clean_patch_vs_image_depth.ipynb`)

Compares the Tukey half-space depth of conv patch distributions against full image-space distributions, establishing that local patches occupy a fundamentally more concentrated region of the feature space — the geometric basis for the generalization advantage.

---

### 5. Synthetic Theory Experiment (`notebooks/Toy_Experiment.ipynb`)

Implements the core synthetic setup from the paper using local view networks on spherical data. Covers data generation, the ground-truth local view function, and training dynamics on controlled synthetic settings.

---

## Setup

### Requirements

```
python >= 3.9
torch >= 2.0
torchvision
numpy
scipy
matplotlib
pandas
tqdm
```

Install dependencies:

```bash
pip install torch torchvision numpy scipy matplotlib pandas tqdm
```

CIFAR-10 is downloaded automatically via `torchvision.datasets.CIFAR10` on first run.

### Running the Experiments

Each experiment is a self-contained Jupyter notebook. Launch with:

```bash
jupyter notebook
```

1. `notebooks/clean_patch_vs_image_depth.ipynb` — patch geometry search
2. `notebooks/flat_interpolation_experiment.ipynb` — interpolation without generalization
3. `notebooks/cifar_regression_experiment.ipynb` — FCN vs. LCN-WS on CIFAR-10
4. `notebooks/parameter_sharing_experiment.ipynb` — parameter sharing study
5. `notebooks/Toy_Experiment.ipynb` — synthetic theory experiment

GPU is recommended for the geometry search and CIFAR-10 experiments. Synthetic experiments run on CPU.

---

## Source Package (`src/`)

| Module | Contents |
|---|---|
| `data.py` | Sphere sampling, noisy label generation, ground-truth LCN teacher |
| `networks.py` | `SharedCNN` (LCN-WS), `SyntheticModel`, `FullyConnected`, `UnsharedCNN` |
| `training.py` | Training loops with loss, excess risk, and Hessian tracking |
| `metrics.py` | Generalization gap, convergence detection, metrics dataclass |
| `views.py` | Six view strategies: sliding window, random subsets, disjoint blocks, random overlapping, hierarchical, dense sliding |

`synthetic_teacher_experiments.py` at the root provides additional model classes and data generation utilities used directly by the notebooks.

---

## Generalization Gap Definition

Throughout all experiments, the generalization gap is defined as:

$$\widehat{\text{Gen}}(f, \mathcal{D}) = \left| \widehat{\text{Excess}}(f) + \sigma^2 - \hat{R}_{\mathcal{D}}(f) \right|$$

where $\hat{R}_{\mathcal{D}}(f)$ is the empirical train loss on noisy labels, $\widehat{\text{Excess}}(f)$ is the MSE against clean test labels, and $\sigma^2$ is the label noise variance.

---

## Geometry Metrics

**PCA effective rank** (elbow@90%): the number of principal components needed to explain 90% of variance in the patch distribution. Lower values indicate a more structured, lower-dimensional point cloud.

**Tukey half-space depth area**: for a query patch $x$, the depth is $\min_{u \in S^{K-1}} \min(F_u(u^\top x), 1 - F_u(u^\top x))$ where $F_u$ is the CDF of projections of the reference set onto $u$. We report $\int_0^{0.5} \Psi(t)\, dt$ where $\Psi(t) = P(\text{depth} \geq t)$. Higher values indicate a more concentrated distribution that is harder to shatter.

---

## License

MIT License. See `LICENSE` for details.
