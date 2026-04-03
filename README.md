# CNN Inductive Bias Experiments

Experiments for the paper: **The Inductive Bias of Convolutional Neural Networks: Locality and Weight Sharing Reshape Implicit Regularization**

Our empirical study is demonstrated as a series of Jupyter notebooks comparing LCN-WS vs FCN under the **Boundary of Edge of Stability (BEoS)** constraint. All experiments were run on a single GPU.

---

## Notebooks

### 1. `flat_interpolation_experiment.ipynb`
**Flat Interpolation**

Demonstrates that architecture alone is insufficient for generalization. Even when LCN-WS satisfies BEoS, it can interpolate random labels when patches lack global structure. Each patch is sampled independently from the unit sphere, so the induced patch set has no shared structure across samples. Tracks training loss, Hessian sharpness ($\lambda_{\max}$), per-filter activation ratios, and weight norms. Produces training curves, sharpness plots with BEoS reference lines, activation scatter plots, and learning rate ablations.

**Default config:** `m=10`, `J=8`, `d=80`, `K=1024`, `N=512`, `σ=1.0`

**Key result:** BEoS is satisfied ($\lambda_{\max} \to 2/\eta$) while train loss drops below $\sigma^2$, confirming flat interpolation. Neurons specialize: activation rates drop from ~50% (random init) to selective patterns.

---

### 2. `generalization_experiment.ipynb`
**Generalization Gap (Sample Size & Dimension Ablation)**

Measures how the generalization gap $|\text{Excess}(f) + \sigma^2 - \hat{R}(f)|$ scales with training size $n$ and ambient dimension $d$ for LCN-WS vs FCN. Sweeps over $d \in \{50, 100, 200, 400\}$ and $n \in \{64, 128, 256, 512, 1024\}$ with 5 seeds each, training for 500K epochs per run. Produces log-log gap-vs-$n$ plots (with $n^{-1/4}$ reference slopes), gap-vs-$d$ plots, and combined training curves comparing LCN-WS and FCN sharpness dynamics. Results save incrementally for resume support.

**Key result:** LCN-WS generalization gap scales favorably with patch dimension $m$ rather than ambient dimension $d$, while FCN gap grows with $d$.

---

### 3. `cifar_regression_experiment.ipynb`
**CIFAR-10 Regression: FCN vs LCN-WS**

Compares generalization of FCN vs LCN-WS on a CIFAR-10 regression task with noisy labels (y = class_label + Gaussian noise, all 10 classes, n = 1024). Trains over multiple seeds and produces training curves (train loss and excess risk vs epochs with error bands) and activation analysis plots (path norm vs activation rate per filter).

**Key result:** LCN-WS achieves lower excess risk than FCN on real image data, consistent with the theoretical prediction that local weight sharing provides an inductive bias advantage.

---

### 4. `Toy_Experiment.ipynb`
**Synthetic Regression**

Generates data from a ground-truth network with sparse connectivity, parameter sharing, and global average pooling (cone-based projections). Trains multiple architectures across varying dimensions, kernel sizes, and widths to study convergence behavior on this synthetic task.

---

### 5. `cone_cluster_label_assignment.ipynb`
**Cone-Cluster Classification**

Implements a cone-cluster classification task on the unit sphere where inputs have one signal patch and noise patches drawn from different cluster centers. Compares signal-based vs random label assignments using a single-layer CNN with cross-entropy loss, and analyzes learned representations via GMM clustering, PCA/LDA visualization, and Fisher selectivity scoring.

---

### 6. `parameter_sharing_experiment.ipynb`
**Vocabulary Learning (Parameter Sharing Ablation)**

Studies how CNN (shared weights), locally-connected (unshared weights), and fully-connected architectures recover vocabulary vectors from high-dimensional noisy inputs with randomly partitioned patches. Tracks training curves and L2 error to isolate the effect of parameter sharing on generalization.

---

## Supporting Files

Both `flat_interpolation_experiment.ipynb` and `generalization_experiment.ipynb` depend on:

| File | Role |
|---|---|
| `synthetic_teacher_experiments.py` | Defines `SyntheticModel` (LCN-WS), `create_ground_truth_function`, `generate_views`, `ViewStrategy` enum |
| `src/__init__.py` | Package init for `src` module |
| `src/networks.py` | Defines `FullyConnectedBaseline` (FCN), `SharedCNN` |
| `src/data.py` | Sphere data generation utilities |
| `src/views.py` | View/patch extraction logic |
| `src/training.py` | Training loop utilities |
| `src/metrics.py` | Metric computation (excess risk, generalization gap, sharpness) |
| `pyhessian` *(optional)* | External library for exact Hessian computation; falls back to power iteration if not installed |

`cifar_regression_experiment.ipynb` is self-contained — only uses `torchvision` (CIFAR-10 data auto-downloaded to `data/`).

---

## Shared Architecture

**LCN-WS** (Locally Connected Network with Weight Sharing):
$$f(x) = \sum_k v_k \cdot \frac{1}{L} \sum_{\ell} \text{ReLU}(w_k^\top x^{(\ell)} - b_k) + \beta$$

**FCN** (Fully Connected Network):
$$f(x) = \sum_k v_k \cdot \text{ReLU}(w_k^\top x - b_k) + \beta$$

Key: LCN-WS operates in patch space $\mathbb{R}^m$ ($m \ll d$), while FCN sees the full ambient space $\mathbb{R}^d$.

**BEoS constraint:** $\lambda_{\max}(\nabla^2 \mathcal{L}(\theta)) \leq 2/\eta$, enforced implicitly by gradient descent with learning rate $\eta$.
