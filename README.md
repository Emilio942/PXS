# Apeiron Polysemantic Token Autoencoder (PTA v3.2)

This repository contains the implementation of the **Apeiron Polysemantic Token Autoencoder (PTA v3.2)**. The model is designed to compress an extremely sparse representation space of **1,000,000 features/words** (vocab space) into a dense, **256-dimensional bottleneck** (bottleneck space) by leveraging polysemantic superposition.

---

## 1. Mathematical Foundations

The model is based on the physical insight that neural networks can store significantly more features than dimensions (superposition) if the features are sparse and nearly orthogonal to one another.

### A. Anisotropic Log-Polar Parameterization
To decouple feature directions from their corresponding scales, the dense weight matrix `W` (of size `256 x 1,000,000`) is parameterized as:

`W = exp(theta) * V_norm`

Where:
* `V` is a direction matrix of size `256 x 1,000,000`. The columns of `V` are normalized on the unit sphere: `V_norm = V / ||V||_2`. These represent directions on the Grassmann manifold. No weight decay is applied to `V`.
* `theta` is a scale log-norm vector of size `1,000,000` that learns the feature magnitudes. 
* *Note on Initialization:* To prevent scale explosion and features from becoming permanently trapped below the activation threshold, scales must be initialized near $1.0$ (e.g. $\theta_i \approx 0.0$) rather than proportional to the highly skewed Zipfian prior probabilities.

---

### B. Multi-Shot Residual Recovery ("Thinking" Loop)
Since rare features in the bottleneck are obscured by the background interference noise of frequent features, the model employs an iterative reconstruction loop over `n_shots = 3` steps:

1. **Input Projection (Whitening & Encoding):**
   The input vector `x_t` (at a single sequence step `t`) is whitened to remove correlations: `x_whitened = Whitener(x_t)`.
   The bottleneck activation `S_t` is computed incorporating temporal context (decay factor `decay_lambda = 0.85`):
   
   `S_t = decay_lambda * S_t_prev + linear(x_whitened * exp(theta), V_norm)`

2. **Iterative Reconstruction (Multi-Shot):**
   For `k = 0` to `n_shots - 1`, the model computes:
   * **Projection of the current residual:**
     `proj = matmul(residual_k, V_norm) * exp(theta)`
   * **Activation Gating (Soft-Exponential):**
     `step_recon = clamp(SoftExponential(proj + b), min=0.0, max=1.0)`
   * **Update the reconstruction:**
     `reconstruction_k+1 = reconstruction_k + step_recon`
   * **Update the residual:**
     `residual_k+1 = S_t - linear(Whitener(reconstruction_k+1) * exp(theta), V_norm)`

By using this iterative residual loop, strong signals are reconstructed first, and their mutual interference is subtracted from the residual. This allows quiet, rare signals to emerge and activate in later steps.

---

### C. Stabilized Soft-Exponential Activation Function
The activation function controls noise gating and is Taylor-stabilized near `alpha = 0` to prevent division by zero:

* **Standard Formula (for |alpha| >= 1e-4):**
  `f(x, alpha) = (exp(alpha * x) - 1) / alpha + alpha`
  
* **Taylor Approximation (for |alpha| < 1e-4):**
  `f(x, alpha) = x + alpha * (1 + 0.5 * x^2 + (1/6) * alpha * x^3 + (1/24) * alpha^2 * x^4)`

A converged value of `alpha > 0` makes the activation function slightly exponential for positive inputs (amplifying sparse signals) and strongly damped for negative inputs (suppressing noise).

---

### D. Support-Weighted Focal Loss & Global Background Penalty (GBP)
A naive loss computation over the entire 1M-dimensional vocabulary space is computationally intractable. Instead, the model uses a highly optimized sparse loss function:

1. **Active Index Support:**
   For each active token index `idx` in the batch, we statically define 4 candidate indices:
   `idx - 1`, `idx`, `idx + 1`, and `partner_idx + 1` (for correlated pairs).
   
2. **Support-Weighted Loss:**
   We compute the Focal Loss (`gamma = 2.0`, `lam = 5.0`) only on these gathered active indices. Crucially, the losses must be normalized independently of the vocabulary size to prevent active/background gradient mismatch:
   
   `loss_active = (1 / N_pos) * sum_{i in active} FocalLoss(pred_i, target_i)`

3. **Global Background Penalty (GBP):**
   To prevent unselected dimensions from firing falsely, we penalize the squared predictions outside the active index support, scaled independently:
   
   `loss_background = (1 / N_neg) * sum_{i in inactive} pred_i^2`

The total loss is the sum of the active loss and the global penalty: `loss_active + lambda_gp * loss_background` (with `lambda_gp = 1e-3`).

---

## 2. Mathematical Feature State Classification (Diagnostics)

To analyze the trained checkpoint, [analyze_checkpoint.py](file:///home/emilio/Documents/ai/polysemantic_superposition/analyze_checkpoint.py) computes the following states for each of the 1,000,000 features:

### A. Global Bottleneck Projection Noise
The background interference noise in the bottleneck caused by the superposition of all inactive features has a standard deviation of `sigma_n`:

`sigma_n = sqrt( sum_j p_j * (1 - p_j) * ||w_j||_2^2 / d )`

Where `p_j` is the Zipfian prior probability of feature `j` and `d = 256` is the bottleneck dimension.

### B. Cramer-Rao Recovery Bound
A feature `i` is mathematically recoverable without loss of information (classified as **Healthy**) if and only if its column norm `||w_i||_2` exceeds the Cramer-Rao detection bound `tau_CR`:

`tau_CR(i) = sqrt( Gamma * sigma_eff(i) / p_i )`

Where `sigma_eff(i) = sqrt( sigma_n^2 + p_i * max_j ||w_j||_2^2 )` is the effective noise variance accounting for self-interference, and `Gamma = 3.0` is a statistical detection constant.

### C. Feature State Categories
* **Healthy (Fully Recoverable):** `||w_i||_2 >= tau_CR(i)` and `||w_i||_2 >= sqrt(|b_i|)`
* **Weak (Unrecoverable in isolation, but active):** `epsilon_dead(i) <= ||w_i||_2 < min(tau_CR(i), sqrt(|b_i|))`
* **Dead (Collapsed/Inactive):** `||w_i||_2 < epsilon_dead(i)`
  Where `epsilon_dead(i)` represents the numerical precision floor under which gradient updates disappear into the machine epsilon (`1.19e-7` in FP32).

> [!WARNING]
> Entgegen linearer Näherungen dürfen seltene Features im Zipf-Tail im nicht-linearen Autoencoder keine winzigen Normen (wie $10^{-6}$) besitzen. Sie benötigen eine Mindestnorm von $\approx \sqrt{|b_i|} \approx 0.9$, um den Rausch-Bias $b_i$ bei Aktivierung zu überwinden. Eine Initialisierung proportional zur Zipf-Wahrscheinlichkeit ($r_i \propto p_i$) führt zum sofortigen Absterben des gesamten Tails.

---

## 3. CPU Inference Optimization (Dual Sparse CSR Matrices)

For CPU inference under strict memory limitations (< 200 MB) and to bypass the system DDR RAM bandwidth bottleneck, the model utilizes **Dual Sparse CSR Matrices** combined with **Activation-Sparsified Gating**.

### A. Dual CSR Weight Representation
At `99%` sparsity (keeping only the top 1% largest absolute weights of `V_norm` and setting the rest to exactly 0), the direction matrix `V_norm` (size `[256, 1000000]`) is stored as two complementary Compressed Sparse Row (CSR) matrices:
* `V_sparse` (shape `[256, 1000000]`): Used for the residual update step (multiplying sparse reconstruction by the dictionary).
* `V_T_sparse` (shape `[1000000, 256]`): Used for the projection step (multiplying the bottleneck residual by the transposed dictionary).

The combined memory footprint of these two matrices is **42.88 MB**, which fits entirely within the CPU's L3 cache, avoiding slow system RAM reads.

### B. Activation-Sparsified Gating
Since the activation function is clamped to `[0, 1]`, features only reconstruct if `g_star(proj_i + b_i) > 0`. 
Solving this inequality for `g_star(y) = 0` yields:
`y_threshold = ln(1 - alpha^2) / alpha` (which is `-0.1005` for `alpha = 0.1`).

Therefore, a feature `i` can only be active if:
`proj_i + b_i > y_threshold`

We utilize this mask (`val > y_threshold`) to compute `soft_exponential` only on the active elements (typically `< 50` out of `1,000,000` features per batch element). This speeds up the activation step by over **20,000x**.
