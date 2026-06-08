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
* `theta` is a scale log-norm vector of size `1,000,000` that learns the feature magnitudes. Targeted weight decay is applied to `theta` to prevent scale inflation.

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
   We compute the Focal Loss (`gamma = 2.0`, `lam = 5.0`) only on these gathered active indices:
   `loss_active = weight_active * (pred_active - target_active)^2`
   Where `weight_active` dynamically balances class imbalances based on the number of positive targets in the current batch.

3. **Global Background Penalty (GBP):**
   To prevent unselected dimensions from firing falsely, we penalize the squared predictions outside the active index support:
   `gp_loss = (sum(pred^2) - sum(pred_active^2)) / V_vocab`

The total loss is the sum of the active loss and the global penalty: `loss_active + lambda_gp * gp_loss` (with `lambda_gp = 1e-3`).

---

## 2. Mathematical Feature State Classification (Diagnostics)

To analyze the trained checkpoint,  computes the following states for each of the 1,000,000 features:

### A. Global Bottleneck Projection Noise
The background interference noise in the bottleneck caused by the superposition of all inactive features has a standard deviation of `sigma_n`:

`sigma_n = sqrt( sum_j p_j * (1 - p_j) * ||w_j||_2^2 / d )`

Where `p_j` is the Zipfian prior probability of feature `j` and `d = 256` is the bottleneck dimension.

### B. Cramer-Rao Recovery Bound
A feature `i` is mathematically recoverable without loss of information (classified as **Healthy**) if and only if its column norm `||w_i||_2` exceeds the Cramer-Rao detection bound `tau_CR`:

`tau_CR(i) = sqrt( Gamma * sigma_eff(i) / p_i )`

Where `sigma_eff(i) = sqrt( sigma_n^2 + p_i * max_j ||w_j||_2^2 )` is the effective noise variance accounting for self-interference, and `Gamma = 3.0` is a statistical detection constant.

### C. Feature State Categories
* **Healthy (Fully Recoverable):** `||w_i||_2 >= tau_CR(i)`
* **Weak (Unrecoverable in isolation, but active):** `epsilon_dead(i) <= ||w_i||_2 < tau_CR(i)`
* **Dead (Collapsed/Inactive):** `||w_i||_2 < epsilon_dead(i)`
  Where `epsilon_dead(i)` represents the numerical precision floor under which gradient updates disappear into the machine epsilon (`1.19e-7` in FP32).
