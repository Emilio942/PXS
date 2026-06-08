import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import contextlib

# --- Dynamic CPU Threading & Affinity Configuration ---
try:
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        # Xeon Server: Socket 0 (16 threads)
        torch.set_num_threads(16)
        os.sched_setaffinity(0, set(list(range(0, 8)) + list(range(16, 24))))
        print("Optimized for Xeon Socket 0 (16 threads).")
    elif num_cpus == 16:
        # i5-12600KF: P-cores only (6 threads on physical P-cores)
        torch.set_num_threads(6)
        os.sched_setaffinity(0, {0, 2, 4, 6, 8, 10})
        print("Optimized for i5-12600KF P-cores (6 threads).")
    else:
        torch.set_num_threads(min(8, num_cpus or 4))
except Exception as e:
    pass

# --- Configuration (Apeiron Polysemantic Token Autoencoder - PTA v3.2) ---
V_VOCAB = 1_000_000      # 1M Feature/Word Space in Superposition
N_DIMENSIONS = 256       # 256-dimensional compressed bottleneck
BATCH_SIZE = 4           # Optimized batch size to fit in GPU VRAM (RTX 3060)
ACCUMULATION_STEPS = 16  # Increased to maintain exact effective batch size of 64
SEQ_LEN = 8              # Context window length ("thinking sequence")
LR = 1e-3
N_SHOTS = 3              # Multi-Shot Residual Recovery steps
DECAY_LAMBDA = 0.85      # Context aggregation decay factor
ZIPF_ALPHA = 1.07        # True natural language Zipf exponent
FREQ_WHITEN_LIMIT = 1000 # Whiten the top 1000 most frequent tokens fully, use diagonal for the rest
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "training_v3_2.log"
CHECKPOINT_FILE = "superposition_ae_1m_v3_2.pt"

class SoftExponential(nn.Module):
    """
    Soft-Exponential activation stabilized near alpha = 0.
    Dynamically gates noise and controls reconstruction contractivity.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))

    def forward(self, x):
        x_clamped = torch.clamp(x, min=-10.0, max=10.0)
        alpha_val = self.alpha.item()
        if abs(alpha_val) < 1e-4:
            x2 = x_clamped * x_clamped
            x3 = x2 * x_clamped
            x4 = x3 * x_clamped
            return x_clamped + self.alpha * (1.0 + 0.5 * x2 + (1.0 / 6.0) * self.alpha * x3 + (1.0 / 24.0) * (self.alpha ** 2) * x4)
        else:
            return (torch.exp(self.alpha * x_clamped) - 1.0) / self.alpha + self.alpha

class HybridWhitener:
    """
    Highly scalable Hybrid Whitening Pre-conditioner for 1M features.
    """
    def __init__(self, vocab_size, freq_limit, zipf_alpha, device):
        self.vocab_size = vocab_size
        self.freq_limit = freq_limit
        self.device = device
        
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=device)
        raw_probs = 1.0 / (ranks ** zipf_alpha)
        self.p = raw_probs / raw_probs.sum()
        
        print("Calculating hybrid covariance and inverse square-root whitener...")
        self.diag_std = torch.sqrt(self.p * (1.0 - self.p) + 1e-6)
        
        self.Sigma_freq = torch.eye(freq_limit, device=device) * 0.95
        for i in range(freq_limit - 1):
            correlation = 0.25 * (1.0 / (i + 1)**0.2)
            self.Sigma_freq[i, i+1] = correlation
            self.Sigma_freq[i+1, i] = correlation
            
        freq_std = self.diag_std[:freq_limit]
        self.Sigma_freq = self.Sigma_freq * torch.outer(freq_std, freq_std)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(self.Sigma_freq + torch.eye(freq_limit, device=device) * 1e-8)
        inv_sqrt_eigenvals = 1.0 / torch.sqrt(torch.clamp(eigenvalues, min=1e-9))
        self.Sigma_freq_inv_sqrt = torch.matmul(eigenvectors, torch.matmul(torch.diag(inv_sqrt_eigenvals), eigenvectors.T))
        
        self.p_cum = torch.cumsum(self.p, dim=0)
        print("Hybrid Whitener initialized successfully.")

    def whiten(self, x):
        """
        Applies pre-conditioned whitening.
        x: [batch_size, vocab_size]
        """
        x_freq = x[:, :self.freq_limit]
        x_rare = x[:, self.freq_limit:]
        
        x_freq_whitened = torch.matmul(x_freq, self.Sigma_freq_inv_sqrt.T)
        x_rare_whitened = x_rare / self.diag_std[self.freq_limit:].unsqueeze(0)
        
        return torch.cat([x_freq_whitened, x_rare_whitened], dim=-1)

class PolysemanticTokenAutoencoder(nn.Module):
    """
    Apeiron PTA v3.2: Anisotropic Log-Polar Parameterization
    Replaces W_norm with W = exp(theta) * v_norm.
    """
    def __init__(self, m, n, whitener, n_shots=3):
        super().__init__()
        self.n_shots = n_shots
        self.whitener = whitener
        
        # Learnable directions matrix V (Bfloat16 on CUDA to save 1.95 GB VRAM)
        dtype = torch.bfloat16 if whitener.device.type == "cuda" else torch.float32
        self.V = nn.Parameter(torch.empty(n, m, dtype=dtype))
        std = math.sqrt(2.94 / n)
        nn.init.normal_(self.V, mean=0.0, std=std)
        
        # Learnable Log-Norms theta initialized at Grassmannian equilibrium
        ranks = torch.arange(1, m + 1, dtype=torch.float32, device=whitener.device)
        raw_probs = 1.0 / (ranks ** ZIPF_ALPHA)
        p = raw_probs / raw_probs.sum()
        
        sum_p_sq = torch.sum(p ** 2).item()
        kappa = math.sqrt(10000.0 * n / sum_p_sq)
        equilibrium_norms = kappa * p
        self.theta = nn.Parameter(torch.log(equilibrium_norms))
        
        # Learnable Noise Suppressing Bias b
        p_clamped = torch.clamp(whitener.p, min=1e-7, max=1.0-1e-7)
        optimal_bias_init = -0.0225 * torch.log((1.0 - p_clamped) / p_clamped) - 0.5
        self.b = nn.Parameter(optimal_bias_init)
        self.g_star = SoftExponential(alpha=0.1)

    def compute_step(self, x_t, S_t_prev, V_norm, scales, b, alpha):
        x_whitened = self.whitener.whiten(x_t)
        h_t = nn.functional.linear(x_whitened * scales, V_norm)
        S_t = DECAY_LAMBDA * S_t_prev + h_t
        
        residual = S_t
        reconstruction = torch.zeros_like(x_t)
        
        for _ in range(self.n_shots):
            proj = torch.matmul(residual, V_norm) * scales
            x_clamped = torch.clamp(proj + b, min=-10.0, max=10.0)
            alpha_val = alpha.item()
            if abs(alpha_val) < 1e-4:
                x2 = x_clamped * x_clamped
                x3 = x2 * x_clamped
                x4 = x3 * x_clamped
                g_star_out = x_clamped + alpha * (1.0 + 0.5 * x2 + (1.0 / 6.0) * alpha * x3 + (1.0 / 24.0) * (alpha ** 2) * x4)
            else:
                g_star_out = (torch.exp(alpha * x_clamped) - 1.0) / alpha + alpha
                
            step_recon = torch.clamp(g_star_out, min=0.0, max=1.0)
            reconstruction = reconstruction + step_recon
            reconstruction_whitened = self.whitener.whiten(reconstruction)
            residual = S_t - nn.functional.linear(reconstruction_whitened * scales, V_norm)
            
        return reconstruction, S_t

    def forward(self, x_seq):
        """
        Processes a sequence of correlated tokens.
        x_seq: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, vocab_size = x_seq.shape
        
        # Extracted directions and scales
        V_norm = nn.functional.normalize(self.V, p=2, dim=0)
        scales = torch.exp(self.theta)                      # [m]
        
        S_t = torch.zeros(batch_size, N_DIMENSIONS, device=x_seq.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            # Checkpoint each sequence step to trade compute for VRAM
            reconstruction, S_t = torch.utils.checkpoint.checkpoint(
                self.compute_step,
                x_t, S_t, V_norm, scales, self.b, self.g_star.alpha,
                use_reentrant=False
            )
            outputs.append(reconstruction.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)

def generate_correlated_zipf_sequences(batch_size, seq_len, vocab_size, whitener):
    rand_vals = torch.rand(batch_size, seq_len, device=whitener.device)
    indices = torch.searchsorted(whitener.p_cum, rand_vals)
    indices = torch.clamp(indices, max=vocab_size - 1)
    x = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32, device=whitener.device)
    x.scatter_(2, indices.unsqueeze(-1), 1.0)
    correlated_mask = (torch.rand(batch_size, seq_len, device=whitener.device) < 0.30) & (indices < vocab_size - 1)
    partner_indices = torch.where(correlated_mask, indices + 1, indices)
    x.scatter_(2, partner_indices.unsqueeze(-1), 1.0)
    return x

def focal_regression_loss(pred, target, gamma=2.0, lam=5.0):
    r = pred - target
    abs_r = r.abs()
    c = 1.0 / (1.0 + lam * abs_r)
    w_neg = (1.0 - c) ** gamma
    # Mathematically calibrated dynamic class balancing based on true active support size
    num_pos = torch.clamp(target.sum(), min=1.0)
    w_pos = (target.numel() - num_pos) / num_pos
    weight = torch.where(target > 0.5, w_pos, w_neg)
    loss = weight * (r * r)
    return loss.mean()

def support_weighted_focal_loss(pred, target, gamma=2.0, lam=5.0, lambda_gp=1e-3):
    """
    Support-Weighted loss with Global Background Penalty (GBP): Restricts loss computation
    to the active support indices and their immediate neighbors, and adds a scaled background
    penalty on the unmonitored dimensions to prevent subspace collapse (leakage).
    
    Averages over target.numel() (via .mean()) to naturally apply Gradient Lipschitz Scaling
    and keep loss and gradient magnitudes identical to the stable global loss.
    """
    r = pred - target
    abs_r = r.abs()
    c = 1.0 / (1.0 + lam * abs_r)
    w_neg = (1.0 - c) ** gamma
    
    num_pos = torch.clamp(target.sum(), min=1.0)
    w_pos = (target.numel() - num_pos) / num_pos
    weight = torch.where(target > 0.5, w_pos, w_neg)
    loss = weight * (r * r)
    
    # Create mask for active support and its neighbors (i-1, i, i+1)
    mask = (target > 0.5).float()
    mask_left = torch.zeros_like(mask)
    mask_right = torch.zeros_like(mask)
    mask_left[..., :-1] = mask[..., 1:]
    mask_right[..., 1:] = mask[..., :-1]
    support_mask = torch.clamp(mask + mask_left + mask_right, max=1.0)
    
    # Restrict loss computation to the active support and neighbors (via mask)
    # Average over total vocabulary size (mean) to preserve gradient magnitude
    main_loss = (loss * support_mask).mean()
    
    # Global Background Penalty (GBP) on unmonitored dimensions
    inactive_mask = 1.0 - support_mask
    gp_loss = (pred * inactive_mask).pow(2).mean()
    
    # Combined loss
    return main_loss + lambda_gp * gp_loss

def archive_old_runs():
    import shutil
    from datetime import datetime
    
    archive_dir = "alt"
    files_to_archive = [LOG_FILE, CHECKPOINT_FILE]
    
    has_files_to_move = any(os.path.exists(f) for f in files_to_archive)
    if has_files_to_move:
        os.makedirs(archive_dir, exist_ok=True)
        for file_path in files_to_archive:
            if os.path.exists(file_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(os.path.basename(file_path))
                new_name = f"{base}_alt_{timestamp}{ext}"
                dest_path = os.path.join(archive_dir, new_name)
                
                try:
                    shutil.move(file_path, dest_path)
                    print(f"Archived existing file: '{file_path}' -> '{dest_path}'")
                except Exception:
                    pass

DECAY_LAMBDA = 0.85

def train():
    log_fd = open(LOG_FILE, "a")
    
    def log_print(msg):
        print(msg)
        log_fd.write(msg + "\n")
        log_fd.flush()
        try:
            os.fsync(log_fd.fileno())
        except Exception:
            pass

    log_print(f"--- Starting Apeiron PTA v3.2 | Device: {DEVICE} ---")
    log_print(f"V_vocab={V_VOCAB} | N_dim={N_DIMENSIONS} | Batch={BATCH_SIZE} | SeqLen={SEQ_LEN}")
    
    whitener = HybridWhitener(V_VOCAB, FREQ_WHITEN_LIMIT, ZIPF_ALPHA, DEVICE)
    model = PolysemanticTokenAutoencoder(V_VOCAB, N_DIMENSIONS, whitener).to(DEVICE)
    
    # --- MATHEMATICALLY CERTIFIED OPTIMIZATION ---
    decay_params = [model.theta]
    no_decay_params = [model.V, model.b, model.g_star.alpha]
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': 1e-4},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=LR)
    
    # --- RESUME CHECKPOINT LOGIC ---
    start_step = 0
    checkpoint_to_load = None
    
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_to_load = CHECKPOINT_FILE
        
    if checkpoint_to_load:
        log_print(f"Loading existing checkpoint '{checkpoint_to_load}' to resume training...")
        checkpoint = torch.load(checkpoint_to_load, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Ensure optimizer states match parameter devices and dtypes (excluding 'step' tensor) to avoid mismatches
        for p, state in optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if k == 'step':
                        state[k] = v.to(dtype=torch.float32)
                    else:
                        state[k] = v.to(device=p.device, dtype=p.dtype)
        start_step = checkpoint['step'] + 1
        log_print(f"Resuming training dynamically from Step {start_step}...")
        del checkpoint
        import gc
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log_print("No checkpoint found. Starting training from scratch (Step 0).")
        archive_old_runs()
    
    # Set up Automatic Mixed Precision (AMP) autocast context manager if running on GPU
    if DEVICE.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        log_print("Using CUDA bfloat16 Mixed Precision Autocast.")
    else:
        autocast_ctx = contextlib.nullcontext()
        log_print("Using CPU standard FP32 Eager Execution.")
    
    step_times = []
    
    for step in range(start_step, 5001):
        t_start = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        # 1. Accumulate Reconstruction Loss
        accumulated_recon_loss = 0.0
        for _ in range(ACCUMULATION_STEPS):
            x_sub = generate_correlated_zipf_sequences(BATCH_SIZE, SEQ_LEN, V_VOCAB, whitener)
            
            with autocast_ctx:
                x_hat_sub = model(x_sub)
                reconstruction_loss_sub = support_weighted_focal_loss(x_hat_sub, x_sub) / ACCUMULATION_STEPS
            
            reconstruction_loss_sub.backward()
            accumulated_recon_loss += reconstruction_loss_sub.item() * ACCUMULATION_STEPS
            
        # 2. Compute Coherence Loss exactly ONCE per global step
        sample_idx = torch.randint(0, V_VOCAB, (1000,), device=DEVICE)
        with autocast_ctx:
            V_sample = model.V[:, sample_idx]
            V_norm = nn.functional.normalize(V_sample, p=2, dim=0)
            coherence_matrix = torch.matmul(V_norm.T, V_norm)
            
            collision_mask = (sample_idx.unsqueeze(0) == sample_idx.unsqueeze(1)).float()
            coherence_matrix = coherence_matrix * (1.0 - collision_mask)
            coherence_loss = (coherence_matrix ** 2).mean()
            
            # Hinge coherence penalty starting above 0.35 limit
            hinge_coherence = torch.clamp(coherence_matrix.abs() - 0.35, min=0.0).pow(2).mean()
            coherence_loss_scaled = 0.1 * coherence_loss + 1.0 * hinge_coherence
            
        coherence_loss_scaled.backward()
        
        total_loss = accumulated_recon_loss + coherence_loss_scaled.item()
        
        # 3. Step the optimizer
        optimizer.step()
        
        # 4. Clamping parameter alpha to safe range [-0.4, 0.4]
        with torch.no_grad():
            model.g_star.alpha.clamp_(min=-0.4, max=0.4)
            
        t_end = time.time()
        step_times.append(t_end - t_start)
        
        if step % 1 == 0:
            with torch.no_grad():
                heat = (coherence_matrix ** 2).sum().item()
                max_int = coherence_matrix.abs().max().item()
                alpha_val = model.g_star.alpha.item()
                avg_step_time = sum(step_times[-50:]) / min(len(step_times), 50)
                current_step_time = step_times[-1]
                
                # Fetch scale statistics
                scales = torch.exp(model.theta)
                mean_scale = scales.mean().item()
                min_scale = scales.min().item()
                max_scale = scales.max().item()
                
            status = (f"Step {step:4d} | "
                      f"Total Loss: {total_loss:.6f} | "
                      f"Recon: {accumulated_recon_loss:.6f} | "
                      f"Scales (Mean/Min/Max): {mean_scale:.4f}/{min_scale:.6f}/{max_scale:.2f} | "
                      f"Heat: {heat:.2f} | "
                      f"MaxInt: {max_int:.4f} | "
                      f"Alpha: {alpha_val:.6f} | "
                      f"Time: {current_step_time:.3f}s")
            log_print(status)

        if step % 500 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, CHECKPOINT_FILE)
            try:
                fd = os.open(CHECKPOINT_FILE, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except Exception:
                pass
            log_print(f"Checkpoint saved at step {step}")
            
        # Explicit garbage collection
        import gc
        gc.collect()

    log_fd.close()

if __name__ == "__main__":
    train()
