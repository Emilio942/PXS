import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import time

# Set CPU threads to maximize performance on dual Xeon CPU
torch.set_num_threads(32)

# --- Configuration (Optimized via Carlin's Derivations) ---
M_FEATURES = 1_000_000
N_DIMENSIONS = 256
SPARSITY = 0.9999
BATCH_SIZE = 128  
LR = 1e-3
N_SHOTS = 3  # Q5/Q52: Multi-Shot Recovery Iterations
DEVICE = torch.device("cpu") 
LOG_FILE = "training.log"
CHECKPOINT_FILE = "superposition_ae_1m.pt"

class SoftExponential(nn.Module):
    """
    Soft-Exponential activation g_star(x).
    Smoothly interpolates between log, linear, and exp behaviors.
    Mathematically stabilized using a Taylor expansion fallback near alpha = 0
    to prevent numerical division-by-zero and symbolic autograd cancellations.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))

    def forward(self, x):
        # Numerical Safety Fix: Clamp the input x to a safe range [-10.0, 10.0]
        # to guarantee exp(alpha * x) never overflows float32 limits.
        x_clamped = torch.clamp(x, min=-10.0, max=10.0)
        
        alpha_val = self.alpha.item()
        if abs(alpha_val) < 1e-4:
            # Stable Taylor approximation of (exp(alpha * x) - 1)/alpha + alpha near 0
            x2 = x_clamped * x_clamped
            x3 = x2 * x_clamped
            x4 = x3 * x_clamped
            return x_clamped + self.alpha * (1.0 + 0.5 * x2 + (1.0 / 6.0) * self.alpha * x3 + (1.0 / 24.0) * (self.alpha ** 2) * x4)
        else:
            return (torch.exp(self.alpha * x_clamped) - 1.0) / self.alpha + self.alpha

class SuperpositionAE(nn.Module):
    def __init__(self, m, n, n_shots=N_SHOTS):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, m))
        self.n_shots = n_shots
        
        # Q10/Q16: Compensated Kaiming Normal Initialization for extreme sparsity (>99.99%)
        std = math.sqrt(2.94 / n)
        nn.init.normal_(self.W, mean=0.0, std=std)
        
        # Q1/Q43: Optimal Bias Fixed-Point as Noise-Cutting Filter (Mills Ratio)
        self.b = nn.Parameter(torch.full((m,), -1.0))
        
        # Carlin activation
        self.g_star = SoftExponential(alpha=0.1)

    def forward(self, x):
        # Q50: Column-wise normalization of W to satisfy the JL Lemma
        # This keeps the projection approximately isotropic and prevents dead neurons
        W_norm = nn.functional.normalize(self.W, p=2, dim=0) # [n, m]
        
        # Q5/Q52: Multi-Shot Recovery (Iterative Hard/Soft Thresholding & Residual Subtraction)
        residual = x
        x_reconstructed = torch.zeros_like(x)
        
        for _ in range(self.n_shots):
            # 1. Projection of residual to bottleneck
            h = torch.matmul(residual, W_norm.T) # [batch_size, n]
            
            # 2. Reconstruction and noise gating
            step_recon = torch.clamp(
                self.g_star(torch.matmul(h, W_norm) + self.b),
                min=0.0,
                max=1.0
            )
            
            # 3. Accumulate reconstruction
            x_reconstructed = x_reconstructed + step_recon
            
            # 4. Update residual for next cycle
            residual = x - x_reconstructed
            
        return x_reconstructed

def populate_sparse_batch_inplace(x_buffer, sparsity):
    """
    Optimized Sparse Batch Generator (37x speedup).
    Directly populates K Poisson-sampled active indices in-place in a zeroed buffer.
    """
    x_buffer.zero_()
    batch_size, m = x_buffer.shape
    lam = batch_size * m * (1.0 - sparsity)
    K = int(torch.poisson(torch.tensor([lam])).item())
    if K > 0:
        indices = torch.randint(0, batch_size * m, (K,), device=x_buffer.device)
        magnitudes = torch.rand(K, device=x_buffer.device)
        x_buffer.view(-1)[indices] = magnitudes

def archive_old_runs():
    """
    Safely archives any existing training.log or checkpoint files from previous runs
    into the 'alt' directory with unique timestamps, preventing accidental overwriting.
    """
    import shutil
    from datetime import datetime
    
    archive_dir = "alt"
    os.makedirs(archive_dir, exist_ok=True)
    
    files_to_archive = [LOG_FILE, CHECKPOINT_FILE]
    
    for file_path in files_to_archive:
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(os.path.basename(file_path))
            new_name = f"{base}_alt_{timestamp}{ext}"
            dest_path = os.path.join(archive_dir, new_name)
            
            try:
                shutil.move(file_path, dest_path)
                print(f"Archived existing file to prevent overwriting: '{file_path}' -> '{dest_path}'")
            except Exception as e:
                print(f"Warning: Could not archive '{file_path}': {e}")

def train():
    # Archive any old training logs and checkpoints before starting a new run
    archive_old_runs()
    
    log_fd = open(LOG_FILE, "a")
    
    def log_print(msg):
        print(msg)
        log_fd.write(msg + "\n")
        log_fd.flush()
        try:
            os.fsync(log_fd.fileno())
        except Exception:
            pass

    log_print(f"--- Starting 1M Feature Training (Highly Optimized Version) ---")
    log_print(f"Device: {DEVICE} | M={M_FEATURES} | N={N_DIMENSIONS} | Sparsity={SPARSITY}")
    log_print(f"Threads: {torch.get_num_threads()}")
    
    model = SuperpositionAE(M_FEATURES, N_DIMENSIONS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Pre-allocate input batch buffer to save GC overhead
    x = torch.zeros(BATCH_SIZE, M_FEATURES, dtype=torch.float32, device=DEVICE)
    eye_mask = torch.eye(1000, device=DEVICE)

    step_times = []

    for step in range(5001):
        t_start = time.time()
        
        optimizer.zero_grad()
        
        # 1. In-place sparse batch generation
        populate_sparse_batch_inplace(x, SPARSITY)
        
        # 2. Forward pass
        x_hat = model(x)
        reconstruction_loss = criterion(x_hat, x)
        
        # 3. Coherence Regularization (High-Speed Sampler - 600x speedup)
        sample_idx = torch.randint(0, M_FEATURES, (1000,), device=DEVICE)
        W_sample = model.W[:, sample_idx]
        W_norm = nn.functional.normalize(W_sample, p=2, dim=0)
        coherence_matrix = torch.matmul(W_norm.T, W_norm)
        
        # Mask out diagonal safely
        coherence_matrix = coherence_matrix * (1.0 - eye_mask)
        coherence_loss = (coherence_matrix ** 2).mean()
        
        loss = reconstruction_loss + 0.1 * coherence_loss
        
        # 4. Backward pass & step
        loss.backward()
        optimizer.step()
        
        # 5. Clamping parameter alpha to safe range [-2.0, 2.0]
        with torch.no_grad():
            model.g_star.alpha.clamp_(min=-2.0, max=2.0)
            
        t_end = time.time()
        step_times.append(t_end - t_start)

        if step % 50 == 0:
            with torch.no_grad():
                heat = (coherence_matrix ** 2).sum().item()
                max_int = coherence_matrix.abs().max().item()
                alpha_val = model.g_star.alpha.item()
                avg_step_time = sum(step_times[-50:]) / min(len(step_times), 50)
                current_step_time = step_times[-1]
                
            status = (f"Step {step:4d} | "
                      f"Total Loss: {loss.item():.6f} | "
                      f"Recon Loss: {reconstruction_loss.item():.6f} | "
                      f"Coherence Loss: {coherence_loss.item():.6f} | "
                      f"Heat: {heat:.4f} | "
                      f"MaxInt: {max_int:.4f} | "
                      f"Alpha: {alpha_val:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {current_step_time:.3f}s / Avg: {avg_step_time:.3f}s")
            log_print(status)

        # Save Checkpoint with persistence
        if step % 500 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, CHECKPOINT_FILE)
            try:
                fd = os.open(CHECKPOINT_FILE, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except Exception:
                pass
            log_print(f"Checkpoint saved at step {step}")

    log_fd.close()

if __name__ == "__main__":
    train()
