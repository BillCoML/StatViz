"""
Score model training script for the Score Matching lesson.
Trains one ScoreNet on a 4-cluster 2D GMM with a geometric noise schedule.
Outputs: src/lessons/score-matching/assets/score-weights.json
"""

import json
import math
import sys
import numpy as np
import torch
import torch.nn as nn

# ── 1. Data ──────────────────────────────────────────────────────────────────
np.random.seed(0)
torch.manual_seed(0)

CENTERS = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]], dtype=np.float32)
N_PER_CLUSTER = 250
X = np.vstack([
    np.random.normal(c, 0.2, (N_PER_CLUSTER, 2))
    for c in CENTERS
]).astype(np.float32)
Xt = torch.tensor(X)

# ── 2. Noise schedule ─────────────────────────────────────────────────────────
SIGMA_MAX, SIGMA_MIN, L = 2.0, 0.01, 10
sigmas = np.exp(np.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), L)).astype(np.float32)

# ── 3. Score network ──────────────────────────────────────────────────────────
HIDDEN = 64

class ScoreNet(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        # Input: concat(x, log(sigma)) → 3 dims
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, log_sigma.unsqueeze(-1)], dim=-1))


# ── 4. DSM loss with sigma^2 weighting ────────────────────────────────────────
def dsm_loss(model: ScoreNet, x_clean: torch.Tensor, sigmas_batch: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(x_clean)
    x_noisy = x_clean + sigmas_batch.unsqueeze(-1) * eps
    target = -eps / sigmas_batch.unsqueeze(-1)
    log_sigma = torch.log(sigmas_batch)
    pred = model(x_noisy, log_sigma)
    # Weight by sigma^2 so all noise levels contribute comparably
    weighted_err = (sigmas_batch.unsqueeze(-1) * (pred - target)) ** 2
    return weighted_err.sum(dim=-1).mean()


# ── 5. Training ───────────────────────────────────────────────────────────────
EPOCHS = 15000
BATCH  = 256

model = ScoreNet(HIDDEN)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-4)

print(f"Training for {EPOCHS} epochs...", flush=True)
for epoch in range(EPOCHS):
    idx        = np.random.choice(len(Xt), BATCH, replace=True)
    sigma_idx  = np.random.choice(L, BATCH)
    sig_batch  = torch.tensor(sigmas[sigma_idx])
    loss       = dsm_loss(model, Xt[idx], sig_batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    if (epoch + 1) % 2000 == 0:
        print(f"  epoch {epoch+1:5d}  loss={loss.item():.4f}", flush=True)

print("Training complete.", flush=True)

# ── 6. Save weights as JSON ───────────────────────────────────────────────────
model.eval()
weights: dict = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}
weights['_metadata'] = {
    'sigmas': sigmas.tolist(),
    'data_centers': CENTERS.tolist(),
    'hidden_dim': HIDDEN,
    'epochs': EPOCHS,
}

OUT = 'src/lessons/score-matching/assets/score-weights.json'
with open(OUT, 'w') as f:
    json.dump(weights, f, separators=(',', ':'))

size_kb = len(json.dumps(weights, separators=(',', ':')).encode()) / 1024
print(f"Saved {OUT}  ({size_kb:.1f} KB)", flush=True)

# ── 7. Visual inspection ──────────────────────────────────────────────────────
print("\nInspecting learned score field vs analytical ground truth...")

def logsumexp(a: list) -> float:
    m = max(a)
    return m + math.log(sum(math.exp(v - m) for v in a))

def analytical_score(x: np.ndarray, sigma: float) -> np.ndarray:
    """Score of the noise-smoothed GMM at noise level sigma."""
    pis = np.ones(4) / 4
    # smoothed covariance = 0.09*I + sigma^2*I = (0.09 + sigma^2)*I
    var = 0.09 + sigma ** 2
    log_ws = []
    scores = []
    for c in CENTERS:
        diff = x - c
        log_w = -0.5 * np.sum(diff ** 2) / var - math.log(2 * math.pi * var)
        log_ws.append(log_w + math.log(0.25))
        scores.append(-diff / var)
    log_norm = logsumexp(log_ws)
    rs = np.exp(np.array(log_ws) - log_norm)
    return sum(r * s for r, s in zip(rs, scores))  # type: ignore

def learned_score(x_np: np.ndarray, sigma: float) -> np.ndarray:
    with torch.no_grad():
        xT = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        lsT = torch.tensor([math.log(sigma)], dtype=torch.float32)
        return model(xT, lsT).squeeze(0).numpy()

# Check cosine similarity at 9 test points across 3 sigma levels
test_points = np.array([[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0],
                         [1.0, 1.0], [0.5, -0.5], [-1.5, 0.5],
                         [3.0, 0.0], [0.0, 3.0], [-3.0, -3.0]], dtype=np.float32)

all_ok = True
for sigma in [sigmas[0], sigmas[L // 2], sigmas[-1]]:
    cos_sims = []
    for pt in test_points:
        ana = analytical_score(pt, float(sigma))
        lea = learned_score(pt, float(sigma))
        norm_a = np.linalg.norm(ana)
        norm_l = np.linalg.norm(lea)
        if norm_a > 1e-6 and norm_l > 1e-6:
            cos = float(np.dot(ana, lea) / (norm_a * norm_l))
            cos_sims.append(cos)
    mean_cos = np.mean(cos_sims) if cos_sims else 0.0
    status = "OK" if mean_cos > 0.85 else "WARN"
    if mean_cos <= 0.85:
        all_ok = False
    print(f"  sigma={sigma:.4f}  mean cosine similarity={mean_cos:.3f}  [{status}]")

if all_ok:
    print("\nModel quality check PASSED.")
else:
    print("\nModel quality check has warnings — consider retraining with more epochs.")

# ── 8. Annealed Langevin in Python — cluster recovery ─────────────────────────
print("\nRunning annealed Langevin on 100 particles...")
np.random.seed(42)

N_PARTICLES = 100
T_INNER = 10
EPSILON_BASE = 0.5

sigma_L = float(sigmas[-1])
particles = np.random.normal(0, float(sigmas[0]), (N_PARTICLES, 2)).astype(np.float32)

for sigma in sigmas:
    sigma_f = float(sigma)
    alpha = EPSILON_BASE * (sigma_f / sigma_L) ** 2
    for _ in range(T_INNER):
        scores_np = np.array([learned_score(p, sigma_f) for p in particles])
        noise = np.random.randn(N_PARTICLES, 2).astype(np.float32)
        particles = particles + (alpha / 2) * scores_np + math.sqrt(alpha) * noise

# Measure cluster assignment
dists = np.array([[np.linalg.norm(p - c) for c in CENTERS] for p in particles])
assignments = dists.argmin(axis=1)
counts = np.bincount(assignments, minlength=4)
within_05 = (dists.min(axis=1) < 0.5).sum()

print(f"  Final cluster counts: {counts.tolist()}")
print(f"  Particles within 0.5 of nearest cluster center: {within_05}/{N_PARTICLES}")

if within_05 >= 70:
    print("  Annealed Langevin test PASSED.")
    sys.exit(0)
else:
    print("  Annealed Langevin test FAILED — too few particles converged.")
    sys.exit(1)
