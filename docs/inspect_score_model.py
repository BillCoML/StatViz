"""Inspect the trained score model quality and run annealed Langevin."""
import json
import math
import sys
import numpy as np
import torch
import torch.nn as nn

HIDDEN = 64
CENTERS = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]], dtype=np.float32)

class ScoreNet(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, log_sigma.unsqueeze(-1)], dim=-1))

# Load weights
with open('src/lessons/score-matching/assets/score-weights.json') as f:
    data = json.load(f)

sigmas = np.array(data['_metadata']['sigmas'], dtype=np.float32)
L = len(sigmas)

model = ScoreNet(HIDDEN)
sd = model.state_dict()
for k in sd:
    sd[k] = torch.tensor(data[k])
model.load_state_dict(sd)
model.eval()

def logsumexp(a: list) -> float:
    m = max(a)
    return m + math.log(sum(math.exp(v - m) for v in a))

def analytical_score(x: np.ndarray, sigma: float) -> np.ndarray:
    var = 0.04 + sigma ** 2  # cluster std=0.2, so cluster var=0.04
    log_ws = []
    scores = []
    for c in CENTERS:
        diff = (x - c).astype(np.float64)
        log_w = -0.5 * float(np.sum(diff ** 2)) / var - math.log(2 * math.pi * var)
        log_ws.append(log_w + math.log(0.25))
        scores.append(-diff / var)
    log_norm = logsumexp(log_ws)
    rs = np.exp(np.array(log_ws) - log_norm)
    return sum(r * s for r, s in zip(rs, scores)).astype(np.float32)  # type: ignore

def learned_score(x_np: np.ndarray, sigma: float) -> np.ndarray:
    with torch.no_grad():
        xT = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        lsT = torch.tensor([math.log(sigma)], dtype=torch.float32)
        return model(xT, lsT).squeeze(0).numpy()

# Visual inspection — cosine similarity
test_points = np.array([
    [2.0,  2.0], [2.0, -2.0], [-2.0,  2.0], [-2.0, -2.0],  # cluster centers
    [0.0,  0.0],                                              # saddle
    [1.0,  0.0], [-1.0, 0.0], [0.0,  1.0], [0.0,  -1.0],    # between modes
    [3.0,  3.0], [-3.0, -3.0],                               # far out
], dtype=np.float32)

all_ok = True
for level_name, sigma in [('sigma_max', float(sigmas[0])),
                           ('sigma_mid', float(sigmas[L//2])),
                           ('sigma_min', float(sigmas[-1]))]:
    cos_sims = []
    for pt in test_points:
        ana = analytical_score(pt, sigma)
        lea = learned_score(pt, sigma)
        norm_a = np.linalg.norm(ana)
        norm_l = np.linalg.norm(lea)
        if norm_a > 1e-6 and norm_l > 1e-6:
            cos = float(np.dot(ana, lea) / (norm_a * norm_l))
            cos_sims.append(cos)
    mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
    status = "OK" if mean_cos > 0.85 else "WARN"
    if mean_cos <= 0.85:
        all_ok = False
    print(f"  {level_name:12s} (sigma={sigma:.4f})  mean cosine sim={mean_cos:.3f}  [{status}]")

if not all_ok:
    print("\nQuality check has warnings.")

# Annealed Langevin recovery
print("\nRunning annealed Langevin (100 particles, T=10 inner steps)...")
np.random.seed(42)
N_PARTICLES = 100
T_INNER = 100          # more steps for reliable convergence in Python test
# NCSN-calibrated epsilon: alpha_1 = 0.2 at sigma_max=2.0
# alpha = epsilon * (sigma/sigma_L)^2 → epsilon = 0.2 / (2.0/0.01)^2 = 5e-6
EPSILON_BASE = 5e-6
sigma_L = float(sigmas[-1])
sigma_1 = float(sigmas[0])
particles = np.random.normal(0, sigma_1, (N_PARTICLES, 2)).astype(np.float32)

print(f"  Step sizes: alpha_max={EPSILON_BASE*(sigma_1/sigma_L)**2:.4f}, alpha_min={EPSILON_BASE:.2e}")

for sigma in sigmas:
    sf = float(sigma)
    alpha = EPSILON_BASE * (sf / sigma_L) ** 2
    for _ in range(T_INNER):
        scores_np = np.array([learned_score(p, sf) for p in particles])
        noise = np.random.randn(N_PARTICLES, 2).astype(np.float32)
        particles = particles + (alpha / 2) * scores_np + math.sqrt(alpha) * noise

dists = np.array([[np.linalg.norm(p - c) for c in CENTERS] for p in particles])
assignments = dists.argmin(axis=1)
counts = np.bincount(assignments, minlength=4)
within_05 = int((dists.min(axis=1) < 0.5).sum())
within_10 = int((dists.min(axis=1) < 1.0).sum())
within_15 = int((dists.min(axis=1) < 1.5).sum())

print(f"  Cluster counts: {counts.tolist()}")
print(f"  Within 0.5 of nearest center: {within_05}/100")
print(f"  Within 1.0 of nearest center: {within_10}/100")
print(f"  Within 1.5 of nearest center: {within_15}/100")

balanced = all(c >= 15 for c in counts)
print(f"  Roughly balanced across clusters: {balanced}")

if within_10 >= 70 and balanced:
    print("\nALL CHECKS PASSED — model ready.")
    sys.exit(0)
else:
    print(f"\nFAILED — within_10={within_10}/100, balanced={balanced}.")
    sys.exit(1)
