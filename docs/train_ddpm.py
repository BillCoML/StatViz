"""Train DDPM model + produce 4 validation plots.

Run from project root:  python3 docs/train_ddpm.py
"""
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)

# --- Data ---------------------------------------------------------------
CENTERS = np.array([[2.0, 2.0], [2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0]])
N_per = 250
X = np.vstack([np.random.normal(c, 0.2, (N_per, 2)) for c in CENTERS]).astype(np.float32)
Xt = torch.tensor(X)

# --- Schedule -----------------------------------------------------------
T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
tilde_betas = (1 - alpha_bars_prev) / (1 - alpha_bars) * betas

# --- Architecture -------------------------------------------------------
HIDDEN = 64
TIME_DIM = 32

class EpsNet(nn.Module):
    def __init__(self, hidden=HIDDEN, time_dim=TIME_DIM):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
    def time_embed(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half).float() / (half - 1))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
    def forward(self, x, t):
        emb = self.time_embed(t)
        return self.net(torch.cat([x, emb], dim=-1))


def train(epochs=20000, batch=128, lr=1e-3, verbose=True):
    model = EpsNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        idx = np.random.choice(len(Xt), batch, replace=False)
        x_0 = Xt[idx]
        t = torch.randint(0, T, (batch,))
        eps = torch.randn_like(x_0)
        ab_t = alpha_bars[t].unsqueeze(-1)
        x_t = torch.sqrt(ab_t) * x_0 + torch.sqrt(1 - ab_t) * eps
        eps_pred = model(x_t, t)
        loss = ((eps - eps_pred) ** 2).sum(dim=-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if verbose and epoch % 2000 == 0:
            print(f'epoch {epoch}: loss {loss.item():.4f}')
    return model, losses


@torch.no_grad()
def reverse_sample(model, n=1000, return_trajectories=False):
    x = torch.randn(n, 2)
    traj = [x.clone()] if return_trajectories else None
    for ti in range(T - 1, -1, -1):
        t = torch.full((n,), ti, dtype=torch.long)
        eps_pred = model(x, t)
        a_t = alphas[ti]
        ab_t = alpha_bars[ti]
        mu = (1 / torch.sqrt(a_t)) * (x - betas[ti] / torch.sqrt(1 - ab_t) * eps_pred)
        if ti > 0:
            z = torch.randn_like(x)
            x = mu + torch.sqrt(betas[ti]) * z
        else:
            x = mu
        if return_trajectories:
            traj.append(x.clone())
    if return_trajectories:
        return x, torch.stack(traj)  # (T+1, n, 2)
    return x


@torch.no_grad()
def xhat0_trajectories(model, n=10):
    x = torch.randn(n, 2)
    xhats = []
    for ti in range(T - 1, -1, -1):
        t = torch.full((n,), ti, dtype=torch.long)
        eps_pred = model(x, t)
        ab_t = alpha_bars[ti]
        xhat = (x - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)
        xhats.append(xhat.clone())
        a_t = alphas[ti]
        mu = (1 / torch.sqrt(a_t)) * (x - betas[ti] / torch.sqrt(1 - ab_t) * eps_pred)
        if ti > 0:
            z = torch.randn_like(x)
            x = mu + torch.sqrt(betas[ti]) * z
        else:
            x = mu
    return torch.stack(xhats)  # (T, n, 2)


def save_weights(model, path):
    weights = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}
    weights['_metadata'] = {
        'T': T,
        'betas': betas.tolist(),
        'alpha_bars': alpha_bars.tolist(),
        'data_centers': CENTERS.tolist(),
        'hidden_dim': HIDDEN,
        'time_dim': TIME_DIM,
        'epochs': 20000,
        'schedule': 'linear-1e-4-2e-2',
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(weights))


def validate(model, plots_dir):
    import matplotlib.pyplot as plt
    def chisquare(counts):
        # Chi-squared vs uniform; df = len-1 = 3 → critical values
        # p computed approximately via incomplete gamma; we use exponential approx good enough here.
        n = sum(counts); k = len(counts); exp = n / k
        chi2 = sum((c - exp) ** 2 / exp for c in counts)
        # For df=3, regularized upper incomplete gamma(1.5, chi2/2)
        # Use math.gamma + numerical integral
        import math
        def upper_gamma_reg(s, x, steps=500):
            # Q(s, x) via upper-incomplete integral / Gamma(s)
            # crude trapezoid from x to x+30
            xs = np.linspace(x, x + 30, steps)
            ys = xs ** (s - 1) * np.exp(-xs)
            return np.trapezoid(ys, xs) / math.gamma(s)
        p = float(upper_gamma_reg(k / 2, chi2 / 2))
        return chi2, p
    plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Plot A: 1000 endpoints ----
    samples = reverse_sample(model, n=1000).numpy()
    # Assign each sample to nearest of 4 centers
    dists = np.linalg.norm(samples[:, None, :] - CENTERS[None, :, :], axis=-1)
    assigned = np.argmin(dists, axis=1)
    counts = np.bincount(assigned, minlength=4)
    chi2, p = chisquare(counts)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X[:, 0], X[:, 1], s=4, c='#b8651a', alpha=0.18, label='data')
    ax.scatter(samples[:, 0], samples[:, 1], s=4, c='#2c5f8d', alpha=0.5, label='samples')
    ax.scatter(CENTERS[:, 0], CENTERS[:, 1], s=80, c='black', marker='x')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    ax.set_title(f'A: 1000 reverse samples. counts={counts}, chi2={chi2:.2f}, p={p:.3f}')
    ax.legend()
    fig.savefig(plots_dir / 'A_endpoints.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot A: counts={counts}, chi2={chi2:.3f}, p={p:.4f}')

    # ---- Plot B: score field at five t's ----
    times = [1, 25, 50, 75, 99]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    grid = np.linspace(-4, 4, 16)
    XX, YY = np.meshgrid(grid, grid)
    pts = np.stack([XX.flatten(), YY.flatten()], axis=1).astype(np.float32)
    for ax, ti in zip(axes, times):
        with torch.no_grad():
            t_in = torch.full((len(pts),), ti, dtype=torch.long)
            eps_pred = model(torch.tensor(pts), t_in).numpy()
        ab_t = alpha_bars[ti].item()
        score = -eps_pred / np.sqrt(1 - ab_t)
        ax.quiver(pts[:, 0], pts[:, 1], score[:, 0], score[:, 1],
                  color='#2c5f8d', scale=80, width=0.003)
        ax.scatter(X[:, 0], X[:, 1], s=2, c='#b8651a', alpha=0.2)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
        ax.set_title(f't={ti}, ab={ab_t:.3f}')
    fig.suptitle('B: learned score field s = -eps/sqrt(1-ab)')
    fig.savefig(plots_dir / 'B_score_field.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- Plot C: x_hat_0 trajectories ----
    xhats = xhat0_trajectories(model, n=10).numpy()  # (T, 10, 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], s=4, c='#b8651a', alpha=0.15)
    for i in range(10):
        path = xhats[:, i, :]
        ax.plot(path[:, 0], path[:, 1], '-', alpha=0.5, lw=1)
        ax.scatter(path[-1, 0], path[-1, 1], s=30, c='#2c5f8d', zorder=5)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    ax.set_title('C: x_hat_0 trajectories (10 reverse samples)')
    fig.savefig(plots_dir / 'C_xhat0.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- Plot D: histogram of distance to nearest center ----
    nearest_dists = np.min(dists, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nearest_dists, bins=40, color='#2c5f8d', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', label='0.5')
    ax.set_xlabel('distance to nearest center')
    ax.set_ylabel('count')
    ax.set_title(f'D: nearest-center distance. median={np.median(nearest_dists):.3f}, frac<0.5={np.mean(nearest_dists < 0.5):.2f}')
    ax.legend()
    fig.savefig(plots_dir / 'D_dist_to_center.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot D: median dist {np.median(nearest_dists):.3f}, fraction<0.5 = {np.mean(nearest_dists < 0.5):.3f}')

    return {
        'counts': counts.tolist(),
        'chi2': float(chi2),
        'p': float(p),
        'median_dist': float(np.median(nearest_dists)),
        'frac_lt_0.5': float(np.mean(nearest_dists < 0.5)),
    }


if __name__ == '__main__':
    model, losses = train(epochs=20000)
    save_weights(model, 'src/lessons/ddpm/assets/ddpm-weights.json')
    print('Saved weights.')
    metrics = validate(model, plots_dir='docs/ddpm_validation_plots')
    print('Validation metrics:', metrics)
