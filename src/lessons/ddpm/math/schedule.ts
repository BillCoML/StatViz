/**
 * Noise schedules for DDPM.
 *
 * A schedule is the sequence beta_1, ..., beta_T (here zero-indexed:
 * betas[t] for t = 0, ..., T-1) plus derived quantities used throughout
 * the lesson:
 *
 *   alpha_t       = 1 - beta_t
 *   alpha_bar_t   = prod_{s <= t} alpha_s
 *   tilde_beta_t  = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
 *
 * tilde_beta is the forward-posterior variance q(x_{t-1} | x_t, x_0).
 * It satisfies tilde_beta_t <= beta_t with equality at large t.
 */

export interface Schedule {
  T: number;
  betas: number[];
  alphas: number[];
  alpha_bars: number[];
  alpha_bars_prev: number[];
  tilde_betas: number[];
}

function deriveFromBetas(betas: number[]): Schedule {
  const T = betas.length;
  const alphas = betas.map(b => 1 - b);
  const alpha_bars: number[] = [];
  let cum = 1;
  for (const a of alphas) { cum *= a; alpha_bars.push(cum); }
  const alpha_bars_prev = [1, ...alpha_bars.slice(0, -1)];
  const tilde_betas = betas.map((b, t) =>
    (1 - alpha_bars_prev[t]) / (1 - alpha_bars[t]) * b
  );
  return { T, betas, alphas, alpha_bars, alpha_bars_prev, tilde_betas };
}

/** Linear schedule from beta_start (at t=0) to beta_end (at t=T-1). */
export function linearSchedule(T: number, beta_start: number, beta_end: number): Schedule {
  const betas = Array.from({ length: T }, (_, i) =>
    beta_start + (beta_end - beta_start) * i / (T - 1)
  );
  return deriveFromBetas(betas);
}

/**
 * Cosine schedule from Nichol & Dhariwal (2021).
 * alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi/2) / cos^2(s/(1+s) * pi/2).
 * Yields a more uniform information destruction curve.
 */
export function cosineSchedule(T: number, s = 0.008): Schedule {
  const f = (t: number) => Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2) ** 2;
  const f0 = f(0);
  const alphaBarRaw = Array.from({ length: T + 1 }, (_, i) => f(i) / f0);
  const betas = Array.from({ length: T }, (_, i) => {
    const b = 1 - alphaBarRaw[i + 1] / alphaBarRaw[i];
    return Math.min(Math.max(b, 1e-6), 0.999);
  });
  return deriveFromBetas(betas);
}

/** Sigmoid schedule: smoothly transitions between beta_start and beta_end. */
export function sigmoidSchedule(T: number, beta_start: number, beta_end: number, k = 6): Schedule {
  const sig = (u: number) => 1 / (1 + Math.exp(-u));
  const lo = sig(-k / 2);
  const hi = sig(k / 2);
  const betas = Array.from({ length: T }, (_, i) => {
    const u = -k / 2 + k * i / (T - 1);
    const s = (sig(u) - lo) / (hi - lo);
    return beta_start + (beta_end - beta_start) * s;
  });
  return deriveFromBetas(betas);
}

/** The DDPM paper's schedule: T=1000, beta linear from 1e-4 to 0.02. */
export function paperSchedule(): Schedule {
  return linearSchedule(1000, 1e-4, 0.02);
}

/** Browser-affordable schedule for §10: T=100, beta linear from 1e-4 to 0.02. */
export function browserSchedule(): Schedule {
  return linearSchedule(100, 1e-4, 0.02);
}
