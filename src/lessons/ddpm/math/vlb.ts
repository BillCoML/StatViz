/**
 * Variational lower bound (VLB) pieces.
 *
 * The decomposition:
 *   L = L_T + sum_{t > 1} L_{t-1} + L_0
 *
 * with L_T = KL(q(x_T | x_0) || N(0, I)) and L_{t-1} a Gaussian KL.
 * Under sigma_t = beta_t and the epsilon-parameterization,
 * L_{t-1} reduces to a weighted MSE on epsilon prediction with weight
 *
 *   w_t = beta_t^2 / (2 * sigma_t^2 * alpha_t * (1 - alpha_bar_t))
 *       = beta_t / (2 * alpha_t * (1 - alpha_bar_t))     (when sigma_t^2 = beta_t)
 *
 * L_simple drops this weight (sets w_t = 1).
 */

import type { Schedule } from './schedule';
import type { SigmaChoice } from './reverse-process';

/** Per-timestep VLB weight on || eps - eps_theta ||^2. */
export function vlbWeight(t: number, sched: Schedule, sigma_choice: SigmaChoice = 'beta'): number {
  const bt = sched.betas[t];
  const at = sched.alphas[t];
  const ab = sched.alpha_bars[t];
  const sigma2 = sigma_choice === 'beta' ? sched.betas[t] : sched.tilde_betas[t];
  return (bt * bt) / (2 * sigma2 * at * (1 - ab));
}

/**
 * L_T per unit dimension (in nats).
 * KL(N(sqrt(ab_T) x_0, (1 - ab_T) I) || N(0, I)) per dim:
 *   = 0.5 * (sigma2 + mu^2 - 1 - log(sigma2))
 *
 * For unit-norm x_0 (mean ||x_0||^2 / d = 1), this collapses to the average.
 */
export function L_T_perDim(sched: Schedule, x0_meanSqNorm = 1): number {
  const T = sched.T;
  const ab = sched.alpha_bars[T - 1];
  const sigma2 = 1 - ab;
  // mean of mu^2 per dim is ab * (||x0||^2 / d)
  const mu2 = ab * x0_meanSqNorm;
  return 0.5 * (sigma2 + mu2 - 1 - Math.log(sigma2));
}

/** Convert nats per dimension to bits per dimension. */
export const NATS_TO_BITS = 1 / Math.log(2);

/**
 * Gaussian KL between two isotropic Gaussians with shared variance sigma^2:
 *   KL(N(mu_q, sigma^2 I) || N(mu_p, sigma^2 I)) = || mu_q - mu_p ||^2 / (2 sigma^2)
 */
export function isoSharedKL(mu_q: number[], mu_p: number[], sigma2: number): number {
  let s = 0;
  for (let i = 0; i < mu_q.length; i++) {
    const d = mu_q[i] - mu_p[i];
    s += d * d;
  }
  return s / (2 * sigma2);
}
