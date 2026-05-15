/**
 * Reverse process: one Algorithm 2 step.
 *
 *   x_{t-1} = mu_theta(x_t, t) + sigma_t * z, z ~ N(0, I)
 *
 * with mu_theta in the epsilon parameterization. The Langevin-style noise
 * injection sigma_t * z vanishes on the final step (t = 0).
 *
 * Two sigma choices from the paper:
 *   - 'beta':       sigma_t^2 = beta_t            (works as well as tilde_beta empirically)
 *   - 'tilde_beta': sigma_t^2 = tilde_beta_t      (theoretically optimal under known x_0)
 */

import type { Schedule } from './schedule';
import { posteriorMeanFromEps } from './forward-process';

export type SigmaChoice = 'beta' | 'tilde_beta';

export function reverseStep(
  x_t: number[],
  eps_pred: number[],
  t: number,
  z: number[],
  sched: Schedule,
  sigma_choice: SigmaChoice = 'beta',
): number[] {
  const mu = posteriorMeanFromEps(x_t, eps_pred, t, sched);
  if (t === 0) return mu; // no noise on the final step
  const sigma2 = sigma_choice === 'beta' ? sched.betas[t] : sched.tilde_betas[t];
  const sigma = Math.sqrt(sigma2);
  return mu.map((m, i) => m + sigma * z[i]);
}

/** Standard Gaussian sample of dim d via Box-Muller. */
export function gaussSample(d: number, rng: () => number = Math.random): number[] {
  const out: number[] = [];
  while (out.length < d) {
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(2 * Math.PI * u2));
    if (out.length < d) out.push(r * Math.sin(2 * Math.PI * u2));
  }
  return out;
}
