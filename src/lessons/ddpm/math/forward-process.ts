/**
 * Forward (diffusion) process: q(x_t | x_0), q(x_{t-1} | x_t, x_0).
 *
 * Convention: t is zero-indexed. forwardSample with t=0 returns
 * x_0 + small noise (the first forward step); t = T-1 is the final
 * near-Gaussian step.
 */

import type { Schedule } from './schedule';

/** x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps. */
export function forwardSample(x_0: number[], t: number, eps: number[], sched: Schedule): number[] {
  const ab = sched.alpha_bars[t];
  const sqrtA = Math.sqrt(ab);
  const sqrt1mA = Math.sqrt(1 - ab);
  return x_0.map((x, i) => sqrtA * x + sqrt1mA * eps[i]);
}

/**
 * Forward posterior mean tilde_mu_t(x_t, x_0):
 *   = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x_0
 *   + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
 */
export function posteriorMean(x_t: number[], x_0: number[], t: number, sched: Schedule): number[] {
  const ab = sched.alpha_bars[t];
  const ab_prev = sched.alpha_bars_prev[t];
  const at = sched.alphas[t];
  const bt = sched.betas[t];
  const coef0 = Math.sqrt(ab_prev) * bt / (1 - ab);
  const coefT = Math.sqrt(at) * (1 - ab_prev) / (1 - ab);
  return x_0.map((x0i, i) => coef0 * x0i + coefT * x_t[i]);
}

/**
 * Forward posterior mean in the epsilon-parameterization:
 *   mu = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1 - alpha_bar_t)) * eps)
 *
 * For the true noise eps that produced x_t, this equals posteriorMean.
 * For an estimate eps_theta, this is the reverse-process mean mu_theta.
 */
export function posteriorMeanFromEps(x_t: number[], eps: number[], t: number, sched: Schedule): number[] {
  const at = sched.alphas[t];
  const ab = sched.alpha_bars[t];
  const coef = sched.betas[t] / Math.sqrt(1 - ab);
  const factor = 1 / Math.sqrt(at);
  return x_t.map((xi, i) => factor * (xi - coef * eps[i]));
}

/**
 * The model's running estimate of x_0:
 *   x_hat_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
 *
 * With the true noise, this exactly recovers x_0.
 */
export function xHat0(x_t: number[], eps: number[], t: number, sched: Schedule): number[] {
  const ab = sched.alpha_bars[t];
  const sqrtA = Math.sqrt(ab);
  const sqrt1mA = Math.sqrt(1 - ab);
  return x_t.map((xi, i) => (xi - sqrt1mA * eps[i]) / sqrtA);
}

/** Forward posterior variance tilde_beta_t (scalar; isotropic). */
export function posteriorVar(t: number, sched: Schedule): number {
  return sched.tilde_betas[t];
}
