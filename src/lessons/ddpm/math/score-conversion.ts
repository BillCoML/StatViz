/**
 * Conversion between the DDPM epsilon parameterization and the
 * score-matching score parameterization.
 *
 *   s_theta(x_t, t) = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)
 *
 * The conversion is the punchline of §7: the same network output
 * is both "predicted noise" (DDPM) and "rescaled score" (score matching).
 */

import type { Schedule } from './schedule';

export function epsToScore(eps_pred: number[], t: number, sched: Schedule): number[] {
  const denom = Math.sqrt(1 - sched.alpha_bars[t]);
  return eps_pred.map(e => -e / denom);
}

export function scoreToEps(score: number[], t: number, sched: Schedule): number[] {
  const factor = Math.sqrt(1 - sched.alpha_bars[t]);
  return score.map(s => -factor * s);
}

/**
 * Effective noise scale at timestep t in score-matching units:
 *   sigma_eff(t) = sqrt((1 - alpha_bar_t) / alpha_bar_t)
 *
 * Useful for the §7 dictionary: when the noisy DDPM sample is rescaled
 * to unit signal strength, the equivalent additive-noise std is sigma_eff.
 */
export function sigmaEff(t: number, sched: Schedule): number {
  const ab = sched.alpha_bars[t];
  return Math.sqrt((1 - ab) / ab);
}
