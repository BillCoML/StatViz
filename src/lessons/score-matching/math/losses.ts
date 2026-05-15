/**
 * Denoising score matching target at a noisy point.
 * The conditional score of q(x_noisy | x_clean) = N(x_clean, sigma^2 I) is:
 *   -(x_noisy - x_clean) / sigma^2 = -eps / sigma
 */
export function dsmTarget(x_clean: number[], x_noisy: number[], sigma: number): number[] {
  return x_clean.map((c, i) => -(x_noisy[i] - c) / (sigma * sigma));
}

/**
 * DSM loss for one (x_clean, eps, sigma, s_theta) tuple.
 * s_theta should approximate -eps/sigma; loss = ||s_theta - (-eps/sigma)||^2.
 */
export function dsmLoss(
  s_theta: number[],
  x_clean: number[],
  eps: number[],
  sigma: number,
): number {
  let total = 0;
  for (let i = 0; i < x_clean.length; i++) {
    const target = -eps[i] / sigma;
    total += (s_theta[i] - target) ** 2;
  }
  return total;
}

/**
 * ISM loss for a linear score model s_theta(x) = a*x on N(0, I_d).
 * Useful for the sanity-check test: minimize d*(a^2 + 2a) → a = -1.
 */
export function ismLossLinear(a: number, d: number): number {
  // E[||ax||^2] + 2*tr(nabla s_theta) = a^2 * d + 2 * a * d
  return d * (a * a + 2 * a);
}
