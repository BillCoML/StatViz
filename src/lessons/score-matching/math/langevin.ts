/**
 * One step of Langevin dynamics:
 *   x' = x + eta * score(x) + sqrt(2 * eta) * eps
 */
export function langevinStep(
  x: number[],
  score: number[],
  eta: number,
  eps: number[],
): number[] {
  const sqrt2eta = Math.sqrt(2 * eta);
  return x.map((xi, i) => xi + eta * score[i] + sqrt2eta * eps[i]);
}

/**
 * Full annealed Langevin: walk down a geometric noise schedule.
 * At each sigma level, run T inner steps with step size alpha = epsilonBase * (sigma/sigma_L)^2.
 * The score is computed at sigma/2 from annealedLangevin formula:
 *   x = x + (alpha/2) * score(x, sigma) + sqrt(alpha) * eps
 */
export function annealedLangevin(
  x_init: number[],
  sigmas: number[],
  T: number,
  scoreFn: (x: number[], sigma: number) => number[],
  epsilonBase: number,
  rngNormal: () => number[],
): number[] {
  let x = x_init.slice();
  const sigma_L = sigmas[sigmas.length - 1];
  for (const sigma of sigmas) {
    const alpha = epsilonBase * (sigma / sigma_L) ** 2;
    for (let t = 0; t < T; t++) {
      const s = scoreFn(x, sigma);
      const eps = rngNormal();
      // NCSN step: x = x + (alpha/2)*score + sqrt(alpha)*eps
      x = x.map((xi, i) => xi + (alpha / 2) * s[i] + Math.sqrt(alpha) * eps[i]);
    }
  }
  return x;
}
