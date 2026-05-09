/** KL between two discrete distributions (arrays of equal length, each summing to 1). */
export function klDiscrete(p: number[], q: number[]): number {
  if (p.length !== q.length) throw new Error('p and q must have same length');
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] === 0) continue;          // 0 log 0 = 0
    if (q[i] === 0) return Infinity;   // p > 0, q = 0
    s += p[i] * Math.log(p[i] / q[i]);
  }
  return s;
}

/** KL between two Bernoullis. */
export function klBernoulli(p: number, q: number): number {
  return klDiscrete([p, 1 - p], [q, 1 - q]);
}

/** Closed-form KL between two univariate Gaussians N(mu1, sigma1^2) || N(mu2, sigma2^2). */
export function klGaussian(mu1: number, sigma1: number,
                           mu2: number, sigma2: number): number {
  return Math.log(sigma2 / sigma1)
       + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
       - 0.5;
}

/** Numeric KL between two continuous densities sampled on a uniform grid.
 *  Uses a ratio-based guard: only points where p has meaningful mass but q is
 *  effectively zero relative to p contribute Infinity. A tiny floor on q
 *  prevents log(0) without spuriously declaring the integral infinite when q
 *  is merely small in a tail. */
export function klContinuous(p: number[], q: number[], dx: number): number {
  if (p.length !== q.length) throw new Error('p and q must have same length');
  const pMax = Math.max(...p);
  const pFloor = pMax * 1e-12;
  const qFloor = 1e-300;
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= pFloor) continue;
    if (q[i] <= 0) return Infinity;
    s += p[i] * Math.log(p[i] / Math.max(q[i], qFloor)) * dx;
  }
  return s;
}

/** Standard univariate Gaussian density. */
export function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}
