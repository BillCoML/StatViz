import { Matrix, inverse, determinant } from 'ml-matrix';

function logMVN(x: number[], mu: number[], Sigma: number[][]): number {
  const d = mu.length;
  const SigmaM = new Matrix(Sigma);
  const det = determinant(SigmaM);
  const inv = inverse(SigmaM);
  const diff = x.map((xi, i) => xi - mu[i]);
  const diffM = Matrix.columnVector(diff);
  const quad = diffM.transpose().mmul(inv).mmul(diffM).get(0, 0);
  return -0.5 * d * Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * quad;
}

/** Score of a multivariate Gaussian: -Sigma^{-1}(x - mu). */
export function scoreGaussian(x: number[], mu: number[], Sigma: number[][]): number[] {
  const inv = inverse(new Matrix(Sigma));
  const diff = Matrix.columnVector(x.map((xi, i) => xi - mu[i]));
  return inv.mmul(diff).to1DArray().map(v => -v);
}

/** Score of a Gaussian mixture: responsibility-weighted sum of component scores. */
export function scoreGMM(
  x: number[],
  pis: number[],
  mus: number[][],
  Sigmas: number[][][],
): number[] {
  const logProbs = mus.map((mu, k) => Math.log(pis[k]) + logMVN(x, mu, Sigmas[k]));
  const m = Math.max(...logProbs);
  const logNorm = m + Math.log(logProbs.reduce((s, lp) => s + Math.exp(lp - m), 0));
  const rs = logProbs.map(lp => Math.exp(lp - logNorm));

  const d = x.length;
  const s = new Array<number>(d).fill(0);
  for (let k = 0; k < mus.length; k++) {
    const sk = scoreGaussian(x, mus[k], Sigmas[k]);
    for (let i = 0; i < d; i++) s[i] += rs[k] * sk[i];
  }
  return s;
}

/**
 * Score of the noise-smoothed distribution p_sigma of a GMM.
 * Convolution of a GMM with N(0, sigma^2 I) yields a GMM with Sigma_k + sigma^2 I.
 */
export function scoreSmoothedGMM(
  x: number[],
  pis: number[],
  mus: number[][],
  Sigmas: number[][][],
  sigma: number,
): number[] {
  const smoothedSigmas = Sigmas.map(S =>
    S.map((row, i) => row.map((v, j) => v + (i === j ? sigma * sigma : 0)))
  );
  return scoreGMM(x, pis, mus, smoothedSigmas);
}

/** Score of the banana / Rosenbrock distribution (unnormalized). */
export function scoreBanana(x: number[], a = 1, b = 10): number[] {
  // p(x) ∝ exp(-[(1-x0)^2/a^2 + b*(x1 - x0^2)^2])
  // log p = -(1-x0)^2/a^2 - b*(x1 - x0^2)^2 + const
  const dx0 = 2 * (1 - x[0]) / (a * a) + 4 * b * x[0] * (x[1] - x[0] * x[0]);
  const dx1 = -2 * b * (x[1] - x[0] * x[0]);
  return [dx0, dx1];
}

/** Score of a ring distribution: p(r) ∝ exp(-0.5*((r-R)/w)^2). */
export function scoreRing(x: number[], R = 2, w = 0.3): number[] {
  const r = Math.hypot(x[0], x[1]);
  if (r < 1e-9) return [0, 0];
  const drLogP = -(r - R) / (w * w);
  return [drLogP * x[0] / r, drLogP * x[1] / r];
}
