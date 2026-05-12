import { Matrix, determinant, inverse } from 'ml-matrix';

/** KL(N(mu1, Sigma1) || N(mu2, Sigma2)). */
export function klMVN(
  mu1: number[],
  Sigma1: number[][],
  mu2: number[],
  Sigma2: number[][],
): number {
  const d = mu1.length;
  const S1 = new Matrix(Sigma1);
  const S2 = new Matrix(Sigma2);
  const S2inv = inverse(S2);
  const det1 = determinant(S1);
  const det2 = determinant(S2);
  const diff = Matrix.columnVector(mu1.map((m, i) => mu2[i] - m));
  const maha = diff.transpose().mmul(S2inv).mmul(diff).get(0, 0);
  const tr = S2inv.mmul(S1).trace();
  return 0.5 * (Math.log(det2 / det1) - d + tr + maha);
}

/**
 * Sub-terms of klMVN broken out — used by KLMVNExplorer readouts.
 * Returns { logDet, negD, traceterm, mahalanobis, total }.
 */
export function klMVNTerms(
  mu1: number[],
  Sigma1: number[][],
  mu2: number[],
  Sigma2: number[][],
): { logDet: number; negD: number; traceterm: number; mahalanobis: number; total: number } {
  const d = mu1.length;
  const S1 = new Matrix(Sigma1);
  const S2 = new Matrix(Sigma2);
  const S2inv = inverse(S2);
  const logDet = 0.5 * Math.log(determinant(S2) / determinant(S1));
  const negD = -0.5 * d;
  const traceterm = 0.5 * S2inv.mmul(S1).trace();
  const diff = Matrix.columnVector(mu1.map((m, i) => mu2[i] - m));
  const mahalanobis = 0.5 * diff.transpose().mmul(S2inv).mmul(diff).get(0, 0);
  return { logDet, negD, traceterm, mahalanobis, total: logDet + negD + traceterm + mahalanobis };
}

/**
 * KL(N(mu, diag(exp(2*log_sigma))) || N(0, I)) — the VAE regularizer.
 * Takes log_sigma (not sigma^2) to match the standard VAE parameterization.
 */
export function klDiagFromStandard(mu: number[], log_sigma: number[]): number {
  let s = 0;
  for (let i = 0; i < mu.length; i++) {
    const sigma2 = Math.exp(2 * log_sigma[i]);
    s += sigma2 + mu[i] * mu[i] - 1 - 2 * log_sigma[i];
  }
  return 0.5 * s;
}
