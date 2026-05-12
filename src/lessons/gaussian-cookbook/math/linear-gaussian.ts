import { Matrix, inverse } from 'ml-matrix';

/**
 * Given prior z ~ N(mu_0, Sigma_0) and likelihood x | z ~ N(A z + b, Sigma_n),
 * compute the posterior z | x = N(mu_post, Sigma_post).
 *
 * Posterior precision = Sigma_0^{-1} + A^T Sigma_n^{-1} A
 * Posterior mean      = Sigma_post * (Sigma_0^{-1} mu_0 + A^T Sigma_n^{-1} (x - b))
 */
export function linearGaussianPosterior(
  mu_0: number[],
  Sigma_0: number[][],
  A: number[][],
  b: number[],
  Sigma_n: number[][],
  x: number[],
): { mu: number[]; Sigma: number[][] } {
  const S0_inv = inverse(new Matrix(Sigma_0));
  const Sn_inv = inverse(new Matrix(Sigma_n));
  const AM = new Matrix(A);
  const xb = Matrix.columnVector(x.map((xi, i) => xi - b[i]));
  const mu0_vec = Matrix.columnVector(mu_0);

  const precision = S0_inv.add(AM.transpose().mmul(Sn_inv).mmul(AM));
  const Sigma_post = inverse(precision);
  const rhs = S0_inv.mmul(mu0_vec).add(AM.transpose().mmul(Sn_inv).mmul(xb));
  const mu_post = Sigma_post.mmul(rhs).to1DArray();

  return { mu: mu_post, Sigma: Sigma_post.to2DArray() };
}
