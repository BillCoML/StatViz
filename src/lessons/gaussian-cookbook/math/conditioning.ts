import { Matrix, inverse } from 'ml-matrix';

/**
 * Given jointly Gaussian (X, Y) with block mean/covariance, compute Y | X = x.
 *
 * Returns the conditional distribution N(mu_cond, Sigma_cond) where:
 *   mu_cond    = mu_Y + Sigma_YX * Sigma_XX^{-1} * (x - mu_X)
 *   Sigma_cond = Sigma_YY - Sigma_YX * Sigma_XX^{-1} * Sigma_XY   (Schur complement of Sigma_XX)
 */
export function conditionalGaussian(
  mu_X: number[],
  mu_Y: number[],
  Sigma_XX: number[][],
  Sigma_XY: number[][],
  Sigma_YX: number[][],
  Sigma_YY: number[][],
  x: number[],
): { mu: number[]; Sigma: number[][] } {
  const SXX_inv = inverse(new Matrix(Sigma_XX));
  const SYX = new Matrix(Sigma_YX);
  const SXY = new Matrix(Sigma_XY);
  const SYY = new Matrix(Sigma_YY);
  const xDev = Matrix.columnVector(x.map((xi, i) => xi - mu_X[i]));

  const shift = SYX.mmul(SXX_inv).mmul(xDev).to1DArray();
  const mu_cond = mu_Y.map((m, i) => m + shift[i]);

  const Sigma_cond = SYY.sub(SYX.mmul(SXX_inv).mmul(SXY)).to2DArray();
  return { mu: mu_cond, Sigma: Sigma_cond };
}

/**
 * Convenience wrapper for the 2D bivariate case:
 * (X, Y) ~ N([mu_x, mu_y], [[s_xx, s_xy],[s_xy, s_yy]]).
 * Returns scalar (mu_cond, sigma2_cond).
 */
export function conditionalGaussian2D(
  mu_x: number,
  mu_y: number,
  s_xx: number,
  s_xy: number,
  s_yy: number,
  x: number,
): { mu: number; sigma2: number } {
  const mu_cond = mu_y + (s_xy / s_xx) * (x - mu_x);
  const sigma2_cond = s_yy - (s_xy * s_xy) / s_xx;
  return { mu: mu_cond, sigma2: sigma2_cond };
}
