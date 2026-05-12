import { Matrix, determinant, inverse, CholeskyDecomposition } from 'ml-matrix';

/** Log density of x under N(mu, Sigma). */
export function logMVNDensity(x: number[], mu: number[], Sigma: number[][]): number {
  const d = mu.length;
  const SigmaM = new Matrix(Sigma);
  const det = determinant(SigmaM);
  const inv = inverse(SigmaM);
  const diff = x.map((xi, i) => xi - mu[i]);
  const diffM = Matrix.columnVector(diff);
  const quad = diffM.transpose().mmul(inv).mmul(diffM).get(0, 0);
  return -0.5 * d * Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * quad;
}

/** Sample one realization from N(mu, Sigma) via Cholesky reparameterization. */
export function sampleMVN(
  mu: number[],
  Sigma: number[][],
  rng: () => number = Math.random,
): number[] {
  const d = mu.length;
  const SigmaM = new Matrix(Sigma);
  const chol = new CholeskyDecomposition(SigmaM);
  const L = chol.lowerTriangularMatrix;
  const eps: number[] = Array.from({ length: d }, () => boxMuller(rng));
  const epsM = Matrix.columnVector(eps);
  const Leps = L.mmul(epsM).to1DArray();
  return mu.map((m, i) => m + Leps[i]);
}

function boxMuller(rng: () => number): number {
  let u = rng();
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/** Eigenvalues and eigenvectors of a 2×2 symmetric matrix (for visualization). */
export function eigen2x2(Sigma: number[][]): {
  values: [number, number];
  vectors: [[number, number], [number, number]];
} {
  const a = Sigma[0][0], b = Sigma[0][1], d = Sigma[1][1];
  const trace = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, (trace / 2) ** 2 - det));
  const l1 = trace / 2 + disc;
  const l2 = trace / 2 - disc;
  let v1: [number, number], v2: [number, number];
  if (Math.abs(b) > 1e-12) {
    const n1 = Math.hypot(l1 - a, b);
    const n2 = Math.hypot(l2 - a, b);
    v1 = [(l1 - a) / n1, b / n1];
    v2 = [(l2 - a) / n2, b / n2];
  } else {
    v1 = [1, 0];
    v2 = [0, 1];
  }
  return { values: [l1, l2], vectors: [v1, v2] };
}
