import { describe, it, expect } from 'vitest';
import { linearGaussianPosterior } from './linear-gaussian';

describe('linearGaussianPosterior', () => {
  it('matches the §6 worked example', () => {
    // Prior: z ~ N(0, I_2)
    // Likelihood: x | z ~ N(A z, 0.1 I_2), A = [[1, 0.5],[0.3, 1]]
    // Observe: x = [1.5, 0.8]
    const mu_0 = [0, 0];
    const Sigma_0 = [[1, 0], [0, 1]];
    const A = [[1, 0.5], [0.3, 1]];
    const b = [0, 0];
    const Sigma_n = [[0.1, 0], [0, 0.1]];
    const x = [1.5, 0.8];

    const { mu, Sigma } = linearGaussianPosterior(mu_0, Sigma_0, A, b, Sigma_n, x);

    expect(mu[0]).toBeCloseTo(1.147, 2);
    expect(mu[1]).toBeCloseTo(0.468, 2);
    expect(Sigma[0][0]).toBeCloseTo(0.1397, 3);
    expect(Sigma[1][1]).toBeCloseTo(0.1231, 3);
  });

  it('with no data (A=0 effectively), posterior = prior', () => {
    // If A = 0 matrix and huge noise, posterior ≈ prior
    const mu_0 = [2, -1];
    const Sigma_0 = [[1, 0], [0, 1]];
    const A = [[0, 0], [0, 0]];
    const b = [0, 0];
    const Sigma_n = [[1e6, 0], [0, 1e6]];
    const x = [100, -100];

    const { mu, Sigma } = linearGaussianPosterior(mu_0, Sigma_0, A, b, Sigma_n, x);
    expect(mu[0]).toBeCloseTo(mu_0[0], 3);
    expect(mu[1]).toBeCloseTo(mu_0[1], 3);
    expect(Sigma[0][0]).toBeCloseTo(1, 3);
  });
});
