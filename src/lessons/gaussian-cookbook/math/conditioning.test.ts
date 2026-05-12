import { describe, it, expect } from 'vitest';
import { conditionalGaussian, conditionalGaussian2D } from './conditioning';

describe('conditionalGaussian', () => {
  it('Y | X=1 ~ N(0.7, 0.51) for bivariate N(0, [[1, 0.7],[0.7, 1]])', () => {
    // (X,Y) ~ N(0, [[1, 0.7],[0.7, 1]]), observe X = 1
    const result = conditionalGaussian(
      [0],    // mu_X
      [0],    // mu_Y
      [[1]],  // Sigma_XX
      [[0.7]],// Sigma_XY
      [[0.7]],// Sigma_YX
      [[1]],  // Sigma_YY
      [1],    // x
    );
    expect(result.mu[0]).toBeCloseTo(0.7, 6);
    expect(result.Sigma[0][0]).toBeCloseTo(0.51, 6);
  });

  it('uncorrelated X,Y: conditioning does not shift mean', () => {
    const result = conditionalGaussian(
      [0], [5], [[2]], [[0]], [[0]], [[3]], [1],
    );
    expect(result.mu[0]).toBeCloseTo(5, 10);
    expect(result.Sigma[0][0]).toBeCloseTo(3, 10);
  });
});

describe('conditionalGaussian2D', () => {
  it('matches the Y | X=1 ~ N(0.7, 0.51) example', () => {
    const { mu, sigma2 } = conditionalGaussian2D(0, 0, 1, 0.7, 1, 1);
    expect(mu).toBeCloseTo(0.7, 6);
    expect(sigma2).toBeCloseTo(0.51, 6);
  });
});
