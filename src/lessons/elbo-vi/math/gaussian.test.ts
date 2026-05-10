import { describe, it, expect } from 'vitest';
import { posterior, logEvidence, elboGaussian } from './gaussian';

const model = { tau2: 1, sigma2: 1, data: [2.5, 1.7, 3.1] };

describe('posterior', () => {
  it('returns correct mean for conjugate Gaussian', () => {
    const { mu } = posterior(model);
    expect(mu).toBeCloseTo(1.825, 10);
  });
  it('returns correct variance for conjugate Gaussian', () => {
    const { var_ } = posterior(model);
    expect(var_).toBeCloseTo(0.25, 10);
  });
});

describe('logEvidence', () => {
  it('matches the pre-verified value ≈ −6.1637', () => {
    expect(logEvidence(model)).toBeCloseTo(-6.1637, 2);
  });
});

describe('elboGaussian', () => {
  it('matches log evidence at the posterior optimum (KL gap = 0)', () => {
    expect(elboGaussian(model, 1.825, 0.25)).toBeCloseTo(-6.1637, 2);
  });
  it('matches pre-verified value for q = prior N(0,1)', () => {
    expect(elboGaussian(model, 0, 1)).toBeCloseTo(-13.6318, 2);
  });
  it('matches pre-verified value for q ≈ MLE N(2.4333, 0.3333)', () => {
    expect(elboGaussian(model, 2.4333, 0.3333)).toBeCloseTo(-6.9267, 2);
  });
  it('is always ≤ log evidence (bound holds)', () => {
    const logP = logEvidence(model);
    for (const [mu, v] of [[0, 1], [1, 0.5], [2, 2], [1.825, 0.25], [-1, 3]] as [number, number][]) {
      expect(elboGaussian(model, mu, v)).toBeLessThanOrEqual(logP + 1e-4);
    }
  });
});
