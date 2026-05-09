import { describe, it, expect } from 'vitest';
import { klBernoulli, klGaussian, klDiscrete } from './kl';
import { jensenGap } from './jensen';

describe('klBernoulli', () => {
  it('matches the worked example for p=0.7, q=0.5', () => {
    expect(klBernoulli(0.7, 0.5)).toBeCloseTo(0.0823, 4);
  });
  it('matches the worked example for p=0.5, q=0.7 (asymmetric)', () => {
    expect(klBernoulli(0.5, 0.7)).toBeCloseTo(0.0872, 4);
  });
  it('is zero when p === q', () => {
    expect(klBernoulli(0.5, 0.5)).toBe(0);
  });
  it('is asymmetric: D(p||q) != D(q||p) in general', () => {
    expect(klBernoulli(0.7, 0.5)).not.toBeCloseTo(klBernoulli(0.5, 0.7), 6);
  });
});

describe('klGaussian', () => {
  it('returns 0.5 for N(0,1) || N(1,1)', () => {
    expect(klGaussian(0, 1, 1, 1)).toBeCloseTo(0.5, 10);
  });
  it('returns 0.318 for N(0,1) || N(0,4) [variance notation; σ=1, σ=2]', () => {
    expect(klGaussian(0, 1, 0, 2)).toBeCloseTo(0.3181, 4);
  });
  it('returns 0.807 for N(0,4) || N(0,1) [variance notation; σ=2, σ=1]', () => {
    expect(klGaussian(0, 2, 0, 1)).toBeCloseTo(0.8069, 4);
  });
  it('is zero when distributions match', () => {
    expect(klGaussian(1.5, 2, 1.5, 2)).toBeCloseTo(0, 12);
  });
});

describe('klDiscrete', () => {
  it('is zero for identical distributions', () => {
    expect(klDiscrete([0.5, 0.5], [0.5, 0.5])).toBe(0);
  });
  it('is +Infinity when q assigns zero where p does not', () => {
    expect(klDiscrete([0.5, 0.5], [1, 0])).toBe(Infinity);
  });
  it('treats 0*log(0/q) as 0', () => {
    expect(klDiscrete([1, 0], [0.5, 0.5])).toBeCloseTo(Math.log(2), 12);
  });
  it('throws on length mismatch', () => {
    expect(() => klDiscrete([0.5, 0.5], [1])).toThrow();
  });
  it('is non-negative for 100 random pairs (Gibbs fuzz test)', () => {
    const rng = (seed: number) => {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    };
    const r = rng(42);
    for (let trial = 0; trial < 100; trial++) {
      const k = 2 + Math.floor(r() * 5);
      const rawP = Array.from({ length: k }, () => r() + 1e-3);
      const rawQ = Array.from({ length: k }, () => r() + 1e-3);
      const sP = rawP.reduce((a, b) => a + b, 0);
      const sQ = rawQ.reduce((a, b) => a + b, 0);
      const p = rawP.map(x => x / sP);
      const q = rawQ.map(x => x / sQ);
      expect(klDiscrete(p, q)).toBeGreaterThanOrEqual(-1e-12);
    }
  });
});

describe('jensenGap', () => {
  it('returns 0 for a degenerate distribution (point mass)', () => {
    const result = jensenGap([3], [1], x => x * x);
    expect(result.gap).toBe(0);
  });
  it('returns the variance for phi(x) = x^2 (Var(X) = E[X^2] - E[X]^2)', () => {
    const values = [-1, 0, 1];
    const probs = [1 / 3, 1 / 3, 1 / 3];
    const result = jensenGap(values, probs, x => x * x);
    expect(result.gap).toBeCloseTo(2 / 3, 12);
  });
  it('is non-negative for convex phi (x^2) and any distribution', () => {
    const values = [-2, -1, 0, 1, 2];
    const probs = [0.1, 0.2, 0.4, 0.2, 0.1];
    const result = jensenGap(values, probs, x => x * x);
    expect(result.gap).toBeGreaterThanOrEqual(0);
  });
});
