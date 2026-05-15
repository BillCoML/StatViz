import { describe, it, expect } from 'vitest';
import { paperSchedule } from './schedule';
import { forwardSample, posteriorMean } from './forward-process';
import { reverseStep, gaussSample } from './reverse-process';

describe('reverseStep', () => {
  const s = paperSchedule();

  it('with perfect eps prediction and z=0, equals posterior mean', () => {
    const x_0 = [1.0, -0.5];
    const eps = [0.3, -0.7];
    const t = 500;
    const x_t = forwardSample(x_0, t, eps, s);
    const r = reverseStep(x_t, eps, t, [0, 0], s, 'beta');
    const pm = posteriorMean(x_t, x_0, t, s);
    expect(r[0]).toBeCloseTo(pm[0], 9);
    expect(r[1]).toBeCloseTo(pm[1], 9);
  });

  it('at t=0 returns mu regardless of z (no noise on final step)', () => {
    const x_t = [0.1, 0.1];
    const eps = [0.05, 0.05];
    const r1 = reverseStep(x_t, eps, 0, [10, 10], s);
    const r2 = reverseStep(x_t, eps, 0, [0, 0], s);
    expect(r1[0]).toBe(r2[0]);
    expect(r1[1]).toBe(r2[1]);
  });

  it('adds noise scaled by sigma_t when t > 0', () => {
    const x_t = [0.5, 0.5];
    const eps = [0.1, 0.1];
    const z = [1, -1];
    const t = 500;
    const r_with = reverseStep(x_t, eps, t, z, s, 'beta');
    const r_no   = reverseStep(x_t, eps, t, [0, 0], s, 'beta');
    const sigma = Math.sqrt(s.betas[t]);
    expect(r_with[0] - r_no[0]).toBeCloseTo(sigma * 1, 9);
    expect(r_with[1] - r_no[1]).toBeCloseTo(sigma * -1, 9);
  });
});

describe('gaussSample', () => {
  it('produces approximately unit variance for large samples', () => {
    const N = 4000;
    const xs = gaussSample(N);
    const mean = xs.reduce((a, b) => a + b, 0) / N;
    const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / N;
    expect(Math.abs(mean)).toBeLessThan(0.06);
    expect(variance).toBeGreaterThan(0.9);
    expect(variance).toBeLessThan(1.1);
  });
});
