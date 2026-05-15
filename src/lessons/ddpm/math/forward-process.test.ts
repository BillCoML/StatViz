import { describe, it, expect } from 'vitest';
import { paperSchedule } from './schedule';
import {
  forwardSample, posteriorMean, posteriorMeanFromEps, xHat0
} from './forward-process';

describe('forwardSample', () => {
  const s = paperSchedule();

  it('spec target: x_0=(1,-0.5), eps=(0.3,-0.7), t=500 → ≈ (0.567, -0.812)', () => {
    const x_t = forwardSample([1.0, -0.5], 500, [0.3, -0.7], s);
    expect(x_t[0]).toBeCloseTo(0.567, 2);
    expect(x_t[1]).toBeCloseTo(-0.812, 2);
  });

  it('at t=0 with eps=0, x_t ≈ x_0 (tiny noise scale)', () => {
    const x_t = forwardSample([1, 1], 0, [0, 0], s);
    expect(x_t[0]).toBeCloseTo(1 * Math.sqrt(s.alpha_bars[0]), 8);
  });
});

describe('posteriorMean and posteriorMeanFromEps', () => {
  const s = paperSchedule();

  it('agree (within 1e-10) when given the true noise', () => {
    const x_0 = [1.0, -0.5];
    const eps = [0.3, -0.7];
    const t = 500;
    const x_t = forwardSample(x_0, t, eps, s);
    const mu_a = posteriorMean(x_t, x_0, t, s);
    const mu_b = posteriorMeanFromEps(x_t, eps, t, s);
    expect(mu_a[0]).toBeCloseTo(mu_b[0], 10);
    expect(mu_a[1]).toBeCloseTo(mu_b[1], 10);
  });

  it('agree across multiple random samples', () => {
    const x_0 = [0.2, 1.7];
    for (let t of [10, 100, 500, 900]) {
      const eps = [Math.random(), Math.random()];
      const x_t = forwardSample(x_0, t, eps, s);
      const a = posteriorMean(x_t, x_0, t, s);
      const b = posteriorMeanFromEps(x_t, eps, t, s);
      expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeLessThan(1e-10);
    }
  });
});

describe('xHat0', () => {
  const s = paperSchedule();

  it('recovers x_0 exactly under perfect epsilon prediction', () => {
    const x_0 = [1.0, -0.5];
    const eps = [0.3, -0.7];
    const t = 500;
    const x_t = forwardSample(x_0, t, eps, s);
    const recovered = xHat0(x_t, eps, t, s);
    expect(recovered[0]).toBeCloseTo(x_0[0], 10);
    expect(recovered[1]).toBeCloseTo(x_0[1], 10);
  });
});
