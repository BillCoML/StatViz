import { describe, it, expect } from 'vitest';
import { cosineSchedule, paperSchedule, browserSchedule } from './schedule';

describe('linearSchedule', () => {
  const s = paperSchedule();

  it('T=1000, paper endpoints', () => {
    expect(s.T).toBe(1000);
    expect(s.betas[0]).toBeCloseTo(1e-4, 8);
    expect(s.betas[999]).toBeCloseTo(0.02, 8);
  });

  it('alpha_bars[500] ≈ 0.0778 (spec target, within 1e-4)', () => {
    expect(s.alpha_bars[500]).toBeCloseTo(0.0778, 3);
  });

  it('alpha_bars[999] ≈ 4.04e-5 (spec target)', () => {
    expect(s.alpha_bars[999]).toBeGreaterThan(3.8e-5);
    expect(s.alpha_bars[999]).toBeLessThan(4.3e-5);
  });

  it('tilde_betas <= betas everywhere, equality only at t=0', () => {
    for (let t = 1; t < s.T; t++) {
      expect(s.tilde_betas[t]).toBeLessThanOrEqual(s.betas[t] + 1e-12);
    }
    expect(s.tilde_betas[0]).toBe(0); // alpha_bars_prev[0] = 1 ⇒ (1-1)/(1-ab0) * b0 = 0
  });

  it('alphas, alpha_bars consistency', () => {
    let prod = 1;
    for (let t = 0; t < s.T; t++) {
      prod *= s.alphas[t];
      expect(s.alpha_bars[t]).toBeCloseTo(prod, 10);
    }
  });
});

describe('cosineSchedule', () => {
  it('alpha_bars monotonically decreasing in [eps, 1]', () => {
    const s = cosineSchedule(100);
    for (let t = 1; t < s.T; t++) {
      expect(s.alpha_bars[t]).toBeLessThan(s.alpha_bars[t - 1]);
    }
    expect(s.alpha_bars[0]).toBeGreaterThan(0.99);
    expect(s.alpha_bars[s.T - 1]).toBeLessThan(0.05);
  });
});

describe('browserSchedule', () => {
  it('T=100 with same linear endpoints', () => {
    const s = browserSchedule();
    expect(s.T).toBe(100);
    expect(s.betas[0]).toBeCloseTo(1e-4, 8);
    expect(s.betas[99]).toBeCloseTo(0.02, 8);
    // alpha_bar_99 for T=100 is much larger than T=1000 (less destruction)
    expect(s.alpha_bars[99]).toBeGreaterThan(0.3);
  });
});
