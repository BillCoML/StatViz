import { describe, it, expect } from 'vitest';
import { dsmTarget, dsmLoss, ismLossLinear } from './losses';

describe('dsmTarget', () => {
  it('(0.5,1.0) noisy (0.53,0.95) sigma=0.1 ≈ (-3, 5)', () => {
    const t = dsmTarget([0.5, 1.0], [0.53, 0.95], 0.1);
    expect(t[0]).toBeCloseTo(-3, 4);
    expect(t[1]).toBeCloseTo(5, 4);
  });

  it('at the clean point itself: target is (0, 0)', () => {
    const t = dsmTarget([1, 2], [1, 2], 0.5);
    expect(t[0]).toBeCloseTo(0, 10);
    expect(t[1]).toBeCloseTo(0, 10);
  });

  it('direction is always from noisy toward clean', () => {
    const t = dsmTarget([0, 0], [1, 0], 1.0);
    // noisy - clean = (1,0), so target = -(1,0)/1 = (-1, 0)
    expect(t[0]).toBeCloseTo(-1, 5);
    expect(t[1]).toBeCloseTo(0, 5);
  });
});

describe('dsmLoss', () => {
  it('s_theta=(0,0) x=(0.5,1.0) eps=(0.3,-0.5) sigma=0.1 → 34', () => {
    // target = -eps/sigma = (-3, 5); loss = 9 + 25 = 34
    const loss = dsmLoss([0, 0], [0.5, 1.0], [0.3, -0.5], 0.1);
    expect(loss).toBeCloseTo(34, 6);
  });

  it('zero when s_theta equals the target', () => {
    const loss = dsmLoss([-3, 5], [0.5, 1.0], [0.3, -0.5], 0.1);
    expect(loss).toBeCloseTo(0, 10);
  });

  it('is non-negative', () => {
    for (let i = 0; i < 20; i++) {
      const r = () => Math.random() * 4 - 2;
      const loss = dsmLoss([r(), r()], [r(), r()], [r(), r()], 0.5);
      expect(loss).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('ismLossLinear', () => {
  it('ISM loss for linear score on N(0,I_d): argmin a = -1', () => {
    const d = 2;
    // Minimize d*(a^2 + 2a) by grid search
    let bestA = 0;
    let bestLoss = Infinity;
    for (let a = -3; a <= 1; a += 0.0001) {
      const l = ismLossLinear(a, d);
      if (l < bestLoss) { bestLoss = l; bestA = a; }
    }
    expect(bestA).toBeCloseTo(-1, 3);
  });

  it('d*(a^2 + 2a) = d*(a+1)^2 - d, so minimum value is -d', () => {
    const d = 5;
    expect(ismLossLinear(-1, d)).toBeCloseTo(-d, 10);
  });
});
