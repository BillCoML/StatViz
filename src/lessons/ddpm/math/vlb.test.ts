import { describe, it, expect } from 'vitest';
import { paperSchedule } from './schedule';
import { vlbWeight, L_T_perDim, NATS_TO_BITS, isoSharedKL } from './vlb';

describe('vlbWeight', () => {
  const s = paperSchedule();

  it('weight at small t is much larger than at large t', () => {
    const w1 = vlbWeight(1, s, 'beta');
    const w500 = vlbWeight(500, s, 'beta');
    expect(w1 / w500).toBeGreaterThan(20); // spec §6 says ~45x
  });
});

describe('L_T per dim', () => {
  const s = paperSchedule();

  it('paper schedule: L_T ≈ 2-4e-5 bits/dim for unit-norm x_0', () => {
    const nats = L_T_perDim(s, 1);
    const bits = nats * NATS_TO_BITS;
    expect(bits).toBeGreaterThan(1e-5);
    expect(bits).toBeLessThan(5e-5);
  });
});

describe('isoSharedKL', () => {
  it('zero when means coincide', () => {
    expect(isoSharedKL([1, 2], [1, 2], 0.5)).toBe(0);
  });
  it('squared distance over 2*sigma^2', () => {
    expect(isoSharedKL([1, 0], [0, 0], 0.5)).toBeCloseTo(1 / (2 * 0.5), 10);
  });
});
