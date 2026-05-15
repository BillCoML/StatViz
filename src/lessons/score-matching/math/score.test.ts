import { describe, it, expect } from 'vitest';
import { scoreGaussian, scoreGMM, scoreSmoothedGMM } from './score';

const eye2: number[][] = [[1, 0], [0, 1]];
const corr: number[][] = [[1, 0.5], [0.5, 1]];

describe('scoreGaussian', () => {
  it('N(0,I): score at x=(1,0) should be (-1, 0)', () => {
    const s = scoreGaussian([1, 0], [0, 0], eye2);
    expect(s[0]).toBeCloseTo(-1, 5);
    expect(s[1]).toBeCloseTo(0, 5);
  });

  it('N(0,Sigma) with corr: score at (1,0) mu=(0,0) ≈ (-4/3, 2/3)', () => {
    // Sigma^{-1} = (1/0.75)*[[1,-0.5],[-0.5,1]]
    // score = -Sigma^{-1}*(1,0) = -(1/0.75)*(1,-0.5) = (-4/3, 2/3)
    const s = scoreGaussian([1, 0], [0, 0], corr);
    expect(s[0]).toBeCloseTo(-4 / 3, 4);
    expect(s[1]).toBeCloseTo(2 / 3, 4);
  });

  it('score is zero at the mode (mu)', () => {
    const s = scoreGaussian([2, -1], [2, -1], corr);
    expect(s[0]).toBeCloseTo(0, 10);
    expect(s[1]).toBeCloseTo(0, 10);
  });
});

describe('scoreGMM', () => {
  const pis = [0.5, 0.5];
  const mus = [[2, 0], [-2, 0]];
  const Sigmas = [[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]];

  it('is near zero at the midpoint between modes (saddle point)', () => {
    const s = scoreGMM([0, 0], pis, mus, Sigmas);
    expect(Math.abs(s[0])).toBeLessThan(1e-6);
  });

  it('points toward right mode from (1, 0)', () => {
    const s = scoreGMM([1, 0], pis, mus, Sigmas);
    expect(s[0]).toBeGreaterThan(0);
  });

  it('points toward left mode from (-1, 0)', () => {
    const s = scoreGMM([-1, 0], pis, mus, Sigmas);
    expect(s[0]).toBeLessThan(0);
  });
});

describe('scoreSmoothedGMM', () => {
  const pis = [0.5, 0.5];
  const mus = [[2, 0], [-2, 0]];
  const Sigmas = [[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]];

  it('large sigma: smoothed score at origin ≈ 0 (symmetric)', () => {
    const s = scoreSmoothedGMM([0, 0], pis, mus, Sigmas, 5);
    expect(Math.abs(s[0])).toBeLessThan(1e-4);
  });

  it('small sigma: approaches raw GMM score near mode', () => {
    const raw     = scoreGMM([2.1, 0], pis, mus, Sigmas);
    const smooth  = scoreSmoothedGMM([2.1, 0], pis, mus, Sigmas, 0.01);
    expect(smooth[0]).toBeCloseTo(raw[0], 1);
  });
});
