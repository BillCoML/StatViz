import { describe, it, expect } from 'vitest';
import { langevinStep, annealedLangevin } from './langevin';

describe('langevinStep', () => {
  it('x=(1,1) score=(-1,-1) eta=0.1 eps=(0.2,-0.3) ≈ (0.9894, 0.7658)', () => {
    // x' = (1,1) + 0.1*(-1,-1) + sqrt(0.2)*(0.2,-0.3)
    //    = (1,1) + (-0.1,-0.1) + (0.08944, -0.13416)
    //    = (0.98944, 0.76584)
    const result = langevinStep([1, 1], [-1, -1], 0.1, [0.2, -0.3]);
    expect(result[0]).toBeCloseTo(0.9894, 3);
    expect(result[1]).toBeCloseTo(0.7658, 3);
  });

  it('zero score and zero noise: x stays the same', () => {
    const result = langevinStep([3, -2], [0, 0], 0.1, [0, 0]);
    expect(result[0]).toBeCloseTo(3, 10);
    expect(result[1]).toBeCloseTo(-2, 10);
  });

  it('large negative score pulls x toward origin for N(0,I) score', () => {
    const x = [5, 5];
    const score = [-5, -5]; // score of N(0,I) at (5,5)
    const result = langevinStep(x, score, 0.01, [0, 0]);
    expect(result[0]).toBeLessThan(x[0]);
    expect(result[1]).toBeLessThan(x[1]);
  });
});

describe('annealedLangevin', () => {
  it('converges particles toward origin for N(0,I)', () => {
    // Use the analytical score of N(0,I): s(x) = -x
    const scoreFn = (x: number[], _sigma: number) => x.map(xi => -xi);
    const sigmas = [1.0, 0.5, 0.2, 0.1];
    let rngIdx = 0;
    // Deterministic "noise" that's actually small to let drift dominate
    const rngNormal = () => {
      rngIdx++;
      return [0.01 * Math.sin(rngIdx * 1.7), 0.01 * Math.cos(rngIdx * 1.3)];
    };
    const final = annealedLangevin([4, 4], sigmas, 50, scoreFn, 0.01, rngNormal);
    // Should move substantially toward origin from (4, 4) — initial dist ≈ 5.66
    const finalDist  = Math.hypot(final[0], final[1]);
    expect(finalDist).toBeLessThan(2.0);
  });
});
