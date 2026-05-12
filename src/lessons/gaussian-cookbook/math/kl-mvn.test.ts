import { describe, it, expect } from 'vitest';
import { klMVN, klDiagFromStandard } from './kl-mvn';

const diag = (vals: number[]): number[][] =>
  vals.map((v, i) => vals.map((_, j) => (i === j ? v : 0)));

const eye = (n: number): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

describe('klMVN', () => {
  it('3D diagonal vs standard normal ≈ 1.1534', () => {
    const result = klMVN([1, 0, -1], diag([1, 2, 1]), [0, 0, 0], eye(3));
    expect(result).toBeCloseTo(1.153426, 4);
  });

  it('2D dense vs 2I ≈ 0.8370', () => {
    const result = klMVN([1, 1], [[1, 0.5], [0.5, 1]], [0, 0], [[2, 0], [0, 2]]);
    expect(result).toBeCloseTo(0.836988, 4);
  });

  it('reverse: 2I vs 2D dense ≈ 1.4963', () => {
    const result = klMVN([0, 0], [[2, 0], [0, 2]], [1, 1], [[1, 0.5], [0.5, 1]]);
    expect(result).toBeCloseTo(1.4963, 3);
  });

  it('equal distributions → 0', () => {
    const result = klMVN([1, 2], [[2, 1], [1, 2]], [1, 2], [[2, 1], [1, 2]]);
    expect(result).toBeCloseTo(0, 10);
  });

  it('is non-negative (random fuzz)', () => {
    for (let t = 0; t < 20; t++) {
      const mu1 = [Math.random() * 4 - 2, Math.random() * 4 - 2];
      const mu2 = [Math.random() * 4 - 2, Math.random() * 4 - 2];
      const a = Math.random() + 0.5;
      const b = Math.random() + 0.5;
      expect(klMVN(mu1, diag([a, b]), mu2, diag([a + 0.1, b + 0.1]))).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('klDiagFromStandard', () => {
  it('≈ 0.2301 for ([0.5, -0.2], [0.1, -0.3])', () => {
    const result = klDiagFromStandard([0.5, -0.2], [0.1, -0.3]);
    expect(result).toBeCloseTo(0.230107, 4);
  });

  it('zero when mu=0, log_sigma=0 (i.e., sigma=1)', () => {
    expect(klDiagFromStandard([0, 0], [0, 0])).toBeCloseTo(0, 10);
  });

  it('is non-negative', () => {
    for (let t = 0; t < 20; t++) {
      const mu = [Math.random() * 4 - 2, Math.random() * 4 - 2];
      const ls = [Math.random() * 2 - 1, Math.random() * 2 - 1];
      expect(klDiagFromStandard(mu, ls)).toBeGreaterThanOrEqual(0);
    }
  });
});
