import { describe, it, expect } from 'vitest';
import { runUntilConvergence, observedLogLikelihood, eStep } from './algorithm';

describe('runUntilConvergence(0.6, 0.5)', () => {
  const history = runUntilConvergence(0.6, 0.5);

  it('reaches thetaA ≈ 0.7967 at iteration 10', () => {
    expect(history[10].thetaA).toBeCloseTo(0.7967, 3);
  });

  it('reaches thetaB ≈ 0.5197 at iteration 10', () => {
    expect(history[10].thetaB).toBeCloseTo(0.5197, 3);
  });

  it('observedLogLikelihood ≈ -31.5702 at iteration 10', () => {
    expect(history[10].observedLogLikelihood).toBeCloseTo(-31.5702, 2);
  });

  it('log-likelihood is monotonically non-decreasing', () => {
    for (let i = 1; i < history.length; i++) {
      expect(history[i].observedLogLikelihood).toBeGreaterThanOrEqual(
        history[i - 1].observedLogLikelihood - 1e-9,
      );
    }
  });
});

describe('runUntilConvergence(0.5, 0.5) — saddle stationarity', () => {
  const history = runUntilConvergence(0.5, 0.5);

  it('thetaA and thetaB remain equal throughout', () => {
    for (const s of history) {
      expect(Math.abs(s.thetaA - s.thetaB)).toBeLessThan(1e-9);
    }
  });
});

describe('eStep at (0.6, 0.5)', () => {
  const R = eStep(0.6, 0.5);

  it('γ₁ᴬ ≈ 0.4491 (trial 1, x=5)', () => {
    expect(R[0].gammaA).toBeCloseTo(0.4491, 3);
  });

  it('γ₂ᴬ ≈ 0.8050 (trial 2, x=9)', () => {
    expect(R[1].gammaA).toBeCloseTo(0.8050, 3);
  });

  it('γ₃ᴬ ≈ 0.7335 (trial 3, x=8)', () => {
    expect(R[2].gammaA).toBeCloseTo(0.7335, 3);
  });

  it('γ₄ᴬ ≈ 0.3522 (trial 4, x=4)', () => {
    expect(R[3].gammaA).toBeCloseTo(0.3522, 3);
  });

  it('γ₅ᴬ ≈ 0.6472 (trial 5, x=7)', () => {
    expect(R[4].gammaA).toBeCloseTo(0.6472, 3);
  });
});

describe('observedLogLikelihood at initial (0.6, 0.5)', () => {
  it('≈ -33.094', () => {
    expect(observedLogLikelihood(0.6, 0.5)).toBeCloseTo(-33.094, 2);
  });
});

describe('monotonicity for multiple initializations', () => {
  const inits: [number, number][] = [
    [0.8, 0.2],
    [0.3, 0.7],
    [0.9, 0.1],
    [0.4, 0.6],
  ];

  for (const [a, b] of inits) {
    it(`non-decreasing from (${a}, ${b})`, () => {
      const h = runUntilConvergence(a, b);
      for (let i = 1; i < h.length; i++) {
        expect(h[i].observedLogLikelihood).toBeGreaterThanOrEqual(
          h[i - 1].observedLogLikelihood - 1e-9,
        );
      }
    });
  }
});
