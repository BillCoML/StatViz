import { describe, it, expect } from 'vitest';
import { paperSchedule } from './schedule';
import { epsToScore, scoreToEps, sigmaEff } from './score-conversion';

describe('score conversion', () => {
  const s = paperSchedule();

  it('roundtrip is identity', () => {
    const eps = [0.3, -0.7];
    const t = 500;
    const score = epsToScore(eps, t, s);
    const back = scoreToEps(score, t, s);
    expect(back[0]).toBeCloseTo(eps[0], 12);
    expect(back[1]).toBeCloseTo(eps[1], 12);
  });

  it('sigma_eff monotonically increasing in t', () => {
    let prev = sigmaEff(0, s);
    for (let t = 1; t < s.T; t += 50) {
      const cur = sigmaEff(t, s);
      expect(cur).toBeGreaterThan(prev);
      prev = cur;
    }
  });

  it('score points away from the Gaussian center at large t', () => {
    // At t close to T, q(x_t | x_0) ≈ N(0, I) so the score of N(0,I) is -x.
    // eps that produced a typical sample x_t ≈ x_t / sqrt(1 - ab_T).
    // score = -eps / sqrt(1 - ab) ≈ -x_t / (1 - ab) ≈ -x_t (when ab ≈ 0).
    const t = 999;
    const x_t = [1, -1];
    const eps_pred_typical = [
      x_t[0] / Math.sqrt(1 - s.alpha_bars[t]),
      x_t[1] / Math.sqrt(1 - s.alpha_bars[t]),
    ];
    const sc = epsToScore(eps_pred_typical, t, s);
    // sc ≈ -x_t / (1 - ab_t), almost = -x_t since 1 - ab_t ≈ 1.
    expect(sc[0]).toBeCloseTo(-x_t[0] / (1 - s.alpha_bars[t]), 8);
    expect(sc[1]).toBeCloseTo(-x_t[1] / (1 - s.alpha_bars[t]), 8);
  });
});
