import { describe, it, expect } from 'vitest';
import { elboBimodal } from './bimodal';

describe('elboBimodal', () => {
  it('matches pre-verified value at mode-seeking optimum N(3, 1.05)', () => {
    expect(elboBimodal(3, 1.05)).toBeCloseTo(-0.6888, 3);
  });
  it('matches pre-verified value for mass-covering N(0, 10)', () => {
    expect(elboBimodal(0, 10)).toBeCloseTo(-0.7789, 3);
  });
  it('matches pre-verified value for centered narrow N(0, 1)', () => {
    expect(elboBimodal(0, 1)).toBeCloseTo(-2.6934, 3);
  });
  it('is symmetric: ELBO(+mu) ≈ ELBO(-mu) for this symmetric target', () => {
    expect(elboBimodal(3, 1.05)).toBeCloseTo(elboBimodal(-3, 1.05), 4);
  });
  it('optimum at ±3 beats centered narrow', () => {
    expect(elboBimodal(3, 1.05)).toBeGreaterThan(elboBimodal(0, 1));
  });
});
