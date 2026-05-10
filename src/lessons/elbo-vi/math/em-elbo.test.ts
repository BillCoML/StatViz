import { describe, it, expect } from 'vitest';
import { elboTwoCoins } from './em-elbo';

const uniformGammas = [0.5, 0.5, 0.5, 0.5, 0.5];
const eStepGammas   = [0.4491, 0.8050, 0.7335, 0.3522, 0.6472];

describe('elboTwoCoins', () => {
  it('matches pre-verified value for uniform q at (0.6, 0.5)', () => {
    expect(elboTwoCoins(uniformGammas, 0.6, 0.5)).toBeCloseTo(-33.5458, 3);
  });
  it('matches pre-verified value for E-step q at (0.6, 0.5) — bound is tight', () => {
    expect(elboTwoCoins(eStepGammas, 0.6, 0.5)).toBeCloseTo(-33.0939, 3);
  });
  it('matches pre-verified value for E-step q after M-step to (0.7130, 0.5813)', () => {
    expect(elboTwoCoins(eStepGammas, 0.7130, 0.5813)).toBeCloseTo(-31.9972, 3);
  });
  it('E-step q gives higher ELBO than uniform q (E-step improves the bound)', () => {
    expect(elboTwoCoins(eStepGammas, 0.6, 0.5))
      .toBeGreaterThan(elboTwoCoins(uniformGammas, 0.6, 0.5));
  });
});
