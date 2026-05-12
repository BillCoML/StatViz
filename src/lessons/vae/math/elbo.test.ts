import { describe, it, expect } from 'vitest';
import { vaeELBO, gaussianReconLogProb } from './elbo';
import { klDiagFromStandard } from '@lessons/gaussian-cookbook/math/kl-mvn';

describe('gaussianReconLogProb', () => {
  it('computes -1/(2σ²)||x - x_mean||² without the log-normalization constant', () => {
    // -0.5 * (0.25² + 0.09²) / 0.01 = -0.5 * 0.0706 / 0.01 = -3.53
    const r = gaussianReconLogProb([1.2, -0.8], [0.95, -0.71], 0.1);
    expect(r).toBeCloseTo(-3.53, 1);
  });

  it('returns 0 when x equals x_mean', () => {
    expect(gaussianReconLogProb([1, 2], [1, 2], 0.5)).toBeCloseTo(0, 10);
  });
});

describe('vaeELBO — §7 worked example', () => {
  const x           = [1.2, -0.8];
  const mu_phi      = [0.34, -0.12];
  const log_sigma   = [-0.2, -0.4];
  const mu_theta_z  = [0.95, -0.71];
  const sigma_x     = 0.1;

  it('recon ≈ −3.53 (within ±0.05)', () => {
    const { recon } = vaeELBO(x, mu_phi, log_sigma, mu_theta_z, sigma_x);
    expect(recon).toBeGreaterThan(-3.53 - 0.05);
    expect(recon).toBeLessThan(-3.53 + 0.05);
  });

  it('kl ≈ 0.224 (within ±0.01)', () => {
    const { kl } = vaeELBO(x, mu_phi, log_sigma, mu_theta_z, sigma_x);
    expect(kl).toBeGreaterThan(0.224 - 0.01);
    expect(kl).toBeLessThan(0.224 + 0.01);
  });

  it('elbo = recon - kl ≈ −3.75', () => {
    const { recon, kl, elbo } = vaeELBO(x, mu_phi, log_sigma, mu_theta_z, sigma_x);
    expect(elbo).toBeCloseTo(recon - kl, 10);
    expect(elbo).toBeCloseTo(-3.75, 1);
  });

  it('kl is non-negative', () => {
    const { kl } = vaeELBO(x, mu_phi, log_sigma, mu_theta_z, sigma_x);
    expect(kl).toBeGreaterThanOrEqual(0);
  });
});

describe('kl ↔ gaussian-cookbook consistency', () => {
  it('vaeELBO kl matches klDiagFromStandard directly', () => {
    const mu = [0.5, -0.2];
    const ls = [0.1, -0.3];
    const direct = klDiagFromStandard(mu, ls);
    const { kl } = vaeELBO([0, 0], mu, ls, [0, 0], 1);
    expect(kl).toBeCloseTo(direct, 10);
  });
});
