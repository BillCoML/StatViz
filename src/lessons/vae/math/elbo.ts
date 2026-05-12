import { klDiagFromStandard } from '@lessons/gaussian-cookbook/math/kl-mvn';

/** Reconstruction term: -1/(2σ²) ||x - x_mean||², the quadratic part of log p(x|z). */
export function gaussianReconLogProb(x: number[], x_mean: number[], sigma_x: number): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += -0.5 * (x[i] - x_mean[i]) ** 2 / (sigma_x * sigma_x);
  }
  return s;
}

/**
 * VAE ELBO for one example.
 * recon = -1/(2σ_x²)||x - μθ(z)||²
 * kl    = KL(N(μφ, diag(exp(2logσφ))) || N(0,I))
 * elbo  = recon - kl
 */
export function vaeELBO(
  x: number[],
  mu_phi: number[],
  log_sigma_phi: number[],
  mu_theta_z: number[],
  sigma_x: number,
): { recon: number; kl: number; elbo: number } {
  const recon = gaussianReconLogProb(x, mu_theta_z, sigma_x);
  const kl = klDiagFromStandard(mu_phi, log_sigma_phi);
  return { recon, kl, elbo: recon - kl };
}
