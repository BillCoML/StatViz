import { klGaussian } from '@lessons/kl-jensen/math/kl';

export interface GaussianConjugate {
  tau2: number;
  sigma2: number;
  data: number[];
}

export function posterior(model: GaussianConjugate): { mu: number; var_: number } {
  const n = model.data.length;
  const xbar = model.data.reduce((a, b) => a + b, 0) / n;
  const var_ = 1 / (1 / model.tau2 + n / model.sigma2);
  const mu = var_ * (n * xbar / model.sigma2);
  return { mu, var_ };
}

export function logEvidence(model: GaussianConjugate, gridSize = 10000): number {
  const a = -10, b = 10;
  const zs: number[] = [];
  for (let i = 0; i < gridSize; i++) zs.push(a + (b - a) * i / (gridSize - 1));
  const dz = zs[1] - zs[0];
  const logJoint = zs.map(z => {
    let logPrior = -0.5 * Math.log(2 * Math.PI * model.tau2) - z * z / (2 * model.tau2);
    let logLik = 0;
    for (const x of model.data) {
      logLik += -0.5 * Math.log(2 * Math.PI * model.sigma2)
              - (x - z) * (x - z) / (2 * model.sigma2);
    }
    return logPrior + logLik;
  });
  const m = Math.max(...logJoint);
  let s = 0;
  for (const lj of logJoint) s += Math.exp(lj - m);
  return m + Math.log(s * dz);
}

export function elboGaussian(model: GaussianConjugate, phi_mu: number, phi_var: number): number {
  let recon = 0;
  for (const x of model.data) {
    recon += -0.5 * Math.log(2 * Math.PI * model.sigma2)
           - ((x - phi_mu) * (x - phi_mu) + phi_var) / (2 * model.sigma2);
  }
  const kl = klGaussian(phi_mu, Math.sqrt(phi_var), 0, Math.sqrt(model.tau2));
  return recon - kl;
}
