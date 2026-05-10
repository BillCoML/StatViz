export function bimodalPosterior(z: number): number {
  const g1 = Math.exp(-(z + 3) * (z + 3) / 2) / Math.sqrt(2 * Math.PI);
  const g2 = Math.exp(-(z - 3) * (z - 3) / 2) / Math.sqrt(2 * Math.PI);
  return 0.5 * g1 + 0.5 * g2;
}

export function elboBimodal(phi_mu: number, phi_var: number, gridSize = 4000): number {
  const a = -8, b = 8;
  const zs: number[] = [];
  for (let i = 0; i < gridSize; i++) zs.push(a + (b - a) * i / (gridSize - 1));
  const dz = zs[1] - zs[0];
  let s = 0;
  for (const z of zs) {
    const pUnnorm = bimodalPosterior(z);
    const dz_mu = z - phi_mu;
    const q = Math.exp(-dz_mu * dz_mu / (2 * phi_var)) / Math.sqrt(2 * Math.PI * phi_var);
    if (q < 1e-30) continue;
    const logRatio = Math.log(Math.max(pUnnorm, 1e-30)) - Math.log(q);
    s += q * logRatio * dz;
  }
  return s;
}
