/**
 * §10 supplement: Figure 9 recreation. Grid of t* x λ.
 * Cells render the reverse-sampled output. Heavy compute; precomputed at mount.
 */
import { getWeights, mkRng, gauss2, browserSched, DOMAIN, makeLinearScale, generateDataset } from './_shared';
import { epsNetForward } from '../math/eps-net';
import { forwardSample } from '../math/forward-process';
import { reverseStep } from '../math/reverse-process';

const T = browserSched.T;
const T_STARS = [0, 25, 50, 75, T - 1];
const LAMBDAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Coarse-to-fine interpolation (Figure 9 recreation)</div>
      <div id="c2f-grid" class="ddpm-c2f" style="grid-template-columns:repeat(${LAMBDAS.length + 1}, 1fr);"></div>
      <div class="viz-caption">Rows: diffusion depth t*. Columns: interpolation parameter λ. Row t*=0: linear interpolation in data space. Row t*=T−1: independent samples. Middle rows: smooth morphing through latent space.</div>
    </div>`;

  const grid = container.querySelector('#c2f-grid') as HTMLElement;
  const cellSize = 70;

  function makeCell(x: number[][] | null, label?: string) {
    const c = document.createElement('canvas');
    c.width = cellSize; c.height = cellSize;
    c.style.width = '100%'; c.style.aspectRatio = '1'; c.style.height = 'auto';
    c.className = 'ddpm-c2f__cell';
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = 'rgba(0,0,0,0.05)'; ctx.fillRect(0, 0, cellSize, cellSize);
    const sx = makeLinearScale(DOMAIN, [3, cellSize-3]);
    const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [3, cellSize-3]);
    if (x) {
      ctx.fillStyle = 'rgba(44,95,141,0.85)';
      for (const p of x) { ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 2, 0, Math.PI*2); ctx.fill(); }
    }
    if (label) {
      ctx.fillStyle = '#1c1c1c'; ctx.font = '10px JetBrains Mono';
      ctx.fillText(label, 4, 12);
    }
    return c;
  }

  // Header row: lambdas
  grid.appendChild(makeCell(null, 't*\\λ'));
  for (const lam of LAMBDAS) grid.appendChild(makeCell(null, lam.toFixed(1)));

  // Skeleton with placeholders
  const placeholders: HTMLCanvasElement[][] = [];
  for (const ts of T_STARS) {
    const row: HTMLCanvasElement[] = [];
    grid.appendChild(makeCell(null, `${ts}`));
    for (let _li = 0; _li < LAMBDAS.length; _li++) {
      const c = makeCell(null);
      grid.appendChild(c);
      row.push(c);
    }
    placeholders.push(row);
  }

  // Generate samples to interpolate from the data clusters
  const rng = mkRng(7);
  const data = generateDataset(5, 20);
  const A = data[Math.floor(rng() * data.length)];
  const B = data[Math.floor(rng() * data.length) + 10];

  getWeights().then(w => {
    if (!w) return;
    for (let r = 0; r < T_STARS.length; r++) {
      const tStar = T_STARS[r];
      // Diffuse A and B to t*
      const epsA = gauss2(rng); const epsB = gauss2(rng);
      const xtA = forwardSample(A, tStar, epsA, browserSched);
      const xtB = forwardSample(B, tStar, epsB, browserSched);
      for (let c = 0; c < LAMBDAS.length; c++) {
        const lam = LAMBDAS[c];
        let x = [(1 - lam) * xtA[0] + lam * xtB[0], (1 - lam) * xtA[1] + lam * xtB[1]];
        for (let ti = tStar; ti >= 0; ti--) {
          const eps = epsNetForward(x, ti, w);
          const z = ti > 0 ? gauss2(rng) : [0, 0];
          x = reverseStep(x, eps, ti, z, browserSched, 'beta');
        }
        // Render single dot
        const canvas = placeholders[r][c];
        const ctx = canvas.getContext('2d')!;
        ctx.clearRect(0, 0, cellSize, cellSize);
        ctx.fillStyle = 'rgba(0,0,0,0.05)'; ctx.fillRect(0, 0, cellSize, cellSize);
        const sx = makeLinearScale(DOMAIN, [3, cellSize-3]);
        const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [3, cellSize-3]);
        // tiny faded data
        ctx.fillStyle = 'rgba(184,101,26,0.18)';
        for (const [dx, dy] of data) { ctx.beginPath(); ctx.arc(sx(dx), sy(dy), 1, 0, Math.PI*2); ctx.fill(); }
        ctx.fillStyle = 'rgba(44,95,141,0.85)';
        ctx.beginPath(); ctx.arc(sx(x[0]), sy(x[1]), 3.5, 0, Math.PI*2); ctx.fill();
      }
    }
  });
}
