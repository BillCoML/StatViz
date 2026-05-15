/**
 * §10 supplement: Figure 9 recreation. A grid of (t*, λ) cells, each showing
 *  where a reverse-sampled interpolation ends up in 2D.
 *
 * Reading the picture:
 *  - source points A = (-2, +2), B = (+2, -2) are two specific data clusters.
 *  - for each row t*: diffuse A and B to t*, then linearly interpolate
 *    between x_{t*}(A) and x_{t*}(B) for 11 lambdas.
 *  - for each cell: run the reverse chain from that interpolated point back
 *    to t=0 and plot the result.
 *
 *  Row t*=0      → no diffusion: linear interpolation in data space.
 *                  Cells trace a straight line A → B through the empty middle.
 *  Row t*=T-1    → full diffusion: starts are both 𝒩(0, I); interpolations
 *                  are essentially independent samples scattered over data.
 *  Middle rows   → smooth morph: cells land at intermediate clusters with
 *                  graded probability of A's vs B's mode.
 */
import {
  getWeights, mkRng, gauss2, browserSched, DOMAIN, makeLinearScale, generateDataset,
} from './_shared';
import { epsNetForward } from '../math/eps-net';
import { forwardSample } from '../math/forward-process';
import { reverseStep } from '../math/reverse-process';

const T = browserSched.T;
const T_STARS = [0, 25, 50, 75, T - 1];
const LAMBDAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

const SOURCE_A: [number, number] = [-2,  2];   // upper-left cluster
const SOURCE_B: [number, number] = [ 2, -2];   // lower-right cluster

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Coarse-to-fine interpolation — Figure 9 recreation</div>
      <div style="font-size:0.85rem;color:var(--ink-soft);margin-bottom:0.6rem;line-height:1.45;">
        Sources A (orange) and B (purple) live in opposite clusters: A = (−2, 2), B = (2, −2).
        Each row diffuses both to depth <span style="font-family:JetBrains Mono;">t*</span>,
        linearly interpolates by <span style="font-family:JetBrains Mono;">λ</span>, then
        runs the reverse chain. Blue dot in each cell = where the interpolation lands.
      </div>
      <div id="c2f-grid" class="ddpm-c2f" style="grid-template-columns:54px repeat(${LAMBDAS.length}, 1fr);"></div>
      <div id="c2f-status" class="ddpm-readout" style="margin-top:0.5rem;">Computing 55 reverse chains…</div>
      <div class="viz-caption" style="margin-top:0.4rem;">
        Row <strong>t*=0</strong>: blue dots trace a straight line A → B through the empty between-cluster region — linear interpolation in <em>data</em> space.
        Row <strong>t*=${T-1}</strong>: dots scatter randomly across all four clusters — at full diffusion the two sources are indistinguishable, so each cell is essentially an independent sample.
        Middle rows: <strong>smooth morph through latent space</strong> — λ near 0 lands near A's cluster, λ near 1 lands near B's, intermediate λ may land in either or in a different cluster entirely.
      </div>
    </div>`;

  const grid = container.querySelector('#c2f-grid') as HTMLElement;
  const status = container.querySelector('#c2f-status') as HTMLElement;
  const cellSize = 80;
  const data = generateDataset(5, 30);

  function makeCell(label?: string, isHeader = false): HTMLCanvasElement {
    const c = document.createElement('canvas');
    c.width = cellSize; c.height = cellSize;
    c.style.width = '100%'; c.style.aspectRatio = '1'; c.style.height = 'auto';
    c.className = 'ddpm-c2f__cell';
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = isHeader ? 'rgba(0,0,0,0.0)' : 'rgba(0,0,0,0.03)';
    ctx.fillRect(0, 0, cellSize, cellSize);
    if (label) {
      ctx.fillStyle = '#1c1c1c';
      ctx.font = 'bold 12px Fraunces, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, cellSize / 2, cellSize / 2);
    }
    return c;
  }

  // Header row: t* \ λ, then λ values
  grid.appendChild(makeCell('t*  \\  λ', true));
  for (const lam of LAMBDAS) grid.appendChild(makeCell(lam.toFixed(1), true));

  const cells: HTMLCanvasElement[][] = [];
  for (const ts of T_STARS) {
    grid.appendChild(makeCell(String(ts), true));
    const row: HTMLCanvasElement[] = [];
    for (let li = 0; li < LAMBDAS.length; li++) {
      const c = makeCell();
      grid.appendChild(c);
      row.push(c);
    }
    cells.push(row);
  }

  function paintCell(canvas: HTMLCanvasElement, result: number[], lam: number) {
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, cellSize, cellSize);
    ctx.fillStyle = 'rgba(0,0,0,0.03)';
    ctx.fillRect(0, 0, cellSize, cellSize);
    const sx = makeLinearScale(DOMAIN, [4, cellSize - 4]);
    const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [4, cellSize - 4]);
    // faded background data
    ctx.fillStyle = 'rgba(184,101,26,0.16)';
    for (const [dx, dy] of data) {
      ctx.beginPath(); ctx.arc(sx(dx), sy(dy), 1.2, 0, Math.PI * 2); ctx.fill();
    }
    // source A and B markers (always)
    ctx.fillStyle = 'rgba(184,101,26,0.55)';
    ctx.beginPath(); ctx.arc(sx(SOURCE_A[0]), sy(SOURCE_A[1]), 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(107,58,140,0.55)';
    ctx.beginPath(); ctx.arc(sx(SOURCE_B[0]), sy(SOURCE_B[1]), 3, 0, Math.PI * 2); ctx.fill();
    // result dot — color-blend A↔B by lambda for cohesion
    const r = Math.round(184 * (1 - lam) + 44  * lam);
    const g = Math.round(101 * (1 - lam) + 95  * lam);
    const b = Math.round(26  * (1 - lam) + 141 * lam);
    ctx.fillStyle = `rgba(${r},${g},${b},0.95)`;
    ctx.beginPath();
    ctx.arc(sx(result[0]), sy(result[1]), 5.5, 0, Math.PI * 2);
    ctx.fill();
    // small ring outline
    ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  function compute() {
    getWeights().then(w => {
      if (!w) { status.textContent = 'Failed to load trained model weights.'; return; }
      const rng = mkRng(11);
      let completed = 0;
      const total = T_STARS.length * LAMBDAS.length;

      // Run row-by-row asynchronously so the UI stays responsive.
      let r = 0;
      const step = () => {
        if (r >= T_STARS.length) {
          status.textContent = `Done — ${total} reverse chains rendered.`;
          return;
        }
        const tStar = T_STARS[r];
        const epsA = gauss2(rng); const epsB = gauss2(rng);
        const xtA = forwardSample(SOURCE_A, tStar, epsA, browserSched);
        const xtB = forwardSample(SOURCE_B, tStar, epsB, browserSched);
        for (let li = 0; li < LAMBDAS.length; li++) {
          const lam = LAMBDAS[li];
          let x: number[] = [(1 - lam) * xtA[0] + lam * xtB[0], (1 - lam) * xtA[1] + lam * xtB[1]];
          for (let ti = tStar; ti >= 0; ti--) {
            const eps = epsNetForward(x, ti, w);
            const z = ti > 0 ? gauss2(rng) : [0, 0];
            x = reverseStep(x, eps, ti, z, browserSched, 'beta');
          }
          paintCell(cells[r][li], x, lam);
          completed++;
        }
        status.textContent = `Computing… row ${r + 1}/${T_STARS.length} (${completed}/${total} cells)`;
        r++;
        setTimeout(step, 0);  // yield to browser
      };
      step();
    });
  }

  // Defer to allow the placeholder grid to render first.
  setTimeout(compute, 50);
}
