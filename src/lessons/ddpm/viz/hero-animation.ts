import {
  getWeights, generateDataset, mkRng, gauss2, browserSched, makeLinearScale, DOMAIN,
} from './_shared';
import { epsNetForward } from '../math/eps-net';
import { reverseStep } from '../math/reverse-process';

const T = browserSched.T;
const SNAPSHOT_TS = [0, 14, 29, 43, 57, 71, 86, T - 1];

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">The lesson in one image</div>
      <canvas id="hero-canvas" width="960" height="380" style="width:100%;height:auto;display:block;background:var(--paper,#fafaf6);border-radius:6px;"></canvas>
      <div class="viz-caption">Top: forward process turning 4-cluster data into Gaussian noise. Bottom: the trained model reversing it.</div>
    </div>`;

  const canvas = container.querySelector('#hero-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  const panelW = W / 8;
  const data = generateDataset(13, 80);
  const rng = mkRng(5);
  const noises = data.map(() => [gauss2(rng)[0], gauss2(rng)[1]]);
  let reverseSnaps: number[][][] = SNAPSHOT_TS.map(() => []);

  function drawPanel(idx: number, points: number[][], stripeY: number, label: string, color: string) {
    const x0 = idx * panelW;
    const x1 = (idx + 1) * panelW;
    const sx = makeLinearScale(DOMAIN, [x0 + 6, x1 - 6]);
    const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [stripeY + 6, stripeY + 160 - 6]);
    ctx.strokeStyle = 'rgba(0,0,0,0.08)';
    ctx.strokeRect(x0 + 2, stripeY + 2, panelW - 4, 156);
    ctx.fillStyle = color;
    for (const p of points) { ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 1.4, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = '#1c1c1c';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText(label, x0 + 6, stripeY + 168);
  }

  function renderAll() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#1c1c1c'; ctx.font = '600 13px Fraunces, serif';
    ctx.fillText('Forward (data → noise)', 8, 14);
    ctx.fillText('Reverse (noise → data, trained DDPM)', 8, 198);
    for (let i = 0; i < SNAPSHOT_TS.length; i++) {
      const t = SNAPSHOT_TS[i];
      const ab = browserSched.alpha_bars[t];
      const sA = Math.sqrt(ab), sN = Math.sqrt(1 - ab);
      const fwd: number[][] = data.map((p, j) => [sA * p[0] + sN * noises[j][0], sA * p[1] + sN * noises[j][1]]);
      drawPanel(i, fwd, 20, `t=${t}`, 'rgba(184,101,26,0.6)');
      drawPanel(i, reverseSnaps[SNAPSHOT_TS.length - 1 - i], 200, `t=${SNAPSHOT_TS[SNAPSHOT_TS.length - 1 - i]}`, 'rgba(44,95,141,0.7)');
    }
  }

  getWeights().then(w => {
    if (!w) { renderAll(); return; }
    let xs: number[][] = Array.from({ length: 80 }, () => [...gauss2(rng)]);
    const snapSet = new Set(SNAPSHOT_TS);
    const snaps: Record<number, number[][]> = {};
    for (let ti = T - 1; ti >= 0; ti--) {
      if (snapSet.has(ti)) snaps[ti] = xs.map(p => [p[0], p[1]]);
      xs = xs.map(p => {
        const eps = epsNetForward(p, ti, w);
        const z = ti > 0 ? gauss2(rng) : [0, 0];
        return reverseStep(p, eps, ti, z, browserSched, 'beta');
      });
    }
    snaps[0] = xs;
    reverseSnaps = SNAPSHOT_TS.map(t => snaps[t] ?? []);
    renderAll();
  });

  renderAll();
}
