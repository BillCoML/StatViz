import { browserSched, makeLinearScale, DOMAIN, generateDataset } from './_shared';
import { posteriorMean, posteriorVar } from '../math/forward-process';

const T = browserSched.T;

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Forward posterior q(x_{t-1} | x_t, x_0)</div>
      <canvas id="fp-canvas" width="600" height="500" style="width:100%;height:auto;display:block;background:var(--paper,#fafaf6);border-radius:6px;cursor:move;"></canvas>
      <div class="viz-controls" style="margin-top:0.6rem;">
        <label class="viz-label">t: <span id="fp-tval">50</span>
          <input type="range" id="fp-slider" min="1" max="${T-1}" value="50" style="width:240px;">
        </label>
        <span class="ddpm-readout" id="fp-readout"></span>
      </div>
      <div class="viz-caption">Orange = x₀ (drag). Blue = x_t (drag). Amber = posterior mean μ̃_t with β̃_t circle. Move t to see the convex combination shift between x₀ and x_t.</div>
    </div>`;

  const canvas = container.querySelector('#fp-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  const sx = makeLinearScale(DOMAIN, [40, W-40]);
  const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [40, H-40]);
  const data = generateDataset(19, 30);

  let t = 50;
  const x0: [number, number] = [2, 2];
  const xt: [number, number] = [-1, -1];
  let dragging: 'x0' | 'xt' | null = null;

  function render() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(40, sy(i)); ctx.lineTo(W-40, sy(i)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(i), 40); ctx.lineTo(sx(i), H-40); ctx.stroke();
    }
    ctx.fillStyle = 'rgba(184,101,26,0.13)';
    for (const [dx, dy] of data) { ctx.beginPath(); ctx.arc(sx(dx), sy(dy), 2, 0, Math.PI*2); ctx.fill(); }
    const mu = posteriorMean(xt, x0, t, browserSched) as [number, number];
    const tb = posteriorVar(t, browserSched);
    const sd = Math.sqrt(tb);
    const rr = sx(sd) - sx(0);
    ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sx(x0[0]), sy(x0[1])); ctx.lineTo(sx(xt[0]), sy(xt[1])); ctx.stroke();
    ctx.strokeStyle = '#d4a437'; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(sx(mu[0]), sy(mu[1]), rr, 0, Math.PI*2); ctx.stroke();
    ctx.fillStyle = '#b8651a'; ctx.beginPath(); ctx.arc(sx(x0[0]), sy(x0[1]), 9, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#2c5f8d'; ctx.beginPath(); ctx.arc(sx(xt[0]), sy(xt[1]), 9, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#d4a437'; ctx.beginPath(); ctx.arc(sx(mu[0]), sy(mu[1]), 6, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#1c1c1c'; ctx.font = '600 12px JetBrains Mono';
    ctx.fillText('x_0', sx(x0[0]) + 12, sy(x0[1]));
    ctx.fillText('x_t', sx(xt[0]) + 12, sy(xt[1]));
    ctx.fillText('μ̃_t', sx(mu[0]) + 10, sy(mu[1]));
    (container.querySelector('#fp-readout') as HTMLElement).textContent =
      `μ̃_t = (${mu[0].toFixed(2)}, ${mu[1].toFixed(2)})   β̃_t = ${tb.toFixed(4)}   β̃/β = ${(tb/browserSched.betas[t]).toFixed(2)}`;
    (container.querySelector('#fp-tval') as HTMLElement).textContent = String(t);
  }

  canvas.addEventListener('mousedown', (e) => {
    const r = canvas.getBoundingClientRect();
    const cx = (e.clientX - r.left) * (W / r.width);
    const cy = (e.clientY - r.top) * (H / r.height);
    const dx0 = Math.hypot(cx - sx(x0[0]), cy - sy(x0[1]));
    const dxt = Math.hypot(cx - sx(xt[0]), cy - sy(xt[1]));
    dragging = dx0 < dxt ? 'x0' : 'xt';
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const r = canvas.getBoundingClientRect();
    const cx = (e.clientX - r.left) * (W / r.width);
    const cy = (e.clientY - r.top) * (H / r.height);
    const dx = sx.invert(cx), dy = sy.invert(cy);
    if (dragging === 'x0') { x0[0] = dx; x0[1] = dy; }
    else { xt[0] = dx; xt[1] = dy; }
    render();
  });
  canvas.addEventListener('mouseup', () => { dragging = null; });
  canvas.addEventListener('mouseleave', () => { dragging = null; });
  (container.querySelector('#fp-slider') as HTMLInputElement).addEventListener('input', (e: any) => {
    t = +e.target.value; render();
  });
  render();
}
