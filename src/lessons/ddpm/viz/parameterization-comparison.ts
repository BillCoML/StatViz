/**
 * §6: same trained-model intermediate state ($t = 500$, x_t fixed), three panels showing
 * Panel A: posterior-mean direct output, Panel B: x_0 prediction, Panel C: epsilon prediction.
 * Histograms show output magnitude scales.
 */
import { paperSchedule } from '../math/schedule';

export function mount(container: HTMLElement): void {
  const sched = paperSchedule();
  const t = 500;
  const ab = sched.alpha_bars[t];
  const sA = Math.sqrt(ab);
  // Simulated outputs (illustrative). Output scales:
  //  A: ~sA (small);  B: ~1 (data scale);  C: ~1 (unit Gaussian)
  const N = 200;
  function rnd() { return Math.random() * 2 - 1; }
  const A = Array.from({ length: N }, () => rnd() * sA);
  const B = Array.from({ length: N }, () => rnd());
  const C = Array.from({ length: N }, () => rnd());

  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Three parameterizations at t = ${t}: same model, different network output</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;">
        ${[
          {l:'A: μ_θ (predicts posterior mean)', d:A, range:'~√ᾱ_t scale', c:'#3a6b8c'},
          {l:'B: x_0 (predicts clean data)',      d:B, range:'~data scale', c:'#b8651a'},
          {l:'C: ε_θ (predicts noise)',           d:C, range:'unit Gaussian, all t', c:'#2c5f8d'},
        ].map((p, i) => `
          <div style="background:var(--paper-darker,#efece2);padding:0.5rem;border-radius:6px;">
            <h5 style="margin:0 0 0.3rem;">${p.l}</h5>
            <canvas id="pc-c${i}" width="280" height="140" style="width:100%;height:auto;background:var(--paper);border-radius:4px;"></canvas>
            <div style="font-family:JetBrains Mono;font-size:0.78rem;color:var(--ink-soft);margin-top:0.3rem;">${p.range}</div>
          </div>
        `).join('')}
      </div>
      <div class="viz-caption">All three outputs are mathematically equivalent (deriveable from each other). The C parameterization has the most stable scale across t, which is why it trains best with L_simple.</div>
    </div>`;

  function drawHist(canvas: HTMLCanvasElement, vals: number[], color: string) {
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = 'rgba(0,0,0,0.04)';
    ctx.fillRect(0, 0, W, H);
    const bins = 30;
    const hist = new Array(bins).fill(0);
    const lo = -1.2, hi = 1.2;
    for (const v of vals) {
      const idx = Math.floor((v - lo) / (hi - lo) * bins);
      if (idx >= 0 && idx < bins) hist[idx]++;
    }
    const m = Math.max(...hist) || 1;
    ctx.fillStyle = color;
    const barW = (W - 16) / bins;
    for (let i = 0; i < bins; i++) {
      const bh = (hist[i] / m) * (H - 24);
      ctx.fillRect(8 + i * barW, H - 12 - bh, barW - 1, bh);
    }
    ctx.fillStyle = '#1c1c1c'; ctx.font = '10px JetBrains Mono';
    ctx.fillText('-1.2', 8, H - 1);
    ctx.fillText('1.2', W - 22, H - 1);
  }
  drawHist(container.querySelector('#pc-c0') as HTMLCanvasElement, A, '#3a6b8c');
  drawHist(container.querySelector('#pc-c1') as HTMLCanvasElement, B, '#b8651a');
  drawHist(container.querySelector('#pc-c2') as HTMLCanvasElement, C, '#2c5f8d');
}
