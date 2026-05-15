/**
 * §4: stacked bar showing relative magnitudes of L_T, sum L_{t-1}, L_0;
 * line chart below showing per-timestep VLB weight.
 */
import { paperSchedule } from '../math/schedule';
import { vlbWeight, L_T_perDim, NATS_TO_BITS } from '../math/vlb';
import { makeLinearScale } from './_shared';

export function mount(container: HTMLElement): void {
  const sched = paperSchedule();
  const T = sched.T;
  const weights = Array.from({ length: T - 1 }, (_, i) => vlbWeight(i + 1, sched, 'beta'));
  // crude relative magnitudes (illustrative for paper schedule)
  const LT = L_T_perDim(sched) * NATS_TO_BITS;         // ~3e-5 bits/dim
  const sumLt = weights.reduce((a, b) => a + b, 0);     // proportional to total L_{t-1}
  const L0 = 0.05 * sumLt;                              // illustrative

  const total = LT + sumLt + L0;
  const ltPct = LT / total, ltminus1Pct = sumLt / total, l0Pct = L0 / total;

  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">VLB decomposition — L = L_T + Σ L_{t−1} + L_0</div>
      <div style="position:relative;height:46px;background:#efece2;border-radius:6px;overflow:hidden;display:flex;font-family:JetBrains Mono;font-size:11px;color:#fff;">
        <div style="width:${(ltPct*100).toFixed(2)}%;background:#999;display:flex;align-items:center;justify-content:center;">L_T</div>
        <div style="width:${(ltminus1Pct*100).toFixed(2)}%;background:#b54050;display:flex;align-items:center;justify-content:center;">Σ L_{t−1} (dominant)</div>
        <div style="width:${(l0Pct*100).toFixed(2)}%;background:#3a6b8c;display:flex;align-items:center;justify-content:center;">L_0</div>
      </div>
      <div style="margin-top:0.4rem;font-family:JetBrains Mono;font-size:0.8rem;color:var(--ink-soft);">
        L_T ≈ ${LT.toExponential(2)} bits/dim — constant in θ, drop. L_{t−1} is the dominant signal. L_0 is the reconstruction term (discretized Gaussian for image data).
      </div>
      <h5 style="margin:1.2rem 0 0.4rem;">Per-timestep VLB weight β_t² / (2σ_t²α_t(1−ᾱ_t))</h5>
      <canvas id="vlb-curve" width="700" height="220" style="width:100%;height:auto;background:var(--paper);border-radius:6px;"></canvas>
      <div class="viz-caption">Weight is heavily concentrated at small t (easy denoising). Dropping these weights (→ L_simple) shifts capacity toward harder large-t denoising.</div>
    </div>`;

  const canvas = container.querySelector('#vlb-curve') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  const maxW = Math.max(...weights);
  const sx = makeLinearScale([1, T - 1], [40, W - 20]);
  const sy = makeLinearScale([0, maxW * 1.05], [H - 30, 20]);
  // axes
  ctx.strokeStyle = 'rgba(0,0,0,0.15)';
  ctx.beginPath(); ctx.moveTo(40, H-30); ctx.lineTo(W-20, H-30); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(40, 20); ctx.lineTo(40, H-30); ctx.stroke();
  // log-scale line (visualize the dramatic peak)
  ctx.strokeStyle = '#b54050'; ctx.lineWidth = 1.6; ctx.beginPath();
  for (let i = 0; i < weights.length; i++) {
    const x = sx(i + 1), y = sy(weights[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#1c1c1c'; ctx.font = '11px JetBrains Mono';
  ctx.fillText('t = 1', 40, H - 12);
  ctx.fillText(`t = ${T - 1}`, W - 60, H - 12);
  ctx.fillText(`weight(1) ≈ ${weights[0].toExponential(2)}`, 60, 36);
  ctx.fillText(`weight(500) ≈ ${weights[499].toExponential(2)}`, 60, 52);
}
