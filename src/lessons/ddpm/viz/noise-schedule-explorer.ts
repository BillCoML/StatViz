/**
 * §11: compare linear vs cosine vs sigmoid schedules — beta_t and sqrt(alpha_bar_t) curves.
 */
import { linearSchedule, cosineSchedule, sigmoidSchedule } from '../math/schedule';
import { makeLinearScale } from './_shared';

const T = 1000;

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Noise schedule comparison</div>
      <div class="viz-controls">
        <span class="viz-label">Schedule:</span>
        <label class="viz-label"><input type="radio" name="ns-sched" value="linear" checked> Linear (paper)</label>
        <label class="viz-label"><input type="radio" name="ns-sched" value="cosine"> Cosine (IDDPM)</label>
        <label class="viz-label"><input type="radio" name="ns-sched" value="sigmoid"> Sigmoid</label>
      </div>
      <h5 style="margin:0 0 0.3rem;">β_t</h5>
      <canvas id="ns-beta" width="800" height="180" style="width:100%;height:auto;background:var(--paper);border-radius:6px;"></canvas>
      <h5 style="margin:0.8rem 0 0.3rem;">√(ᾱ_t) — signal strength</h5>
      <canvas id="ns-ab" width="800" height="180" style="width:100%;height:auto;background:var(--paper);border-radius:6px;"></canvas>
      <div class="viz-caption">Linear is the paper's choice. Cosine destroys information more uniformly — better for harder datasets.</div>
    </div>`;

  function build(name: string) {
    if (name === 'cosine') return cosineSchedule(T);
    if (name === 'sigmoid') return sigmoidSchedule(T, 1e-4, 0.02);
    return linearSchedule(T, 1e-4, 0.02);
  }

  function plot(canvas: HTMLCanvasElement, ys: number[], color: string, yMax?: number) {
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(0,0,0,0.03)'; ctx.fillRect(0, 0, W, H);
    const sx = makeLinearScale([0, T - 1], [30, W - 10]);
    const sy = makeLinearScale([0, yMax ?? Math.max(...ys) * 1.05], [H - 25, 15]);
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.beginPath(); ctx.moveTo(30, H-25); ctx.lineTo(W-10, H-25); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(30, 15); ctx.lineTo(30, H-25); ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < ys.length; i += 5) {
      const x = sx(i), y = sy(ys[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = '#1c1c1c'; ctx.font = '10px JetBrains Mono';
    ctx.fillText('t=0', 32, H - 8); ctx.fillText('t=' + (T - 1), W - 50, H - 8);
  }

  function refresh() {
    const sel = (container.querySelector<HTMLInputElement>('input[name="ns-sched"]:checked'))!.value;
    const s = build(sel);
    plot(container.querySelector('#ns-beta') as HTMLCanvasElement, s.betas, '#b54050', 0.025);
    plot(container.querySelector('#ns-ab')   as HTMLCanvasElement, s.alpha_bars.map(Math.sqrt), '#5a8a6a', 1);
  }
  container.querySelectorAll('input[name="ns-sched"]').forEach(el => el.addEventListener('change', refresh));
  refresh();
}
