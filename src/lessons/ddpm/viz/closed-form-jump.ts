import { browserSched, makeLinearScale, DOMAIN, generateDataset, mkRng, gauss2 } from './_shared';
import { forwardSample } from '../math/forward-process';

const T = browserSched.T;
const Tlevels = [25, 50, 75, T - 1];

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Iterative vs closed-form forward sampling — both yield the same distribution</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        <div>
          <h5 style="margin:0 0 0.3rem;">Iterative: ${T} small steps</h5>
          <canvas id="cfj-iter" width="320" height="320" style="width:100%;height:auto;background:var(--paper);border-radius:4px;"></canvas>
        </div>
        <div>
          <h5 style="margin:0 0 0.3rem;">Closed-form jump (one step)</h5>
          <canvas id="cfj-jump" width="320" height="320" style="width:100%;height:auto;background:var(--paper);border-radius:4px;"></canvas>
        </div>
      </div>
      <div class="viz-controls" style="margin-top:0.6rem;">
        <span class="viz-label">Target t:</span>
        ${Tlevels.map(t => `<button class="viz-btn-sm" data-t="${t}">${t}</button>`).join('')}
      </div>
      <div class="viz-caption">Picking any t, both panels produce identical-distribution samples. Closed form is O(1); iterative is O(t).</div>
    </div>`;

  const data = generateDataset(33, 60);
  const rng = mkRng(2);
  const noises = data.map(() => [gauss2(rng)[0], gauss2(rng)[1]]);

  function setup(c: HTMLCanvasElement) {
    const ctx = c.getContext('2d')!;
    const W = c.width, H = c.height;
    const sx = makeLinearScale(DOMAIN, [16, W-16]);
    const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [16, H-16]);
    return { ctx, sx, sy, W, H };
  }
  const iterP = setup(container.querySelector('#cfj-iter') as HTMLCanvasElement);
  const jumpP = setup(container.querySelector('#cfj-jump') as HTMLCanvasElement);

  function render(targetT: number) {
    [iterP, jumpP].forEach(p => { p.ctx.clearRect(0, 0, p.W, p.H); p.ctx.fillStyle = 'rgba(0,0,0,0.04)'; p.ctx.fillRect(0, 0, p.W, p.H); });
    // Closed form
    jumpP.ctx.fillStyle = 'rgba(44,95,141,0.7)';
    for (let i = 0; i < data.length; i++) {
      const x = forwardSample(data[i], targetT, noises[i], browserSched);
      jumpP.ctx.beginPath(); jumpP.ctx.arc(jumpP.sx(x[0]), jumpP.sy(x[1]), 2.4, 0, Math.PI*2); jumpP.ctx.fill();
    }
    // Iterative: simulate by composing single-step noise
    iterP.ctx.fillStyle = 'rgba(44,95,141,0.7)';
    const localRng = mkRng(13);
    for (let i = 0; i < data.length; i++) {
      let x = [data[i][0], data[i][1]];
      for (let ti = 0; ti <= targetT; ti++) {
        const eps = gauss2(localRng);
        const a = Math.sqrt(1 - browserSched.betas[ti]);
        const b = Math.sqrt(browserSched.betas[ti]);
        x = [a * x[0] + b * eps[0], a * x[1] + b * eps[1]];
      }
      iterP.ctx.beginPath(); iterP.ctx.arc(iterP.sx(x[0]), iterP.sy(x[1]), 2.4, 0, Math.PI*2); iterP.ctx.fill();
    }
  }
  render(Tlevels[0]);
  container.querySelectorAll<HTMLButtonElement>('[data-t]').forEach(b => {
    b.addEventListener('click', () => render(+b.dataset.t!));
  });
}
