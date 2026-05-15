import { browserSched, makeLinearScale, DOMAIN, generateDataset } from './_shared';
import { sigmaEff } from '../math/score-conversion';

const T = browserSched.T;

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Two framings, one object — DDPM ↔ score matching</div>
      <div class="ddpm-dual">
        <div class="ddpm-dual__col">
          <h4 style="color:#2c5f8d;">DDPM</h4>
          <canvas id="se-ddpm" width="320" height="320" style="width:100%;height:auto;background:var(--paper);border-radius:4px;"></canvas>
          <div class="ddpm-readout" id="se-ddpm-readout"></div>
        </div>
        <div class="ddpm-dual__bridge" id="se-bridge"></div>
        <div class="ddpm-dual__col">
          <h4 style="color:#b8651a;">Score Matching</h4>
          <canvas id="se-sm" width="320" height="320" style="width:100%;height:auto;background:var(--paper);border-radius:4px;"></canvas>
          <div class="ddpm-readout" id="se-sm-readout"></div>
        </div>
      </div>
      <div class="viz-controls" style="margin-top:0.6rem;">
        <label class="viz-label">DDPM t = <span id="se-t">50</span>  ⇄  SM σ = <span id="se-sigma">…</span>
          <input type="range" id="se-slider" min="1" max="${T-1}" value="50" style="width:300px;">
        </label>
      </div>
      <div class="viz-caption">Both panels show the same data under the same noise corruption — left in DDPM coordinates, right in score-matching coordinates. The bridge shows the dictionary entry that converts between them.</div>
    </div>`;

  const data = generateDataset(7, 80);
  const seed = data.map(() => [Math.random() * 2 - 1, Math.random() * 2 - 1]);

  function setupPanel(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    const sx = makeLinearScale(DOMAIN, [16, W-16]);
    const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [16, H-16]);
    return { ctx, sx, sy, W, H };
  }
  const ddpmP = setupPanel(container.querySelector('#se-ddpm') as HTMLCanvasElement);
  const smP   = setupPanel(container.querySelector('#se-sm')   as HTMLCanvasElement);

  let t = 50;
  function render() {
    const ab = browserSched.alpha_bars[t];
    const sA = Math.sqrt(ab), sN = Math.sqrt(1 - ab);
    const sigma = sigmaEff(t, browserSched);

    [ddpmP, smP].forEach((p, k) => {
      const { ctx, sx, sy, W, H } = p;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = 'rgba(0,0,0,0.04)';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = k === 0 ? 'rgba(44,95,141,0.7)' : 'rgba(184,101,26,0.7)';
      for (let i = 0; i < data.length; i++) {
        const [dx, dy] = data[i];
        const [nx, ny] = seed[i];
        let x: number, y: number;
        if (k === 0) { x = sA * dx + sN * nx; y = sA * dy + sN * ny; }
        else         { x = dx + sigma * nx; y = dy + sigma * ny; }
        ctx.beginPath(); ctx.arc(sx(x), sy(y), 2.0, 0, Math.PI * 2); ctx.fill();
      }
    });
    (container.querySelector('#se-t') as HTMLElement).textContent = String(t);
    (container.querySelector('#se-sigma') as HTMLElement).textContent = sigma.toFixed(3);
    (container.querySelector('#se-ddpm-readout') as HTMLElement).innerHTML =
      `x_t = √ᾱ · x_0 + √(1−ᾱ) · ε <br>√ᾱ_t = ${sA.toFixed(3)},  √(1−ᾱ_t) = ${sN.toFixed(3)}`;
    (container.querySelector('#se-sm-readout') as HTMLElement).innerHTML =
      `x̃ = x + σ · ε<br>σ = √((1−ᾱ_t)/ᾱ_t) = ${sigma.toFixed(3)}`;
    (container.querySelector('#se-bridge') as HTMLElement).innerHTML =
      `<div style="text-align:center;">
         <div style="font-size:1.1rem;font-weight:600;">↔</div>
         <div style="margin-top:6px;">s_θ(x_t,t)<br>= −ε_θ / √(1−ᾱ_t)</div>
         <div style="margin-top:8px;">σ ↔ √((1−ᾱ_t)/ᾱ_t)</div>
       </div>`;
  }

  (container.querySelector('#se-slider') as HTMLInputElement).addEventListener('input', (e: any) => {
    t = +e.target.value; render();
  });
  render();
}
