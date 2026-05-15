import { generateDataset, mkRng, boxMuller, browserSched, makeLinearScale, DOMAIN } from './_shared';

const T = browserSched.T;

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Forward chain — destroying signal</div>
      <canvas id="fc-canvas" width="640" height="480" style="width:100%;height:auto;display:block;background:var(--paper,#fafaf6);border-radius:6px;"></canvas>
      <div class="viz-controls" style="margin-top:0.6rem;">
        <label class="viz-label">t: <span id="fc-tval">0</span>
          <input type="range" id="fc-slider" min="0" max="${T-1}" value="0" style="width:240px;">
        </label>
        <button class="viz-btn" id="fc-anim">Animate forward</button>
        <button class="viz-btn" id="fc-reset">Reset</button>
        <span class="ddpm-readout" id="fc-readout" style="margin-left:auto;"></span>
      </div>
      <div class="viz-caption">Drag t. The 1000-point training cloud morphs from 4 clusters into 𝒩(0, I). The highlighted point (large) tracks one specific x₀ through the chain.</div>
    </div>`;

  const canvas = container.querySelector('#fc-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  const sx = makeLinearScale(DOMAIN, [40, W-40]);
  const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [40, H-40]);
  const data = generateDataset(11, 250);
  const rng = mkRng(1);
  const noises = data.map(() => [boxMuller(rng), boxMuller(rng)]);
  const hi = 0;
  const hiNoise = noises[hi];

  let t = 0;
  let rafId: number | null = null;
  const slider = container.querySelector('#fc-slider') as HTMLInputElement;
  const tval = container.querySelector('#fc-tval') as HTMLElement;
  const readout = container.querySelector('#fc-readout') as HTMLElement;

  function render() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(40, sy(i)); ctx.lineTo(W-40, sy(i)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(i), 40); ctx.lineTo(sx(i), H-40); ctx.stroke();
    }
    const ab = browserSched.alpha_bars[t];
    const sA = Math.sqrt(ab), sN = Math.sqrt(1 - ab);
    ctx.fillStyle = 'rgba(184, 101, 26, 0.35)';
    for (let i = 0; i < data.length; i++) {
      const x = sA * data[i][0] + sN * noises[i][0];
      const y = sA * data[i][1] + sN * noises[i][1];
      ctx.beginPath(); ctx.arc(sx(x), sy(y), 1.4, 0, Math.PI*2); ctx.fill();
    }
    const hx = sA * data[hi][0] + sN * hiNoise[0];
    const hy = sA * data[hi][1] + sN * hiNoise[1];
    ctx.fillStyle = '#2c5f8d';
    ctx.beginPath(); ctx.arc(sx(hx), sy(hy), 6, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = 'rgba(44, 95, 141, 0.35)';
    ctx.lineWidth = 1.2;
    const rr = sx(sN) - sx(0);
    ctx.beginPath(); ctx.arc(sx(hx), sy(hy), rr, 0, Math.PI*2); ctx.stroke();
    readout.textContent = `t=${t}  √ᾱ=${sA.toFixed(3)}  √(1−ᾱ)=${sN.toFixed(3)}  SNR=${(ab/(1-ab+1e-12)).toFixed(3)}`;
    tval.textContent = String(t);
  }

  slider.addEventListener('input', () => { t = +slider.value; render(); });
  container.querySelector('#fc-reset')!.addEventListener('click', () => { t = 0; slider.value = '0'; render(); });
  container.querySelector('#fc-anim')!.addEventListener('click', () => {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; t = 0; }
    let start = performance.now();
    const step = (now: number) => {
      const dt = (now - start) / 1000;
      t = Math.min(T - 1, Math.floor(dt * (T / 6)));
      slider.value = String(t); render();
      if (t < T - 1) rafId = requestAnimationFrame(step);
    };
    rafId = requestAnimationFrame(step);
  });

  render();
}
