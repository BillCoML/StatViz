/**
 * §9 visualization: animated reverse process on the trained DDPM,
 * with optional x̂_0 overlay and score-field quiver.
 */
import { getWeights, generateDataset, mkRng, gauss2, browserSched, makeLinearScale, DOMAIN } from './_shared';
import type { DDPMWeights } from '../math/eps-net';
import { epsNetForward } from '../math/eps-net';
import { xHat0 } from '../math/forward-process';
import { reverseStep } from '../math/reverse-process';

const T = browserSched.T;
const N = 100;

interface State {
  weights: DDPMWeights | null;
  particles: number[][];
  t: number;
  playing: boolean;
  rng: () => number;
  rafId: number | null;
  showXHat: boolean;
  showScore: boolean;
  scoreCache: { gx: number; gy: number; vx: number; vy: number }[] | null;
  scoreCacheT: number;
}

const TEMPLATE = `
  <div class="viz-container">
    <div class="viz-title">Algorithm 2 in action — reverse sampling on the trained DDPM</div>
    <canvas id="sc-canvas" width="640" height="500" style="width:100%;height:auto;display:block;background:var(--paper,#fafaf6);border-radius:6px;"></canvas>
    <div class="viz-controls" style="margin-top:0.6rem;">
      <button class="viz-btn" id="sc-play">Play</button>
      <button class="viz-btn" id="sc-step">Step</button>
      <button class="viz-btn" id="sc-reset">Reset</button>
      <label class="viz-label"><input type="checkbox" id="sc-xhat"> Show x̂₀</label>
      <label class="viz-label"><input type="checkbox" id="sc-score"> Show score field</label>
      <span class="ddpm-readout" id="sc-readout" style="margin-left:auto;"></span>
    </div>
    <div class="viz-caption">100 particles start at 𝒩(0, I) and run through ${T} reverse steps. Toggle overlays to see the model's running x̂₀ guesses or the learned score field at the current t.</div>
  </div>
`;

export function mount(container: HTMLElement): void {
  container.innerHTML = TEMPLATE;
  const canvas = container.querySelector('#sc-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  const sx = makeLinearScale(DOMAIN, [40, W - 40]);
  const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [40, H - 40]);
  const data = generateDataset();

  const state: State = {
    weights: null, particles: [], t: T - 1,
    playing: false, rng: mkRng(7), rafId: null,
    showXHat: false, showScore: false,
    scoreCache: null, scoreCacheT: -1,
  };

  function reset() {
    state.t = T - 1;
    state.particles = Array.from({ length: N }, () => gauss2(state.rng));
    state.scoreCache = null;
  }

  function step() {
    if (!state.weights) return;
    const w = state.weights;
    state.particles = state.particles.map(p => {
      const eps = epsNetForward(p, state.t, w);
      const z = state.t > 0 ? gauss2(state.rng) : [0, 0];
      return reverseStep(p, eps, state.t, z, browserSched, 'beta');
    });
    if (state.t > 0) state.t--;
    else state.playing = false;
  }

  function ensureScoreCache() {
    if (!state.weights || state.scoreCacheT === state.t) return;
    const w = state.weights;
    const G = 16;
    const cache = [];
    const denom = Math.sqrt(1 - browserSched.alpha_bars[state.t]);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const gx = -4 + 8 * i / (G - 1);
      const gy = -4 + 8 * j / (G - 1);
      const eps = epsNetForward([gx, gy], state.t, w);
      cache.push({ gx, gy, vx: -eps[0] / denom, vy: -eps[1] / denom });
    }
    state.scoreCache = cache;
    state.scoreCacheT = state.t;
  }

  function render() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(40, sy(i)); ctx.lineTo(W-40, sy(i)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(i), 40); ctx.lineTo(sx(i), H-40); ctx.stroke();
    }
    ctx.fillStyle = 'rgba(184, 101, 26, 0.18)';
    for (const [dx, dy] of data) { ctx.beginPath(); ctx.arc(sx(dx), sy(dy), 2, 0, Math.PI*2); ctx.fill(); }
    if (state.showScore && state.weights) {
      ensureScoreCache();
      ctx.strokeStyle = 'rgba(44, 95, 141, 0.5)';
      for (const a of state.scoreCache!) {
        const len = Math.hypot(a.vx, a.vy);
        const scale = Math.min(0.3 / Math.max(len, 0.01), 0.5);
        ctx.beginPath();
        ctx.moveTo(sx(a.gx), sy(a.gy));
        ctx.lineTo(sx(a.gx + a.vx*scale), sy(a.gy + a.vy*scale));
        ctx.stroke();
      }
    }
    if (state.showXHat && state.weights) {
      ctx.fillStyle = 'rgba(184, 101, 26, 0.35)';
      for (const p of state.particles) {
        const xh = xHat0(p, epsNetForward(p, state.t, state.weights), state.t, browserSched);
        ctx.beginPath(); ctx.arc(sx(xh[0]), sy(xh[1]), 2.2, 0, Math.PI*2); ctx.fill();
      }
    }
    ctx.fillStyle = 'rgba(44, 95, 141, 0.75)';
    for (const p of state.particles) { ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 2.6, 0, Math.PI*2); ctx.fill(); }
    const ab = browserSched.alpha_bars[state.t];
    (container.querySelector('#sc-readout') as HTMLElement).textContent =
      `t=${state.t}  √ᾱ=${Math.sqrt(ab).toFixed(3)}  σ_t=${Math.sqrt(browserSched.betas[state.t]).toFixed(4)}`;
  }

  function loop() { step(); render(); if (state.playing) state.rafId = requestAnimationFrame(loop); }

  container.querySelector('#sc-play')!.addEventListener('click', () => {
    state.playing = !state.playing;
    (container.querySelector('#sc-play') as HTMLElement).textContent = state.playing ? 'Pause' : 'Play';
    if (state.playing) loop();
  });
  container.querySelector('#sc-step')!.addEventListener('click', () => { step(); render(); });
  container.querySelector('#sc-reset')!.addEventListener('click', () => { reset(); render(); });
  (container.querySelector('#sc-xhat') as HTMLInputElement).addEventListener('change', (e: any) => { state.showXHat = e.target.checked; render(); });
  (container.querySelector('#sc-score') as HTMLInputElement).addEventListener('change', (e: any) => { state.showScore = e.target.checked; state.scoreCache = null; render(); });

  getWeights().then(w => { state.weights = w; reset(); render(); });
}
