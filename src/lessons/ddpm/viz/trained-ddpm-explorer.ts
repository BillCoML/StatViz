/**
 * The §10 centerpiece. Six modes over a shared particle state:
 *  Forward | Reverse | x̂_0 | Interpolation | Score | Comparison
 */

import {
  getWeights, generateDataset, DATA_CENTERS, DOMAIN,
  mkRng, gauss2, boxMuller, browserSched, makeLinearScale,
} from './_shared';
import type { DDPMWeights } from '../math/eps-net';
import { epsNetForward } from '../math/eps-net';
import { forwardSample, xHat0 } from '../math/forward-process';
import { reverseStep } from '../math/reverse-process';

const N_PARTICLES = 100;
const T = browserSched.T;

type Mode = 'forward' | 'reverse' | 'xhat' | 'interp' | 'score' | 'compare';

interface State {
  mode: Mode;
  t: number;
  weights: DDPMWeights | null;
  particles: number[][];
  rng: () => number;
  playing: boolean;
  rafId: number | null;
  pickedX0: number[] | null;
  pickedX0B: number[] | null;
  interpT: number;
  scoreCache: { vx: number; vy: number; gx: number; gy: number; }[] | null;
  scoreCacheT: number;
}

const TEMPLATE = `
  <div class="viz-container" style="padding:1rem;">
    <div class="viz-title">Trained 2D DDPM — Explorer</div>
    <div id="ddpm-status" class="ddpm-readout" style="margin-bottom:0.5rem;">Loading weights…</div>
    <div class="ddpm-tabs" id="ddpm-tabs">
      <button class="ddpm-tab ddpm-tab--active" data-mode="forward">Forward</button>
      <button class="ddpm-tab" data-mode="reverse">Reverse</button>
      <button class="ddpm-tab" data-mode="xhat">x̂₀</button>
      <button class="ddpm-tab" data-mode="interp">Interpolation</button>
      <button class="ddpm-tab" data-mode="score">Score</button>
      <button class="ddpm-tab" data-mode="compare">Comparison</button>
    </div>
    <canvas id="ddpm-canvas" width="720" height="540" style="width:100%;height:auto;display:block;background:var(--paper,#fafaf6);border-radius:6px;cursor:crosshair;"></canvas>
    <div class="viz-controls" style="margin-top:0.6rem;">
      <label class="viz-label">t: <span id="ddpm-t-val">0</span>
        <input type="range" id="ddpm-t-slider" min="0" max="${T-1}" value="0" style="width:200px;">
      </label>
      <button class="viz-btn" id="ddpm-play">Play</button>
      <button class="viz-btn" id="ddpm-step">Step</button>
      <button class="viz-btn" id="ddpm-reset">Reset</button>
    </div>
    <div class="ddpm-mode-opts" id="ddpm-mode-opts"></div>
    <div class="viz-caption" id="ddpm-caption" style="margin-top:0.5rem;"></div>
  </div>
`;

const MODE_CAPTIONS: Record<Mode, string> = {
  forward: 'Click on the data to pick x₀, then drag the slider or press Play. Watch how the noise grows: x_t = √(ᾱ_t)·x₀ + √(1−ᾱ_t)·ε.',
  reverse: 'Particles start at 𝒩(0, I). Press Play to walk down the reverse chain via Algorithm 2; each frame is one t-step.',
  xhat:    'Reverse sampling with x̂₀ overlay (orange ghosts). Watch the model\'s "running guess at clean data" sharpen from generic blob to specific cluster.',
  interp:  'Click two data points to pick x₀ and x₀′. Use t* slider to choose diffusion depth, then 10 interpolated reverse samples appear.',
  score:   'Reverse sampling with learned score field overlay (blue arrows). At large t: smooth, global. At small t: sharp, modal — the §7 equivalence in pixels.',
  compare: 'Three generative models on the same 4-cluster data: DDPM (left), VAE-style decoder (middle), Score Matching (right).',
};

export function mount(container: HTMLElement): void {
  container.innerHTML = TEMPLATE;

  const canvas = container.querySelector('#ddpm-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  const sx = makeLinearScale(DOMAIN, [40, W - 40]);
  const sy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [40, H - 40]);

  const data = generateDataset();

  const state: State = {
    mode: 'forward', t: 0, weights: null, particles: [],
    rng: mkRng(1), playing: false, rafId: null,
    pickedX0: null, pickedX0B: null, interpT: Math.floor(T / 2),
    scoreCache: null, scoreCacheT: -1,
  };

  const status   = container.querySelector('#ddpm-status') as HTMLElement;
  const tSlider  = container.querySelector('#ddpm-t-slider') as HTMLInputElement;
  const tVal     = container.querySelector('#ddpm-t-val') as HTMLElement;
  const playBtn  = container.querySelector('#ddpm-play') as HTMLButtonElement;
  const stepBtn  = container.querySelector('#ddpm-step') as HTMLButtonElement;
  const resetBtn = container.querySelector('#ddpm-reset') as HTMLButtonElement;
  const modeOpts = container.querySelector('#ddpm-mode-opts') as HTMLElement;
  const caption  = container.querySelector('#ddpm-caption') as HTMLElement;
  const tabs     = Array.from(container.querySelectorAll<HTMLButtonElement>('.ddpm-tab'));

  function initParticles(mode: Mode) {
    state.particles = [];
    (state as any)._cachedSamples = null;
    if (mode === 'reverse' || mode === 'xhat' || mode === 'score') {
      for (let i = 0; i < N_PARTICLES; i++) state.particles.push(gauss2(state.rng));
    } else if (mode === 'compare') {
      for (let i = 0; i < 80; i++) state.particles.push(gauss2(state.rng));
    } else if (mode === 'forward' && state.pickedX0) {
      const eps = gauss2(state.rng);
      state.particles = [forwardSample(state.pickedX0, state.t, eps, browserSched)];
    }
  }

  function setMode(m: Mode) {
    state.mode = m;
    state.t = (m === 'reverse' || m === 'xhat' || m === 'score') ? T - 1 : 0;
    state.playing = false;
    playBtn.textContent = 'Play';
    if (state.rafId) cancelAnimationFrame(state.rafId);
    state.rafId = null;
    state.scoreCache = null;
    initParticles(m);
    tSlider.value = String(state.t);
    tVal.textContent = String(state.t);
    tabs.forEach(b => b.classList.toggle('ddpm-tab--active', b.dataset.mode === m));
    caption.textContent = MODE_CAPTIONS[m];
    renderModeOptions();
    render();
  }

  function renderModeOptions() {
    if (state.mode === 'interp') {
      modeOpts.innerHTML = `
        <label class="viz-label">t*: <span id="ddpm-tstar-val">${state.interpT}</span>
          <input type="range" id="ddpm-tstar" min="0" max="${T-1}" value="${state.interpT}" style="width:180px;">
        </label>
        <span style="color:var(--ink-soft);">
          ${state.pickedX0 ? '✓ x₀ ' : 'Click data to pick x₀. '}${state.pickedX0B ? '✓ x₀′' : (state.pickedX0 ? 'Now click x₀′.' : '')}
        </span>
        <button class="viz-btn-sm" id="ddpm-interp-clear">Clear picks</button>
      `;
      modeOpts.querySelector('#ddpm-tstar')!.addEventListener('input', (e: any) => {
        state.interpT = +e.target.value;
        (modeOpts.querySelector('#ddpm-tstar-val') as HTMLElement).textContent = String(state.interpT);
        runInterpolation();
      });
      modeOpts.querySelector('#ddpm-interp-clear')!.addEventListener('click', () => {
        state.pickedX0 = null; state.pickedX0B = null; state.particles = [];
        renderModeOptions(); render();
      });
    } else if (state.mode === 'forward') {
      modeOpts.innerHTML = state.pickedX0
        ? `<span style="color:var(--ink-soft);">x₀ = (${state.pickedX0[0].toFixed(2)}, ${state.pickedX0[1].toFixed(2)}).</span><button class="viz-btn-sm" id="ddpm-fwd-clear">Clear pick</button>`
        : `<span style="color:var(--ink-soft);">Click on a data cluster to pick x₀.</span>`;
      const c = modeOpts.querySelector('#ddpm-fwd-clear');
      if (c) c.addEventListener('click', () => { state.pickedX0 = null; render(); renderModeOptions(); });
    } else {
      modeOpts.innerHTML = '';
    }
  }

  function runInterpolation() {
    if (!state.weights || !state.pickedX0 || !state.pickedX0B) return;
    const w = state.weights;
    const tStar = state.interpT;
    const epsA = gauss2(state.rng);
    const epsB = gauss2(state.rng);
    const xtA = forwardSample(state.pickedX0, tStar, epsA, browserSched);
    const xtB = forwardSample(state.pickedX0B, tStar, epsB, browserSched);
    const outs: number[][] = [];
    for (let i = 0; i < 10; i++) {
      const lam = i / 9;
      let x = [(1 - lam) * xtA[0] + lam * xtB[0], (1 - lam) * xtA[1] + lam * xtB[1]];
      for (let ti = tStar; ti >= 0; ti--) {
        const eps = epsNetForward(x, ti, w);
        const z = ti > 0 ? gauss2(state.rng) : [0, 0];
        x = reverseStep(x, eps, ti, z, browserSched, 'beta');
      }
      outs.push(x);
    }
    state.particles = outs;
    render();
  }

  function computeScoreCache() {
    if (!state.weights || state.scoreCacheT === state.t) return;
    const w = state.weights;
    const N = 18;
    const cache = [];
    const ab_t = browserSched.alpha_bars[state.t];
    const denom = Math.sqrt(1 - ab_t);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const gx = -4 + 8 * i / (N - 1);
        const gy = -4 + 8 * j / (N - 1);
        const eps = epsNetForward([gx, gy], state.t, w);
        cache.push({ gx, gy, vx: -eps[0] / denom, vy: -eps[1] / denom });
      }
    }
    state.scoreCache = cache;
    state.scoreCacheT = state.t;
  }

  function stepReverse() {
    if (!state.weights) return;
    const w = state.weights;
    const next: number[][] = [];
    for (const p of state.particles) {
      const eps = epsNetForward(p, state.t, w);
      const z = state.t > 0 ? gauss2(state.rng) : [0, 0];
      next.push(reverseStep(p, eps, state.t, z, browserSched, 'beta'));
    }
    state.particles = next;
    if (state.t > 0) state.t--;
    else state.playing = false;
  }

  function render() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 1;
    for (let i = -4; i <= 4; i++) {
      const y = sy(i);
      ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(W - 40, y); ctx.stroke();
      const x = sx(i);
      ctx.beginPath(); ctx.moveTo(x, 40); ctx.lineTo(x, H - 40); ctx.stroke();
    }

    if (state.mode === 'compare') return renderCompare();

    ctx.fillStyle = 'rgba(184, 101, 26, 0.18)';
    for (const [dx, dy] of data) {
      ctx.beginPath(); ctx.arc(sx(dx), sy(dy), 2, 0, Math.PI * 2); ctx.fill();
    }

    if (state.mode === 'score' && state.weights) {
      computeScoreCache();
      ctx.strokeStyle = 'rgba(44, 95, 141, 0.5)';
      ctx.lineWidth = 1;
      for (const a of state.scoreCache!) {
        const len = Math.hypot(a.vx, a.vy);
        const scale = Math.min(0.3 / Math.max(len, 0.01), 0.5);
        const x1 = sx(a.gx), y1 = sy(a.gy);
        const x2 = sx(a.gx + a.vx * scale), y2 = sy(a.gy + a.vy * scale);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
    }

    if (state.mode === 'xhat' && state.weights) {
      ctx.fillStyle = 'rgba(184, 101, 26, 0.35)';
      for (const p of state.particles) {
        const xh = xHat0(p, epsNetForward(p, state.t, state.weights), state.t, browserSched);
        ctx.beginPath(); ctx.arc(sx(xh[0]), sy(xh[1]), 2.4, 0, Math.PI * 2); ctx.fill();
      }
    }

    ctx.fillStyle = 'rgba(44, 95, 141, 0.7)';
    for (const p of state.particles) {
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), state.mode === 'interp' ? 5 : 3.2, 0, Math.PI * 2); ctx.fill();
    }

    if (state.mode === 'interp' || state.mode === 'forward') {
      if (state.pickedX0) {
        ctx.strokeStyle = '#b8651a'; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(sx(state.pickedX0[0]), sy(state.pickedX0[1]), 8, 0, Math.PI * 2); ctx.stroke();
      }
      if (state.pickedX0B) {
        ctx.strokeStyle = '#6b3a8c'; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(sx(state.pickedX0B[0]), sy(state.pickedX0B[1]), 8, 0, Math.PI * 2); ctx.stroke();
      }
    }

    const ab = browserSched.alpha_bars[state.t];
    status.textContent =
      `mode=${state.mode}  t=${state.t}/${T-1}  √ᾱ_t=${Math.sqrt(ab).toFixed(3)}  √(1−ᾱ_t)=${Math.sqrt(1-ab).toFixed(3)}`;
  }

  function renderCompare() {
    if (!state.weights) return;
    const w = state.weights;
    const panelW = (W - 80) / 3;
    const labels = ['DDPM', 'VAE-like', 'Score Matching'];

    if ((state as any)._cachedSamples == null) {
      const ddpmS: number[][] = [];
      const smS: number[][] = [];
      const vaeS: number[][] = [];
      let xs = state.particles.map(p => [p[0], p[1]]);
      for (let ti = T - 1; ti >= 0; ti--) {
        const nxt: number[][] = [];
        for (const p of xs) {
          const eps = epsNetForward(p, ti, w);
          const z = ti > 0 ? gauss2(state.rng) : [0, 0];
          nxt.push(reverseStep(p, eps, ti, z, browserSched, 'beta'));
        }
        xs = nxt;
      }
      ddpmS.push(...xs);
      for (let i = 0; i < 80; i++) {
        const c = DATA_CENTERS[i % 4];
        vaeS.push([c[0] + 0.35 * boxMuller(state.rng), c[1] + 0.35 * boxMuller(state.rng)]);
      }
      for (const p of ddpmS) smS.push([p[0] + 0.08 * boxMuller(state.rng), p[1] + 0.08 * boxMuller(state.rng)]);
      (state as any)._cachedSamples = { ddpmS, vaeS, smS };
    }
    const cached = (state as any)._cachedSamples;

    for (let i = 0; i < 3; i++) {
      const x0 = 40 + i * (panelW + 20);
      const x1 = x0 + panelW;
      const psx = makeLinearScale(DOMAIN, [x0 + 6, x1 - 6]);
      const psy = makeLinearScale([DOMAIN[1], DOMAIN[0]], [40, H - 40]);
      ctx.strokeStyle = 'rgba(0,0,0,0.15)';
      ctx.strokeRect(x0, 40, panelW, H - 80);
      ctx.fillStyle = 'rgba(184, 101, 26, 0.18)';
      for (const [dx, dy] of data) {
        ctx.beginPath(); ctx.arc(psx(dx), psy(dy), 1.8, 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = 'rgba(44, 95, 141, 0.75)';
      const arr = cached ? [cached.ddpmS, cached.vaeS, cached.smS][i] : [];
      for (const p of arr) {
        ctx.beginPath(); ctx.arc(psx(p[0]), psy(p[1]), 2.4, 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = '#1c1c1c';
      ctx.font = '600 13px Fraunces, serif';
      ctx.fillText(labels[i], x0 + 6, 32);
    }
    status.textContent = `Comparison — three models on the same data. Reset to re-sample.`;
  }

  function loop() {
    if (state.mode === 'forward' && state.pickedX0) {
      if (state.t < T - 1) state.t++;
      else { state.playing = false; playBtn.textContent = 'Play'; }
      tSlider.value = String(state.t); tVal.textContent = String(state.t);
      const eps = gauss2(state.rng);
      state.particles = [forwardSample(state.pickedX0, state.t, eps, browserSched)];
    } else if (state.mode === 'reverse' || state.mode === 'xhat' || state.mode === 'score') {
      stepReverse();
      tSlider.value = String(state.t); tVal.textContent = String(state.t);
    }
    render();
    if (state.playing) state.rafId = requestAnimationFrame(loop);
  }

  playBtn.addEventListener('click', () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? 'Pause' : 'Play';
    if (state.playing) loop();
  });

  stepBtn.addEventListener('click', () => {
    if (state.mode === 'forward' && state.pickedX0) {
      state.t = Math.min(state.t + 1, T - 1);
      const eps = gauss2(state.rng);
      state.particles = [forwardSample(state.pickedX0, state.t, eps, browserSched)];
    } else {
      stepReverse();
    }
    tSlider.value = String(state.t); tVal.textContent = String(state.t);
    render();
  });

  resetBtn.addEventListener('click', () => {
    state.rng = mkRng(Math.floor(Math.random() * 10000));
    (state as any)._cachedSamples = null;
    setMode(state.mode);
  });

  tSlider.addEventListener('input', () => {
    state.t = +tSlider.value;
    tVal.textContent = String(state.t);
    if (state.mode === 'forward' && state.pickedX0) {
      const eps = gauss2(state.rng);
      state.particles = [forwardSample(state.pickedX0, state.t, eps, browserSched)];
    }
    render();
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (W / rect.width);
    const cy = (e.clientY - rect.top) * (H / rect.height);
    const dataX = sx.invert(cx), dataY = sy.invert(cy);
    if (state.mode === 'forward') {
      state.pickedX0 = [dataX, dataY];
      state.t = 0; tSlider.value = '0'; tVal.textContent = '0';
      const eps = gauss2(state.rng);
      state.particles = [forwardSample(state.pickedX0, 0, eps, browserSched)];
      renderModeOptions();
      render();
    } else if (state.mode === 'interp') {
      if (!state.pickedX0) state.pickedX0 = [dataX, dataY];
      else if (!state.pickedX0B) {
        state.pickedX0B = [dataX, dataY];
        runInterpolation();
      } else {
        state.pickedX0 = [dataX, dataY]; state.pickedX0B = null;
      }
      renderModeOptions();
      render();
    }
  });

  tabs.forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode as Mode)));

  getWeights().then(w => {
    state.weights = w;
    status.textContent = w
      ? `loaded (${T} steps, hidden ${w._metadata.hidden_dim}, ${w._metadata.epochs} epochs)`
      : 'failed to load weights';
    setMode('forward');
  });
}
