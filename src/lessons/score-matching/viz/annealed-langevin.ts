import * as d3 from 'd3';
import { scoreNetForward, loadScoreWeights } from '../math/score-model';
import type { ScoreNetWeights } from '../math/score-model';

// ── Constants ─────────────────────────────────────────────────────────────────

const DOMAIN: [number, number] = [-4.5, 4.5];
const N_PARTICLES = 100;
const T_INNER = 10;           // Langevin steps per sigma level
const EPSILON_BASE = 5e-6;    // step size: alpha_ell = epsilon * (sigma_ell / sigma_L)^2
const ARROW_GRID = 16;        // score overlay resolution (16×16 grid)
const TRAIL_LEN = 8;          // particle trail length

// Seeded LCG RNG for reproducibility
function mkRng(seed: number): () => number {
  let s = seed;
  return () => { s = (Math.imul(1664525, s) + 1013904223) | 0; return (s >>> 0) / 0x100000000; };
}

function boxMuller(rng: () => number): number {
  let u = rng(); while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ── State ─────────────────────────────────────────────────────────────────────

interface State {
  weights:    ScoreNetWeights | null;
  sigmas:     number[];
  sigmaIdx:   number;
  innerStep:  number;
  particles:  Float32Array;       // N_PARTICLES * 2, interleaved
  trails:     Float32Array[];     // one per particle, ring buffer
  trailHead:  Uint8Array;         // current write position per particle
  arrowCache: { x: number; y: number; vx: number; vy: number }[] | null;
  arrowSigmaIdx: number;          // which sigma the cache was computed at
  playing:    boolean;
  showField:  boolean;
  showTrails: boolean;
  scheduleMode: 'short' | 'standard' | 'long';
  rng:        () => number;
  rafId:      number | null;
}

// ── HTML template ─────────────────────────────────────────────────────────────

const TEMPLATE = `
  <div class="viz-container al-container">
    <div class="viz-title">Annealed Langevin — Learned Score</div>
    <div id="al-status" class="al-status">Loading weights…</div>
    <svg id="al-svg" style="width:100%;height:460px;display:block;"></svg>

    <div class="al-schedule-row">
      <span class="al-schedule-label">σ schedule:</span>
      <div id="al-dots" class="al-dots"></div>
    </div>

    <div class="viz-controls al-controls">
      <button class="viz-btn" id="al-play">Play</button>
      <button class="viz-btn" id="al-step">Step</button>
      <button class="viz-btn" id="al-reset">Reset</button>
      <label class="viz-label" style="margin-left:auto;">
        <input type="checkbox" id="al-show-field"> Show score field
      </label>
      <label class="viz-label">
        <input type="checkbox" id="al-show-trails" checked> Show trails
      </label>
    </div>
    <div class="viz-controls">
      <span class="viz-label">Schedule:</span>
      <label class="viz-label"><input type="radio" name="al-sched" value="short"> Short (5 levels)</label>
      <label class="viz-label"><input type="radio" name="al-sched" value="standard" checked> Standard (10)</label>
      <label class="viz-label"><input type="radio" name="al-sched" value="long"> Long (20 levels)</label>
    </div>
    <div class="viz-caption">
      Particles start at N(0, σ<sub>max</sub>²·I). Each level runs ${T_INNER} Langevin steps
      at the current σ, then σ decreases. The score-field overlay shows the learned score
      morphing from global (large σ) to local/modal (small σ).
    </div>
  </div>
`;

// ── Data overlay (training set) ───────────────────────────────────────────────

const DATA_CENTERS = [[2, 2], [2, -2], [-2, 2], [-2, -2]];
const DATA_STD = 0.2;
const DATA_SEED = 99;
const DATA_N_PER = 50;

function generateData(): [number, number][] {
  const rng = mkRng(DATA_SEED);
  const pts: [number, number][] = [];
  for (const c of DATA_CENTERS) {
    for (let i = 0; i < DATA_N_PER; i++) {
      pts.push([c[0] + DATA_STD * boxMuller(rng), c[1] + DATA_STD * boxMuller(rng)]);
    }
  }
  return pts;
}

const DATA_PTS = generateData();

// ── Sigma schedules ───────────────────────────────────────────────────────────

function makeSchedule(mode: 'short' | 'standard' | 'long', baseSignals: number[]): number[] {
  const L = mode === 'short' ? 5 : mode === 'long' ? 20 : 10;
  const sigMax = baseSignals[0];
  const sigMin = baseSignals[baseSignals.length - 1];
  return Array.from({ length: L }, (_, i) =>
    Math.exp(Math.log(sigMax) + i * (Math.log(sigMin) - Math.log(sigMax)) / (L - 1))
  );
}

// ── Score overlay computation ─────────────────────────────────────────────────

function computeArrows(
  sigmaIdx: number, sigmas: number[], weights: ScoreNetWeights,
  xS: d3.ScaleLinear<number, number>,
  yS: d3.ScaleLinear<number, number>,
): { x: number; y: number; vx: number; vy: number }[] {
  const sigma = sigmas[sigmaIdx];
  const step  = (DOMAIN[1] - DOMAIN[0]) / ARROW_GRID;
  const arrows: { x: number; y: number; vx: number; vy: number }[] = [];
  let maxLen = 0;

  for (let i = 0; i <= ARROW_GRID; i++) {
    for (let j = 0; j <= ARROW_GRID; j++) {
      const gx = DOMAIN[0] + i * step;
      const gy = DOMAIN[0] + j * step;
      const s  = scoreNetForward([gx, gy], sigma, weights);
      const len = Math.hypot(s[0], s[1]);
      if (len > maxLen) maxLen = len;
      arrows.push({ x: xS(gx), y: yS(gy), vx: s[0], vy: s[1] });
    }
  }

  const W = xS.range()[1] - xS.range()[0];
  const pixScale = maxLen > 0 ? (W * 0.04) / maxLen : 1;
  return arrows.map(a => ({ ...a, vx: a.vx * pixScale, vy: a.vy * pixScale }));
}

// ── Particle init ─────────────────────────────────────────────────────────────

function initParticles(sigmaMax: number, rng: () => number): Float32Array {
  const pts = new Float32Array(N_PARTICLES * 2);
  for (let i = 0; i < N_PARTICLES * 2; i++) pts[i] = sigmaMax * boxMuller(rng);
  return pts;
}

function initTrails(): { trails: Float32Array[]; heads: Uint8Array } {
  const trails = Array.from({ length: N_PARTICLES }, () => new Float32Array(TRAIL_LEN * 2).fill(NaN));
  const heads  = new Uint8Array(N_PARTICLES);
  return { trails, heads };
}

// ── Main Langevin step for all particles ──────────────────────────────────────

function stepAllParticles(state: State): void {
  const sigma   = state.sigmas[state.sigmaIdx];
  const sigmaL  = state.sigmas[state.sigmas.length - 1];
  const alpha   = EPSILON_BASE * (sigma / sigmaL) ** 2;
  const sqrtA   = Math.sqrt(alpha);

  if (!state.weights) return;

  for (let i = 0; i < N_PARTICLES; i++) {
    const px = state.particles[i * 2];
    const py = state.particles[i * 2 + 1];
    const s  = scoreNetForward([px, py], sigma, state.weights);
    const nx = px + (alpha / 2) * s[0] + sqrtA * boxMuller(state.rng);
    const ny = py + (alpha / 2) * s[1] + sqrtA * boxMuller(state.rng);
    state.particles[i * 2]     = nx;
    state.particles[i * 2 + 1] = ny;

    // Trail ring buffer
    const hi = (state.trailHead[i] + 1) % TRAIL_LEN;
    state.trails[i][hi * 2]     = nx;
    state.trails[i][hi * 2 + 1] = ny;
    state.trailHead[i] = hi;
  }

  state.innerStep++;
  if (state.innerStep >= T_INNER) {
    state.innerStep = 0;
    if (state.sigmaIdx < state.sigmas.length - 1) {
      state.sigmaIdx++;
      // Invalidate arrow cache when sigma changes
      if (state.arrowSigmaIdx !== state.sigmaIdx) state.arrowCache = null;
    } else {
      state.playing = false;
    }
  }
}

// ── SVG render ────────────────────────────────────────────────────────────────

function drawFrame(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  W: number, H: number,
  xS: d3.ScaleLinear<number, number>,
  yS: d3.ScaleLinear<number, number>,
  state: State,
): void {
  svg.selectAll('*').remove();

  // Training data (background, faded)
  svg.selectAll('.al-data')
    .data(DATA_PTS).join('circle').attr('class', 'al-data')
    .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
    .attr('r', 2).attr('fill', '#b8651a').attr('fill-opacity', 0.18);

  // Score field overlay
  if (state.showField && state.arrowCache) {
    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'al-arrow-head')
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('refX', 4).attr('refY', 2.5).attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L0,5 L5,2.5 z')
      .attr('fill', '#2c5f8d').attr('opacity', 0.45);

    svg.selectAll('.al-field-arrow')
      .data(state.arrowCache).join('line').attr('class', 'al-field-arrow')
      .attr('x1', d => d.x).attr('y1', d => d.y)
      .attr('x2', d => d.x + d.vx).attr('y2', d => d.y - d.vy)
      .attr('stroke', '#2c5f8d').attr('stroke-width', 1.0).attr('stroke-opacity', 0.4)
      .attr('marker-end', 'url(#al-arrow-head)');
  }

  // Particle trails
  if (state.showTrails) {
    for (let i = 0; i < N_PARTICLES; i++) {
      const trail = state.trails[i];
      const head  = state.trailHead[i];
      for (let t = 1; t < TRAIL_LEN; t++) {
        const j0 = ((head - t + 1 + TRAIL_LEN) % TRAIL_LEN);
        const j1 = ((head - t + 2 + TRAIL_LEN) % TRAIL_LEN);
        const x0 = trail[j0 * 2], y0 = trail[j0 * 2 + 1];
        const x1 = trail[j1 * 2], y1 = trail[j1 * 2 + 1];
        if (!isNaN(x0) && !isNaN(x1)) {
          svg.append('line')
            .attr('x1', xS(x0)).attr('y1', yS(y0))
            .attr('x2', xS(x1)).attr('y2', yS(y1))
            .attr('stroke', '#5a8a6a')
            .attr('stroke-width', 0.8)
            .attr('stroke-opacity', (1 - t / TRAIL_LEN) * 0.6);
        }
      }
    }
  }

  // Particles
  const ptData: [number, number][] = Array.from({ length: N_PARTICLES }, (_, i) =>
    [state.particles[i * 2], state.particles[i * 2 + 1]]
  );
  svg.selectAll('.al-particle')
    .data(ptData).join('circle').attr('class', 'al-particle')
    .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
    .attr('r', 2.5).attr('fill', '#5a8a6a').attr('fill-opacity', 0.75);

  // Axes
  svg.append('g').attr('transform', `translate(0,${H-25})`).call(d3.axisBottom(xS).ticks(5))
     .selectAll('text').attr('font-size', '0.7rem');
  svg.append('g').attr('transform', 'translate(38,0)').call(d3.axisLeft(yS).ticks(5))
     .selectAll('text').attr('font-size', '0.7rem');

  // Current sigma label
  if (state.sigmas.length > 0) {
    const sigma = state.sigmas[state.sigmaIdx];
    svg.append('text').attr('x', W - 10).attr('y', 20).attr('text-anchor', 'end')
      .attr('font-family', 'var(--font-mono)').attr('font-size', '0.82rem')
      .attr('fill', '#6b3a8c')
      .text(`σ = ${sigma.toFixed(4)}`);
  }
}

function updateDots(
  dotsEl: HTMLElement, state: State,
): void {
  const L = state.sigmas.length;
  dotsEl.innerHTML = '';
  for (let i = 0; i < L; i++) {
    const dot = document.createElement('div');
    dot.className = 'al-dot';
    if (i === state.sigmaIdx) dot.classList.add('al-dot--active');
    if (i < state.sigmaIdx)  dot.classList.add('al-dot--done');
    dotsEl.appendChild(dot);
  }
}

// ── Mount ─────────────────────────────────────────────────────────────────────

export function mount(outerContainer: HTMLElement): void {
  outerContainer.innerHTML = TEMPLATE;

  const svgEl    = outerContainer.querySelector<SVGSVGElement>('#al-svg')!;
  const statusEl = outerContainer.querySelector<HTMLElement>('#al-status')!;
  const dotsEl   = outerContainer.querySelector<HTMLElement>('#al-dots')!;
  const playBtn  = outerContainer.querySelector<HTMLButtonElement>('#al-play')!;
  const stepBtn  = outerContainer.querySelector<HTMLButtonElement>('#al-step')!;
  const resetBtn = outerContainer.querySelector<HTMLButtonElement>('#al-reset')!;
  const fieldCb  = outerContainer.querySelector<HTMLInputElement>('#al-show-field')!;
  const trailsCb = outerContainer.querySelector<HTMLInputElement>('#al-show-trails')!;
  const schedInputs = outerContainer.querySelectorAll<HTMLInputElement>('input[name="al-sched"]');

  const svg = d3.select(svgEl);
  const W   = Math.max(svgEl.clientWidth || 600, 400);
  const H   = 460;
  const xS  = d3.scaleLinear(DOMAIN, [38, W - 15]);
  const yS  = d3.scaleLinear(DOMAIN, [H - 25, 15]);

  const state: State = {
    weights:       null,
    sigmas:        [],
    sigmaIdx:      0,
    innerStep:     0,
    particles:     new Float32Array(N_PARTICLES * 2),
    trails:        [],
    trailHead:     new Uint8Array(N_PARTICLES),
    arrowCache:    null,
    arrowSigmaIdx: -1,
    playing:       false,
    showField:     false,
    showTrails:    true,
    scheduleMode:  'standard',
    rng:           mkRng(42),
    rafId:         null,
  };

  function resetSim() {
    state.sigmaIdx  = 0;
    state.innerStep = 0;
    state.rng       = mkRng(42);
    if (state.sigmas.length > 0) {
      state.particles = initParticles(state.sigmas[0], state.rng);
    }
    const { trails, heads } = initTrails();
    state.trails    = trails;
    state.trailHead = heads;
    state.arrowCache = null;
    state.arrowSigmaIdx = -1;
    playBtn.textContent = 'Play';
    state.playing = false;
    drawFrame(svg, W, H, xS, yS, state);
    updateDots(dotsEl, state);
  }

  function tick() {
    if (!state.playing || !state.weights) return;
    stepAllParticles(state);

    // Compute (or reuse) score field overlay
    if (state.showField && (state.arrowCache === null || state.arrowSigmaIdx !== state.sigmaIdx)) {
      state.arrowCache    = computeArrows(state.sigmaIdx, state.sigmas, state.weights, xS, yS);
      state.arrowSigmaIdx = state.sigmaIdx;
    }

    drawFrame(svg, W, H, xS, yS, state);
    updateDots(dotsEl, state);

    if (!state.playing) {
      playBtn.textContent = 'Play';
      return;
    }
    state.rafId = requestAnimationFrame(tick);
  }

  playBtn.addEventListener('click', () => {
    if (!state.weights) return;
    if (state.playing) {
      state.playing = false;
      playBtn.textContent = 'Play';
      if (state.rafId) cancelAnimationFrame(state.rafId);
    } else {
      state.playing = true;
      playBtn.textContent = 'Pause';
      state.rafId = requestAnimationFrame(tick);
    }
  });

  stepBtn.addEventListener('click', () => {
    if (!state.weights) return;
    state.playing = false;
    playBtn.textContent = 'Play';
    stepAllParticles(state);
    if (state.showField && (state.arrowCache === null || state.arrowSigmaIdx !== state.sigmaIdx)) {
      state.arrowCache    = computeArrows(state.sigmaIdx, state.sigmas, state.weights, xS, yS);
      state.arrowSigmaIdx = state.sigmaIdx;
    }
    drawFrame(svg, W, H, xS, yS, state);
    updateDots(dotsEl, state);
  });

  resetBtn.addEventListener('click', () => {
    state.playing = false;
    playBtn.textContent = 'Play';
    if (state.rafId) cancelAnimationFrame(state.rafId);
    resetSim();
  });

  fieldCb.addEventListener('change', () => {
    state.showField = fieldCb.checked;
    if (state.showField && state.weights && state.sigmas.length > 0) {
      state.arrowCache    = computeArrows(state.sigmaIdx, state.sigmas, state.weights, xS, yS);
      state.arrowSigmaIdx = state.sigmaIdx;
    }
    drawFrame(svg, W, H, xS, yS, state);
  });

  trailsCb.addEventListener('change', () => {
    state.showTrails = trailsCb.checked;
    drawFrame(svg, W, H, xS, yS, state);
  });

  schedInputs.forEach(inp => {
    inp.addEventListener('change', () => {
      state.scheduleMode = inp.value as State['scheduleMode'];
      if (state.weights) {
        const baseSigmas = state.weights._metadata.sigmas;
        state.sigmas = makeSchedule(state.scheduleMode, baseSigmas);
      }
      resetSim();
    });
  });

  // Load weights
  const weightsUrl = new URL('../assets/score-weights.json', import.meta.url).href;
  statusEl.textContent = 'Loading weights…';

  loadScoreWeights(weightsUrl).then(weights => {
    if (!weights) {
      statusEl.textContent = 'Failed to load weights. Ensure score-weights.json is built.';
      return;
    }
    state.weights = weights;
    const baseSigmas = weights._metadata.sigmas;
    state.sigmas = makeSchedule(state.scheduleMode, baseSigmas);
    statusEl.textContent = `Ready — ${state.sigmas.length} sigma levels, ${N_PARTICLES} particles.`;
    statusEl.style.color = 'var(--sage, #5a8a6a)';

    playBtn.disabled   = false;
    stepBtn.disabled   = false;
    resetBtn.disabled  = false;
    fieldCb.disabled   = false;
    trailsCb.disabled  = false;

    resetSim();
  });

  // Disable controls until weights loaded
  [playBtn, stepBtn, fieldCb].forEach(el => { (el as HTMLButtonElement | HTMLInputElement).disabled = true; });
}
