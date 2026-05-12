import * as d3 from 'd3';
import { encoderForward, decoderForward, reparameterize } from '../math/encoder-decoder';
import type { MLPWeights } from '../math/encoder-decoder';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ModelWeights {
  enc: { W1: number[][]; b1: number[]; W2: number[][]; b2: number[] };
  dec: { W1: number[][]; b1: number[]; W2: number[][]; b2: number[] };
}

interface WeightStore {
  models: Record<string, ModelWeights>;
  metadata: { centers: number[][]; n_per_cluster: number };
}

// ─── Weight loading & conversion ─────────────────────────────────────────────

function convertModel(raw: Record<string, unknown>): ModelWeights {
  const enc1W   = raw['enc1.weight'] as number[][];
  const enc1b   = raw['enc1.bias'] as number[];
  const encMuW  = raw['enc_mu.weight'] as number[][];
  const encMub  = raw['enc_mu.bias'] as number[];
  const encLsW  = raw['enc_logsigma.weight'] as number[][];
  const encLsb  = raw['enc_logsigma.bias'] as number[];
  const dec1W   = raw['dec1.weight'] as number[][];
  const dec1b   = raw['dec1.bias'] as number[];
  const dec2W   = raw['dec2.weight'] as number[][];
  const dec2b   = raw['dec2.bias'] as number[];
  return {
    enc: {
      W1: enc1W, b1: enc1b,
      W2: [...encMuW, ...encLsW],
      b2: [...encMub, ...encLsb],
    },
    dec: { W1: dec1W, b1: dec1b, W2: dec2W, b2: dec2b },
  };
}

// ─── Dataset (same generation as training) ───────────────────────────────────

function generateDataset(centers: number[][], nPer: number, std = 0.2, rngSeed = 42): { X: [number,number][]; labels: number[] } {
  let s = rngSeed;
  const rng = () => { s = (Math.imul(1664525, s) + 1013904223) | 0; return (s >>> 0) / 0x100000000; };
  const rn  = () => { let u; do { u = rng(); } while (u === 0); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*rng()); };
  const X: [number,number][] = [], labels: number[] = [];
  for (let ci = 0; ci < centers.length; ci++)
    for (let i = 0; i < nPer; i++) {
      X.push([centers[ci][0] + std * rn(), centers[ci][1] + std * rn()]);
      labels.push(ci);
    }
  return { X, labels };
}

// ─── Module state ─────────────────────────────────────────────────────────────

let store: WeightStore | null = null;
let activeBeta = 1.0;

const BETAS    = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
const BETA_KEY = (b: number) => `beta_${b}`;
const CLUSTER_COLORS = ['#c0392b','#2c5f8d','#5a8a6a','#d4a437'];

// ─── SVG state ────────────────────────────────────────────────────────────────

interface PlotState {
  svgData:   SVGSVGElement | null;
  svgLatent: SVGSVGElement | null;
  dataPts:   [number,number][];
  dataLabels: number[];
  latentMus: [number,number][];   // encoder means for each data point
  latentSigs: [number,number][];  // encoder log_sigmas
  hoveredIdx: number | null;
  clickedDataIdxs: number[];
  clickedLatentPts: [number,number][];
  priorSamples: [number,number][];
  decodedPts: [number,number][];
  interpPts: [number,number][];
}

const state: PlotState = {
  svgData: null, svgLatent: null,
  dataPts: [], dataLabels: [],
  latentMus: [], latentSigs: [],
  hoveredIdx: null,
  clickedDataIdxs: [], clickedLatentPts: [],
  priorSamples: [], decodedPts: [], interpPts: [],
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function currentModel(): ModelWeights | null {
  if (!store) return null;
  return store.models[BETA_KEY(activeBeta)] || null;
}

function encode(x: [number,number]): { mu: [number,number]; ls: [number,number] } {
  const m = currentModel()!;
  const { mu, log_sigma } = encoderForward(Array.from(x), m.enc as MLPWeights, 2);
  return { mu: [mu[0], mu[1]], ls: [log_sigma[0], log_sigma[1]] };
}

function decode(z: [number,number]): [number,number] {
  const m = currentModel()!;
  const out = decoderForward(Array.from(z), m.dec as MLPWeights);
  return [out[0], out[1]];
}

function sampleNormal(): number { return Math.sqrt(-2*Math.log(Math.random()+1e-12))*Math.cos(2*Math.PI*Math.random()); }

// ─── Recompute latent encodings ───────────────────────────────────────────────

function recomputeLatents(): void {
  if (!store) return;
  state.latentMus  = state.dataPts.map(x => encode(x).mu);
  state.latentSigs = state.dataPts.map(x => encode(x).ls);
}

// ─── Scale helpers ────────────────────────────────────────────────────────────

function makeScales(pts: [number,number][], padFrac = 0.15) {
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const xExt = d3.extent(xs) as [number,number];
  const yExt = d3.extent(ys) as [number,number];
  const xPad = (xExt[1] - xExt[0]) * padFrac || 0.5;
  const yPad = (yExt[1] - yExt[0]) * padFrac || 0.5;
  return { xExt: [xExt[0]-xPad, xExt[1]+xPad] as [number,number], yExt: [yExt[0]-yPad, yExt[1]+yPad] as [number,number] };
}

// ─── Rendering ────────────────────────────────────────────────────────────────

function renderData(container: HTMLElement): void {
  if (!state.svgData) return;
  const W = state.svgData.clientWidth || 300;
  const H = state.svgData.clientHeight || 300;
  const m = {t:16,r:12,b:24,l:32};
  const iW = W - m.l - m.r, iH = H - m.t - m.b;

  state.svgData.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = d3.select(state.svgData);
  svg.selectAll('*').remove();
  const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

  const { xExt, yExt } = makeScales(state.dataPts);
  const xS = d3.scaleLinear().domain(xExt).range([0, iW]);
  const yS = d3.scaleLinear().domain(yExt).range([iH, 0]);

  g.append('g').attr('class','axis').attr('transform',`translate(0,${iH})`).call(d3.axisBottom(xS).ticks(4) as any);
  g.append('g').attr('class','axis').call(d3.axisLeft(yS).ticks(4) as any);

  // Training data — wrap with index so D3 v7 (event, d) callbacks can find original idx
  const indexedPts = state.dataPts.map((p, i) => ({ p, i }));
  g.selectAll('.dp').data(indexedPts).enter().append('circle')
    .attr('class','dp')
    .attr('cx', d => xS(d.p[0])).attr('cy', d => yS(d.p[1]))
    .attr('r', 3)
    .attr('fill', d => CLUSTER_COLORS[state.dataLabels[d.i]])
    .attr('opacity', d => state.hoveredIdx === d.i ? 1 : 0.55)
    .attr('stroke', d => state.clickedDataIdxs.includes(d.i) ? '#fff' : 'none')
    .attr('stroke-width', 1.5)
    .style('cursor', 'pointer')
    .on('mouseover', (_event, d) => { state.hoveredIdx = d.i; renderLatent(container); })
    .on('mouseout',  ()          => { state.hoveredIdx = null; renderLatent(container); })
    .on('click',     (_event, d) => {
      const idx = state.clickedDataIdxs.indexOf(d.i);
      if (idx >= 0) state.clickedDataIdxs.splice(idx, 1); else state.clickedDataIdxs.push(d.i);
      renderData(container); renderLatent(container);
    });

  // Decoded data points (from latent clicks or prior samples)
  if (state.decodedPts.length > 0) {
    g.selectAll('.decoded').data(state.decodedPts).enter().append('circle')
      .attr('class','decoded')
      .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
      .attr('r', 5)
      .attr('fill', 'var(--decoder)')
      .attr('opacity', 0.8)
      .attr('stroke', 'var(--paper)').attr('stroke-width', 1);
  }

  // Prior samples (in blue)
  if (state.priorSamples.length > 0) {
    g.selectAll('.prior').data(state.priorSamples).enter().append('circle')
      .attr('class','prior')
      .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
      .attr('r', 5)
      .attr('fill', 'var(--prior)')
      .attr('opacity', 0.75)
      .attr('stroke', 'var(--paper)').attr('stroke-width', 1);
  }

  // Interpolation path
  if (state.interpPts.length > 0) {
    const line = d3.line<[number,number]>().x(d => xS(d[0])).y(d => yS(d[1])).curve(d3.curveCatmullRom.alpha(0.5));
    g.append('path').datum(state.interpPts)
      .attr('fill','none').attr('stroke','var(--latent-z)').attr('stroke-width',1.5)
      .attr('stroke-dasharray','4,2').attr('d', line(state.interpPts)!);
    g.selectAll('.interp').data(state.interpPts).enter().append('circle')
      .attr('class','interp')
      .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
      .attr('r', 4).attr('fill','var(--latent-z)').attr('opacity',0.7);
  }
}

function renderLatent(container: HTMLElement): void {
  if (!state.svgLatent || state.latentMus.length === 0) return;
  const W = state.svgLatent.clientWidth || 300;
  const H = state.svgLatent.clientHeight || 300;
  const m = {t:16,r:12,b:24,l:32};
  const iW = W - m.l - m.r, iH = H - m.t - m.b;

  state.svgLatent.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = d3.select(state.svgLatent);
  svg.selectAll('*').remove();
  const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

  const { xExt, yExt } = makeScales(state.latentMus, 0.25);
  const xS = d3.scaleLinear().domain(xExt).range([0, iW]);
  const yS = d3.scaleLinear().domain(yExt).range([iH, 0]);

  g.append('g').attr('class','axis').attr('transform',`translate(0,${iH})`).call(d3.axisBottom(xS).ticks(4) as any);
  g.append('g').attr('class','axis').call(d3.axisLeft(yS).ticks(4) as any);

  // Origin crosshair (prior center)
  g.append('line').attr('x1',xS(-0.5)).attr('x2',xS(0.5)).attr('y1',yS(0)).attr('y2',yS(0))
    .attr('stroke','var(--prior)').attr('stroke-width',0.5).attr('stroke-dasharray','3,3').attr('opacity',0.4);
  g.append('line').attr('x1',xS(0)).attr('x2',xS(0)).attr('y1',yS(-0.5)).attr('y2',yS(0.5))
    .attr('stroke','var(--prior)').attr('stroke-width',0.5).attr('stroke-dasharray','3,3').attr('opacity',0.4);

  // Encoder means
  const indexedMus = state.latentMus.map((p, i) => ({ p, i }));
  g.selectAll('.lp').data(indexedMus).enter().append('circle')
    .attr('class','lp')
    .attr('cx', d => xS(d.p[0])).attr('cy', d => yS(d.p[1]))
    .attr('r', 2.5)
    .attr('fill', d => CLUSTER_COLORS[state.dataLabels[d.i]])
    .attr('opacity', d => state.hoveredIdx === d.i ? 1 : 0.45)
    .style('cursor', 'pointer')
    .on('click', (_event, d) => {
      state.clickedLatentPts.push(d.p);
      state.decodedPts.push(decode(d.p));
      renderData(container); renderLatent(container);
    });

  // Hover ellipse
  if (state.hoveredIdx !== null) {
    const hIdx = state.hoveredIdx;
    const mu   = state.latentMus[hIdx];
    const ls   = state.latentSigs[hIdx];
    const rx   = Math.abs(xS(mu[0] + Math.exp(ls[0])) - xS(mu[0]));
    const ry   = Math.abs(yS(mu[1] - Math.exp(ls[1])) - yS(mu[1]));
    const col  = CLUSTER_COLORS[state.dataLabels[hIdx]];
    g.append('ellipse')
      .attr('cx', xS(mu[0])).attr('cy', yS(mu[1]))
      .attr('rx', rx).attr('ry', ry)
      .attr('fill', 'none')
      .attr('stroke', col)
      .attr('stroke-width', 1.5).attr('opacity', 0.8);
    g.append('circle').attr('cx',xS(mu[0])).attr('cy',yS(mu[1])).attr('r',4)
      .attr('fill', col).attr('opacity',0.9);
  }

  // Clicked latent points
  g.selectAll('.clk').data(state.clickedLatentPts).enter().append('circle')
    .attr('class','clk')
    .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
    .attr('r', 5).attr('fill','var(--decoder)').attr('opacity',0.9)
    .attr('stroke','var(--paper)').attr('stroke-width',1.5);

  // Interpolation line in latent space
  if (state.clickedLatentPts.length >= 2) {
    const a = state.clickedLatentPts[state.clickedLatentPts.length-2];
    const b = state.clickedLatentPts[state.clickedLatentPts.length-1];
    g.append('line')
      .attr('x1',xS(a[0])).attr('y1',yS(a[1]))
      .attr('x2',xS(b[0])).attr('y2',yS(b[1]))
      .attr('stroke','var(--latent-z)').attr('stroke-width',1.5).attr('stroke-dasharray','4,2');
  }
}

// ─── Public mount function ────────────────────────────────────────────────────

export function mountTrainedVAEExplorer(container: HTMLElement): void {
  container.innerHTML = `
    <div class="vae-explorer" id="vae-explorer-root">
      <div id="vae-loading" style="text-align:center;padding:3rem;color:var(--ink-soft);font-style:italic;">
        Loading model weights…
      </div>
    </div>
  `;

  const weightsUrl = new URL('../assets/vae-weights.json', import.meta.url).href;

  fetch(weightsUrl)
    .then(r => r.json())
    .then((json: Record<string, unknown>) => {
      const meta = json.metadata as { centers: number[][]; n_per_cluster: number };
      const models: Record<string, ModelWeights> = {};
      for (const beta of BETAS) {
        const key = BETA_KEY(beta);
        if (json[key]) models[key] = convertModel(json[key] as Record<string, unknown>);
      }
      store = { models, metadata: meta };

      const { X, labels } = generateDataset(meta.centers, meta.n_per_cluster);
      state.dataPts    = X;
      state.dataLabels = labels;

      buildUI(container);
      recomputeLatents();
      renderData(container);
      renderLatent(container);
    })
    .catch(() => {
      const loadEl = container.querySelector('#vae-loading') as HTMLElement;
      loadEl.textContent = 'Could not load vae-weights.json. Run `node scripts/train-vae.cjs` first.';
    });
}

function buildUI(container: HTMLElement): void {
  const root = container.querySelector('#vae-explorer-root') as HTMLElement;
  root.innerHTML = `
    <div class="vae-explorer__panels">
      <div class="vae-explorer__panel">
        <div class="vae-explorer__panel-title">Data space <span style="font-family:var(--font-mono);font-weight:400;">x ∈ ℝ²</span></div>
        <svg id="svg-data" style="width:100%;aspect-ratio:1;display:block;"></svg>
        <div style="font-size:0.72em;color:var(--ink-soft);margin-top:0.2rem;">
          Hover to encode · Click to select · <span style="color:var(--decoder);">●</span> decoded · <span style="color:var(--prior);">●</span> prior samples
        </div>
      </div>
      <div class="vae-explorer__panel">
        <div class="vae-explorer__panel-title">Latent space <span style="font-family:var(--font-mono);font-weight:400;">z ∈ ℝ²</span></div>
        <svg id="svg-latent" style="width:100%;aspect-ratio:1;display:block;"></svg>
        <div style="font-size:0.72em;color:var(--ink-soft);margin-top:0.2rem;">
          Hover: show posterior ellipse · Click: decode → data space
        </div>
      </div>
    </div>

    <div class="vae-explorer__controls">
      <button class="viz-btn-sm" id="btn-sample-prior">Sample from prior (10)</button>
      <button class="viz-btn-sm" id="btn-reconstruct">Reconstruct selected</button>
      <button class="viz-btn-sm" id="btn-interpolate">Interpolate (last 2 latent)</button>
      <button class="viz-btn-sm" id="btn-clear">Clear</button>
      <label class="viz-label" style="margin-left:auto;display:flex;align-items:center;gap:0.4rem;flex-shrink:0;">
        β
        <input type="range" id="beta-slider" min="0" max="5" step="1" value="2" style="width:100px;">
        <span class="vae-beta-label" id="beta-val">β = 1.0</span>
      </label>
      <div class="vae-status" id="explorer-status">Weights loaded. Ready.</div>
    </div>
  `;

  state.svgData   = root.querySelector('#svg-data') as SVGSVGElement;
  state.svgLatent = root.querySelector('#svg-latent') as SVGSVGElement;

  const betaSlider = root.querySelector('#beta-slider') as HTMLInputElement;
  const betaVal    = root.querySelector('#beta-val') as HTMLElement;
  const status     = root.querySelector('#explorer-status') as HTMLElement;

  betaSlider.addEventListener('input', () => {
    const idx = +betaSlider.value;
    activeBeta = BETAS[idx];
    betaVal.textContent = `β = ${activeBeta}`;
    status.textContent = `Switching to β = ${activeBeta}…`;
    state.clickedLatentPts = [];
    state.decodedPts = [];
    state.priorSamples = [];
    state.interpPts = [];
    requestAnimationFrame(() => {
      recomputeLatents();
      renderData(container);
      renderLatent(container);
      status.textContent = `β = ${activeBeta} loaded.`;
    });
  });

  root.querySelector('#btn-sample-prior')!.addEventListener('click', () => {
    const samples: [number,number][] = Array.from({length:10}, () => {
      const z: [number,number] = [sampleNormal(), sampleNormal()];
      return z;
    });
    const decoded = samples.map(z => decode(z));
    state.priorSamples = decoded;
    state.clickedLatentPts = samples;
    renderData(container);
    renderLatent(container);
    status.textContent = `Sampled 10 from prior, decoded.`;
  });

  root.querySelector('#btn-reconstruct')!.addEventListener('click', () => {
    if (state.clickedDataIdxs.length === 0) { status.textContent = 'Click data points first.'; return; }
    const eps = [sampleNormal(), sampleNormal()];
    state.decodedPts = state.clickedDataIdxs.map(i => {
      const { mu, ls } = encode(state.dataPts[i]);
      const z = reparameterize(Array.from(mu), Array.from(ls), eps);
      return decode([z[0], z[1]]);
    });
    renderData(container);
    status.textContent = `Reconstructed ${state.clickedDataIdxs.length} selected point(s).`;
  });

  root.querySelector('#btn-interpolate')!.addEventListener('click', () => {
    if (state.clickedLatentPts.length < 2) { status.textContent = 'Click at least 2 latent points first.'; return; }
    const a = state.clickedLatentPts[state.clickedLatentPts.length-2];
    const b = state.clickedLatentPts[state.clickedLatentPts.length-1];
    const N = 12;
    state.interpPts = Array.from({length:N}, (_, k) => {
      const t = k / (N-1);
      const z: [number,number] = [a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t];
      return decode(z);
    });
    renderData(container);
    renderLatent(container);
    status.textContent = `Interpolated ${N} points between last 2 latent selections.`;
  });

  root.querySelector('#btn-clear')!.addEventListener('click', () => {
    state.clickedDataIdxs = [];
    state.clickedLatentPts = [];
    state.decodedPts = [];
    state.priorSamples = [];
    state.interpPts = [];
    renderData(container);
    renderLatent(container);
    status.textContent = 'Cleared.';
  });
}
