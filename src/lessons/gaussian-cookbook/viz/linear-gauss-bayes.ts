import * as d3 from 'd3';
import { linearGaussianPosterior } from '../math/linear-gaussian';

const W = 460, H = 380;
const M = { top: 16, right: 16, bottom: 36, left: 44 };
const IW = W - M.left - M.right;
const IH = H - M.top - M.bottom;
const DOMAIN = 4;

function drawEllipse(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  xSc: d3.ScaleLinear<number, number>,
  ySc: d3.ScaleLinear<number, number>,
  mu: number[], Sigma: number[][],
  color: string, id: string,
  levels = 2,
) {
  g.select(`#${id}`).remove();
  const det = Sigma[0][0] * Sigma[1][1] - Sigma[0][1] * Sigma[1][0];
  if (det <= 0) return;
  const eg = g.append('g').attr('id', id);
  const trace = Sigma[0][0] + Sigma[1][1];
  const disc  = Math.sqrt(Math.max(0, (trace / 2) ** 2 - det));
  const l1 = trace / 2 + disc, l2 = trace / 2 - disc;
  let v1: [number, number];
  if (Math.abs(Sigma[0][1]) > 1e-10) {
    const n = Math.hypot(l1 - Sigma[0][0], Sigma[0][1]);
    v1 = [(l1 - Sigma[0][0]) / n, Sigma[0][1] / n];
  } else {
    v1 = [1, 0];
  }
  const scale = IW / (2 * DOMAIN);
  const cx = xSc(mu[0]), cy = ySc(mu[1]);
  for (let k = 1; k <= levels; k++) {
    const rx = k * Math.sqrt(l1) * scale;
    const ry = k * Math.sqrt(l2) * scale;
    const angle = Math.atan2(v1[1], v1[0]) * 180 / Math.PI;
    eg.append('ellipse')
      .attr('cx', cx).attr('cy', cy)
      .attr('rx', rx).attr('ry', ry)
      .attr('transform', `rotate(${angle}, ${cx}, ${cy})`)
      .attr('fill', k === 1 ? color : 'none')
      .attr('fill-opacity', k === 1 ? 0.15 : 0)
      .attr('stroke', color)
      .attr('stroke-width', k === 1 ? 2.5 : 1.5)
      .attr('stroke-opacity', 0.9 - k * 0.25);
  }
  eg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', 5)
    .attr('fill', color).attr('stroke', 'white').attr('stroke-width', 1.5);
}

export function mountLinearGaussBayes(container: HTMLElement): void {
  // Default: spec's worked example
  let mu0 = [0, 0];
  let prior_s = 1.0;       // Sigma_0 = s * I
  let A11 = 1, A12 = 0.5, A21 = 0.3, A22 = 1;
  let sigma_n = 0.1;
  let obs = [1.5, 0.8];

  container.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:0.75rem;">
      <div class="gauss-viz-row">
        <svg id="lgb-svg" style="width:100%;max-width:${W}px;display:block;"></svg>
        <div class="gauss-readout-panel" style="min-width:180px;">
          <div style="font-weight:700;color:var(--gauss-q);font-size:0.88em;">Prior z ~ 𝒩(μ₀, Σ₀)</div>
          <div id="lgb-r-prior" style="color:var(--ink-soft);font-size:0.8em;">—</div>
          <div style="margin-top:0.5rem;font-weight:700;color:var(--ink-soft);font-size:0.88em;">Likelihood locus</div>
          <div style="font-size:0.75em;color:var(--ink-soft);">Az + b = x</div>
          <div style="margin-top:0.5rem;font-weight:700;color:var(--gauss-p);font-size:0.88em;">Posterior z|x</div>
          <div id="lgb-r-post" style="color:var(--ink-soft);font-size:0.8em;">—</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        <div class="gauss-viz-controls">
          <div style="font-size:0.8em;font-weight:700;margin-bottom:0.3rem;">Model</div>
          <label class="viz-label">A₁₁ <input type="range" id="lgb-a11" min="0.1" max="2" step="0.1" value="1"> <span id="lgb-v-a11">1.0</span></label>
          <label class="viz-label">A₁₂ <input type="range" id="lgb-a12" min="-1" max="1" step="0.05" value="0.5"> <span id="lgb-v-a12">0.5</span></label>
          <label class="viz-label">A₂₁ <input type="range" id="lgb-a21" min="-1" max="1" step="0.05" value="0.3"> <span id="lgb-v-a21">0.3</span></label>
          <label class="viz-label">A₂₂ <input type="range" id="lgb-a22" min="0.1" max="2" step="0.1" value="1"> <span id="lgb-v-a22">1.0</span></label>
          <label class="viz-label">σ_n² <input type="range" id="lgb-sn" min="0.01" max="1" step="0.01" value="0.1"> <span id="lgb-v-sn">0.10</span></label>
        </div>
        <div class="gauss-viz-controls">
          <div style="font-size:0.8em;font-weight:700;margin-bottom:0.3rem;">Prior &amp; Observation</div>
          <label class="viz-label">Prior scale Σ₀ = s·I <input type="range" id="lgb-ps" min="0.1" max="3" step="0.1" value="1"> <span id="lgb-v-ps">1.0</span></label>
          <label class="viz-label">x₁ (observed) <input type="range" id="lgb-x1" min="-3" max="3" step="0.1" value="1.5"> <span id="lgb-v-x1">1.5</span></label>
          <label class="viz-label">x₂ (observed) <input type="range" id="lgb-x2" min="-3" max="3" step="0.1" value="0.8"> <span id="lgb-v-x2">0.8</span></label>
        </div>
      </div>
      <div class="gauss-buttons">
        <button class="viz-btn-sm" id="lgb-reset">Reset to §6 example</button>
      </div>
      <div style="display:flex;gap:1.5em;font-size:0.8em;font-family:var(--font-mono);">
        <span><svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke="var(--gauss-q)" stroke-width="2.5"/></svg> prior</span>
        <span><svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke="var(--ink-soft)" stroke-width="2" stroke-dasharray="4,2"/></svg> likelihood locus</span>
        <span><svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke="var(--gauss-p)" stroke-width="2.5"/></svg> posterior</span>
      </div>
    </div>
  `;

  const svgEl = container.querySelector('#lgb-svg') as SVGSVGElement;
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = d3.select(svgEl);
  const gMain = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
  const xSc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([0, IW]);
  const ySc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([IH, 0]);
  gMain.append('g').attr('class', 'axis').attr('transform', `translate(0,${IH})`).call(d3.axisBottom(xSc).ticks(5) as any);
  gMain.append('g').attr('class', 'axis').call(d3.axisLeft(ySc).ticks(5) as any);
  gMain.append('text').attr('class', 'axis-label').attr('x', IW / 2).attr('y', IH + 32).attr('text-anchor', 'middle').text('z₁');
  gMain.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)').attr('x', -IH / 2).attr('y', -36).attr('text-anchor', 'middle').text('z₂');

  const gaussG = gMain.append('g');
  const locusG = gMain.append('g').attr('id', 'lgb-locus');

  const rPrior = container.querySelector('#lgb-r-prior') as HTMLElement;
  const rPost  = container.querySelector('#lgb-r-post')  as HTMLElement;

  function computeAndRender() {
    const Sigma0 = [[prior_s, 0], [0, prior_s]];
    const A = [[A11, A12], [A21, A22]];
    const Sigma_n = [[sigma_n, 0], [0, sigma_n]];
    const b = [0, 0];
    let postMu: number[], postSigma: number[][];
    try {
      const result = linearGaussianPosterior(mu0, Sigma0, A, b, Sigma_n, obs);
      postMu = result.mu; postSigma = result.Sigma;
    } catch {
      return;
    }

    // Draw prior (blue)
    drawEllipse(gaussG, xSc, ySc, mu0, Sigma0, 'var(--gauss-q)', 'lgb-prior');
    // Draw posterior (red)
    drawEllipse(gaussG, xSc, ySc, postMu, postSigma, 'var(--gauss-p)', 'lgb-post');

    // Likelihood locus: the line Az = x (set of z where Az + b = x)
    // Parametrize as x + t * null-space-ish direction — for 2x2 overdetermined this is a single line
    // Draw as a thickened line: the set of z where ||Sigma_n^{-1}(Az - x)||^2 is small
    // For simplicity draw the locus as contours of the likelihood N(Az; x, Sigma_n)
    // We draw the 1-sigma contour of p(x|z) as a dashed region
    locusG.selectAll('*').remove();
    const GRID = 30;
    const zVals = d3.range(GRID + 1).map(i => -DOMAIN + 2 * DOMAIN * i / GRID);
    const likeData: { z1: number; z2: number; val: number }[] = [];
    for (const z1 of zVals) {
      for (const z2 of zVals) {
        const Az = [A11 * z1 + A12 * z2, A21 * z1 + A22 * z2];
        const res = [(obs[0] - Az[0]), (obs[1] - Az[1])];
        const mahal = (res[0] ** 2 + res[1] ** 2) / sigma_n;
        likeData.push({ z1, z2, val: mahal });
      }
    }
    // Draw the 1-sigma contour of the likelihood as a dashed line
    const contourThreshold = 2.3;  // chi2(2) at ~32% level ≈ 2.3
    // Simple approach: draw a line for the locus Az=x (null-space direction)
    // A is 2x2 so Az=x has a unique solution if A is invertible, but draw the locus as
    // "where mahal < threshold" with opacity encoding
    const likeRects = locusG.selectAll<SVGRectElement, { z1: number; z2: number; val: number }>('rect')
      .data(likeData.filter(d => d.val < contourThreshold * 1.5));
    likeRects.enter().append('rect')
      .attr('x', d => xSc(d.z1) - (IW / (2 * DOMAIN * GRID / 2)) / 2)
      .attr('y', d => ySc(d.z2) - (IH / (2 * DOMAIN * GRID / 2)) / 2)
      .attr('width', IW / GRID)
      .attr('height', IH / GRID)
      .attr('fill', 'var(--ink-soft)')
      .attr('fill-opacity', d => Math.max(0, 0.3 * (1 - d.val / (contourThreshold * 1.5))));

    // Readouts
    rPrior.textContent = `μ₀=(${mu0[0].toFixed(1)}, ${mu0[1].toFixed(1)}), Σ₀=${prior_s.toFixed(1)}·I`;
    rPost.textContent  = `μ=(${postMu[0].toFixed(3)}, ${postMu[1].toFixed(3)})\nΣ[0,0]=${postSigma[0][0].toFixed(4)}\nΣ[1,1]=${postSigma[1][1].toFixed(4)}`;
    rPost.innerHTML = `μ=(${postMu[0].toFixed(3)}, ${postMu[1].toFixed(3)})<br>Σ[0,0]=${postSigma[0][0].toFixed(4)}, Σ[1,1]=${postSigma[1][1].toFixed(4)}`;
  }

  function bindSlider(id: string, setter: (v: number) => void, dispId: string) {
    const sl = container.querySelector(id) as HTMLInputElement;
    sl.addEventListener('input', () => {
      setter(+sl.value);
      (container.querySelector(dispId) as HTMLElement).textContent = (+sl.value).toFixed(2).replace(/\.00$/, '.0');
      computeAndRender();
    });
  }

  bindSlider('#lgb-a11', v => { A11 = v; }, '#lgb-v-a11');
  bindSlider('#lgb-a12', v => { A12 = v; }, '#lgb-v-a12');
  bindSlider('#lgb-a21', v => { A21 = v; }, '#lgb-v-a21');
  bindSlider('#lgb-a22', v => { A22 = v; }, '#lgb-v-a22');
  bindSlider('#lgb-sn',  v => { sigma_n = v; }, '#lgb-v-sn');
  bindSlider('#lgb-ps',  v => { prior_s = v; }, '#lgb-v-ps');
  bindSlider('#lgb-x1',  v => { obs[0] = v; }, '#lgb-v-x1');
  bindSlider('#lgb-x2',  v => { obs[1] = v; }, '#lgb-v-x2');

  function resetToExample() {
    mu0 = [0, 0]; prior_s = 1; A11 = 1; A12 = 0.5; A21 = 0.3; A22 = 1; sigma_n = 0.1; obs = [1.5, 0.8];
    const vals: [string, number][] = [
      ['#lgb-a11', 1], ['#lgb-a12', 0.5], ['#lgb-a21', 0.3], ['#lgb-a22', 1],
      ['#lgb-sn', 0.1], ['#lgb-ps', 1], ['#lgb-x1', 1.5], ['#lgb-x2', 0.8],
    ];
    for (const [id, v] of vals) {
      (container.querySelector(id) as HTMLInputElement).value = String(v);
    }
    const disp: [string, string][] = [
      ['#lgb-v-a11','1.0'],['#lgb-v-a12','0.5'],['#lgb-v-a21','0.3'],['#lgb-v-a22','1.0'],
      ['#lgb-v-sn','0.10'],['#lgb-v-ps','1.0'],['#lgb-v-x1','1.5'],['#lgb-v-x2','0.8'],
    ];
    for (const [id, v] of disp) (container.querySelector(id) as HTMLElement).textContent = v;
    computeAndRender();
  }

  container.querySelector('#lgb-reset')!.addEventListener('click', resetToExample);
  computeAndRender();
}
