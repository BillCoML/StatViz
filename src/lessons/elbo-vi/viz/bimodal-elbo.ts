import * as d3 from 'd3';
import { gaussianPdf } from '@lessons/kl-jensen/math/kl';
import { elboBimodal, bimodalPosterior } from '../math/bimodal';

const LOG_P_X = 0.0;  // by construction: the bimodal target integrates to 1

// Precompute ELBO heatmap
const MU_LO = -6, MU_HI = 6, MU_N = 44;
const LS_LO = Math.log(0.3), LS_HI = Math.log(5), LS_N = 36;
const MU_VALS = d3.range(MU_N).map(i => MU_LO + (MU_HI - MU_LO) * i / (MU_N - 1));
const LS_VALS = d3.range(LS_N).map(i => LS_LO + (LS_HI - LS_LO) * i / (LS_N - 1));

function buildHeatmap(): Float64Array {
  const data = new Float64Array(MU_N * LS_N);
  for (let mi = 0; mi < MU_N; mi++) {
    for (let li = 0; li < LS_N; li++) {
      const mu = MU_VALS[mi];
      const v  = Math.exp(LS_VALS[li]) * Math.exp(LS_VALS[li]);
      data[mi * LS_N + li] = elboBimodal(mu, v, 800);
    }
  }
  return data;
}

export function mountBimodalELBO(container: HTMLElement): void {
  container.innerHTML = `
    <div class="bimodal-elbo">
      <div class="bimodal-elbo__target">
        <strong>Target:</strong> p(z|x) = ½ N(−3, 1) + ½ N(+3, 1) &nbsp;·&nbsp;
        log p(x) = <strong style="color:var(--evidence);">0.0000</strong>
      </div>
      <div class="bimodal-elbo__pair">
        <div class="elbo-panel">
          <div style="font-family:var(--font-display);font-weight:600;margin-bottom:0.5rem;">Variational Fit</div>
          <svg id="bm-density-svg" style="width:100%;max-width:360px;display:block;"></svg>
          <div style="margin-top:0.6rem;display:flex;flex-direction:column;gap:0.35rem;">
            <label class="viz-label" style="font-size:0.85em;">
              φ<sub>μ</sub>
              <input type="range" id="bm-mu"  min="-6" max="6"    step="0.05"  value="0">
              <span id="bm-r-mu"  style="font-family:var(--font-mono);min-width:3em;">0.00</span>
            </label>
            <label class="viz-label" style="font-size:0.85em;">
              φ<sub>σ²</sub>
              <input type="range" id="bm-var" min="0.09" max="25" step="0.05"  value="1">
              <span id="bm-r-var" style="font-family:var(--font-mono);min-width:3em;">1.00</span>
            </label>
          </div>
          <div class="elbo-readout" style="font-size:0.9em;margin-top:0.4rem;">
            ELBO = <strong id="bm-ro-elbo" class="elbo-readout--elbo">—</strong>
            &nbsp;·&nbsp;
            KL gap = <strong id="bm-ro-kl" class="elbo-readout--kl-gap">—</strong>
          </div>
          <div class="elbo-buttons" style="margin-top:0.6rem;">
            <button class="viz-btn-sm" id="btn-bm-m5">Init at −5</button>
            <button class="viz-btn-sm" id="btn-bm-0">Init at 0</button>
            <button class="viz-btn-sm" id="btn-bm-p5">Init at +5</button>
            <button class="viz-btn-sm" id="btn-bm-run">Run gradient ascent</button>
          </div>
        </div>
        <div class="elbo-panel">
          <div style="font-family:var(--font-display);font-weight:600;margin-bottom:0.5rem;">ELBO Landscape</div>
          <div style="font-size:0.78em;color:var(--ink-soft);margin-bottom:0.3rem;">
            x-axis: φ<sub>μ</sub> &nbsp;|&nbsp; y-axis: φ<sub>σ</sub> (log scale) &nbsp;|&nbsp; color: ELBO value
          </div>
          <svg id="bm-heatmap-svg" style="width:100%;max-width:360px;display:block;"></svg>
        </div>
      </div>
    </div>
  `;

  // Density plot setup
  const DW = 360, DH = 200;
  const DM = { top: 10, right: 10, bottom: 26, left: 30 };
  const dEl = container.querySelector('#bm-density-svg') as SVGSVGElement;
  dEl.setAttribute('viewBox', `0 0 ${DW} ${DH}`);
  const dsvg = d3.select(dEl);
  const dg = dsvg.append('g').attr('transform', `translate(${DM.left},${DM.top})`);
  const dW = DW - DM.left - DM.right, dH = DH - DM.top - DM.bottom;
  const dx = d3.scaleLinear().domain([-8, 8]).range([0, dW]);
  const pdfXs = d3.range(300).map(i => -8 + 16 * i / 299);

  const postYs = pdfXs.map(x => bimodalPosterior(x));
  const dy = d3.scaleLinear().domain([0, Math.max(...postYs) * 1.2]).range([dH, 0]);
  dg.append('g').attr('class', 'axis').attr('transform', `translate(0,${dH})`).call(d3.axisBottom(dx).ticks(7) as any);
  dg.append('g').attr('class', 'axis').call(d3.axisLeft(dy).ticks(3) as any);

  const postLine = d3.line<number>().x(x => dx(x)).y(x => dy(bimodalPosterior(x)));
  dg.append('path').attr('fill', 'none').attr('stroke', 'var(--posterior)').attr('stroke-width', 2.5).attr('d', postLine(pdfXs)!);
  dg.append('text').attr('x', dx(-3)).attr('y', dy(bimodalPosterior(-3)) - 6)
    .attr('text-anchor', 'middle').attr('fill', 'var(--posterior)').attr('font-size', 10).text('p(z|x)');

  const qArea = dg.append('path').attr('fill', 'var(--variational)').attr('fill-opacity', 0.18);
  const qPath = dg.append('path').attr('fill', 'none').attr('stroke', 'var(--variational)').attr('stroke-width', 2).attr('stroke-dasharray', '5,3');
  const qLabel = dg.append('text').attr('fill', 'var(--variational)').attr('font-size', 10);

  // Heatmap setup
  const HW = 360, HH = 260;
  const HM = { top: 10, right: 24, bottom: 36, left: 36 };
  const hEl = container.querySelector('#bm-heatmap-svg') as SVGSVGElement;
  hEl.setAttribute('viewBox', `0 0 ${HW} ${HH}`);
  const hsvg = d3.select(hEl);
  const hg = hsvg.append('g').attr('transform', `translate(${HM.left},${HM.top})`);
  const hW = HW - HM.left - HM.right, hH = HH - HM.top - HM.bottom;

  const hx = d3.scaleLinear().domain([MU_LO, MU_HI]).range([0, hW]);
  const hy = d3.scaleLinear().domain([LS_LO, LS_HI]).range([hH, 0]);

  hg.append('g').attr('class', 'axis').attr('transform', `translate(0,${hH})`).call(d3.axisBottom(hx).ticks(7) as any);
  hg.append('g').attr('class', 'axis').call(d3.axisLeft(hy).ticks(4).tickFormat(d => `${Math.exp(+d).toFixed(1)}`) as any);
  hg.append('text').attr('class', 'axis-label').attr('x', hW / 2).attr('y', hH + 30).attr('text-anchor', 'middle').text('φ_μ');
  hg.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)').attr('x', -hH / 2).attr('y', -30).attr('text-anchor', 'middle').text('φ_σ');

  // Build heatmap lazily on first render (expensive)
  let heatmapBuilt = false;
  let heatmapData: Float64Array | null = null;
  let colorScale: d3.ScaleSequential<string> | null = null;
  const cellW = hW / MU_N, cellH = hH / LS_N;

  function buildAndDrawHeatmap() {
    if (heatmapBuilt) return;
    heatmapBuilt = true;
    heatmapData = buildHeatmap();
    const minE = Math.min(...Array.from(heatmapData));
    const maxE = Math.max(...Array.from(heatmapData));
    colorScale = d3.scaleSequential(d3.interpolateViridis).domain([minE, maxE]);

    for (let mi = 0; mi < MU_N; mi++) {
      for (let li = 0; li < LS_N; li++) {
        const e = heatmapData[mi * LS_N + li];
        hg.append('rect')
          .attr('x', hx(MU_VALS[mi]) - cellW / 2)
          .attr('y', hy(LS_VALS[li]) - cellH / 2)
          .attr('width', cellW + 1).attr('height', cellH + 1)
          .attr('fill', colorScale(e));
      }
    }
    hg.append('g').attr('class', 'axis').attr('transform', `translate(0,${hH})`).call(d3.axisBottom(hx).ticks(7) as any);
    hg.append('g').attr('class', 'axis').call(d3.axisLeft(hy).ticks(4).tickFormat(d => `${Math.exp(+d).toFixed(1)}`) as any);
  }

  // Current point on heatmap
  const cursorGroup = hg.append('g');
  const cursor = cursorGroup.append('circle').attr('r', 5).attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 2);
  const trajectoryPath = cursorGroup.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 1.5).attr('stroke-dasharray', '3,2').attr('opacity', 0.7);
  const trajectoryPts: [number, number][] = [];

  const sliderMu  = container.querySelector('#bm-mu') as HTMLInputElement;
  const sliderVar = container.querySelector('#bm-var') as HTMLInputElement;
  const rMu  = container.querySelector('#bm-r-mu') as HTMLElement;
  const rVar = container.querySelector('#bm-r-var') as HTMLElement;
  const roElbo = container.querySelector('#bm-ro-elbo') as HTMLElement;
  const roKl   = container.querySelector('#bm-ro-kl') as HTMLElement;

  let animFrame: ReturnType<typeof setTimeout> | null = null;

  function update(mu: number, v: number, addToTraj = false) {
    sliderMu.value  = String(mu);
    sliderVar.value = String(v);
    rMu.textContent  = mu.toFixed(2);
    rVar.textContent = v.toFixed(2);

    const e = elboBimodal(mu, v);
    const k = LOG_P_X - e;
    roElbo.textContent = e.toFixed(4);
    roKl.textContent   = k.toFixed(4);

    // density
    const sigma = Math.sqrt(v);
    const qYs = pdfXs.map(x => gaussianPdf(x, mu, sigma));
    const areaFn = d3.area<number>().x(x => dx(x)).y0(dH).y1(x => dy(gaussianPdf(x, mu, sigma)));
    const lineFn = d3.line<number>().x(x => dx(x)).y(x => dy(gaussianPdf(x, mu, sigma)));
    qArea.attr('d', areaFn(pdfXs)!);
    qPath.attr('d', lineFn(pdfXs)!);
    const peak = Math.max(...qYs);
    qLabel.attr('x', dx(mu)).attr('y', dy(peak) - 6).attr('text-anchor', 'middle').text(`q`);

    // heatmap cursor
    const logSigma = Math.log(sigma);
    cursor.attr('cx', hx(mu)).attr('cy', hy(logSigma));
    if (addToTraj) {
      trajectoryPts.push([hx(mu), hy(logSigma)]);
      if (trajectoryPts.length > 1) {
        const trajLine = d3.line<[number, number]>().x(p => p[0]).y(p => p[1]);
        trajectoryPath.attr('d', trajLine(trajectoryPts)!);
      }
    }
  }

  function resetTrajectory() {
    trajectoryPts.length = 0;
    trajectoryPath.attr('d', null);
  }

  sliderMu.addEventListener('input', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    resetTrajectory();
    buildAndDrawHeatmap();
    update(+sliderMu.value, +sliderVar.value);
  });
  sliderVar.addEventListener('input', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    resetTrajectory();
    buildAndDrawHeatmap();
    update(+sliderMu.value, +sliderVar.value);
  });

  function initAt(mu: number, v: number) {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    resetTrajectory();
    buildAndDrawHeatmap();
    update(mu, v, true);
  }

  container.querySelector('#btn-bm-m5')!.addEventListener('click', () => initAt(-5, 1));
  container.querySelector('#btn-bm-0')!.addEventListener('click',  () => initAt(0, 1));
  container.querySelector('#btn-bm-p5')!.addEventListener('click', () => initAt(5, 1));

  container.querySelector('#btn-bm-run')!.addEventListener('click', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    buildAndDrawHeatmap();
    let mu = +sliderMu.value, logS = Math.log(Math.sqrt(+sliderVar.value));
    const eps = 0.08, lr = 0.12;
    const steps = 80;
    let i = 0;
    const step = () => {
      const v0 = Math.exp(logS) * Math.exp(logS);
      const e0  = elboBimodal(mu, v0, 600);
      const eMu = elboBimodal(mu + eps, v0, 600);
      const vUp = Math.exp(logS + eps); const eLS = elboBimodal(mu, vUp * vUp, 600);
      const gMu = (eMu - e0) / eps;
      const gLS = (eLS - e0) / eps;
      mu   = Math.max(MU_LO + 0.1, Math.min(MU_HI - 0.1, mu + lr * gMu));
      logS = Math.max(LS_LO + 0.05, Math.min(LS_HI - 0.05, logS + lr * 0.5 * gLS));
      update(mu, Math.exp(logS) * Math.exp(logS), true);
      i++;
      if (i < steps) animFrame = setTimeout(step, 35);
    };
    step();
  });

  // Initial render (don't build heatmap yet — it's expensive; defer to first interaction)
  update(0, 1);

  // Build heatmap after a short delay so the page doesn't block on load
  setTimeout(() => buildAndDrawHeatmap(), 200);
}
