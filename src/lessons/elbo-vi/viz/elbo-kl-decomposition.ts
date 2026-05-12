import * as d3 from 'd3';
import { klGaussian, gaussianPdf } from '@lessons/kl-jensen/math/kl';

// Model: z ~ N(0,2), x|z ~ N(z,2), obs x=2 → posterior = N(1,1)
const LOG_P_X = -0.5 * Math.log(2 * Math.PI * 4) - 4 / 8;  // x ~ N(0,4), x=2
const POST_MU = 1, POST_VAR = 1;

function elbo(phi: number): number {
  return LOG_P_X - klGaussian(phi, 1, POST_MU, Math.sqrt(POST_VAR));
}

function kl(phi: number): number {
  return klGaussian(phi, 1, POST_MU, Math.sqrt(POST_VAR));
}

export function mountELBOKLDecomposition(container: HTMLElement): void {
  const W = 680, H = 300;
  const M = { top: 16, right: 24, bottom: 36, left: 52 };
  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;

  const PHI_LO = -2, PHI_HI = 4;
  const GRID = 200;
  const phis = d3.range(GRID).map(i => PHI_LO + (PHI_HI - PHI_LO) * i / (GRID - 1));

  container.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:0.75rem;">
      <div class="elbo-viz-top" style="display:grid;grid-template-columns:2fr 1fr;gap:1rem;">
        <div>
          <svg id="elbo-decomp-svg" style="width:100%;max-width:${W}px;display:block;"></svg>
        </div>
        <div style="background:var(--paper-soft);border:1px solid var(--rule);border-radius:6px;padding:0.75rem;">
          <div style="font-family:var(--font-display);font-weight:600;margin-bottom:0.5rem;">
            Posterior (fixed) &amp; q (moving)
          </div>
          <svg id="elbo-inset-svg" style="width:100%;max-width:260px;display:block;"></svg>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:0.4rem;">
        <label class="viz-label">
          φ<sub>μ</sub> (mean of q)
          <input type="range" id="phi-slider" min="-2" max="4" step="0.02" value="2.5">
          <span id="phi-val" style="font-family:var(--font-mono);min-width:3em;">2.50</span>
        </label>
      </div>
      <div class="elbo-readout" style="font-family:var(--font-mono);font-size:0.9em;display:flex;gap:2em;flex-wrap:wrap;">
        <span>log p(x) = <strong id="r-logpx" class="elbo-readout--evidence">${LOG_P_X.toFixed(4)}</strong></span>
        <span>ELBO = <strong id="r-elbo" class="elbo-readout--elbo">—</strong></span>
        <span>KL gap = <strong id="r-kl" class="elbo-readout--kl-gap">—</strong></span>
        <span style="color:var(--ink-soft);">ELBO + KL = <strong id="r-sum">—</strong></span>
      </div>
      <div style="display:flex;gap:0.5em;">
        <button class="viz-btn-sm" id="btn-optimize">Optimize (animate to φ = 1)</button>
        <button class="viz-btn-sm" id="btn-reset">Reset</button>
      </div>
      <div style="display:flex;gap:2em;font-size:0.82em;font-family:var(--font-mono);flex-wrap:wrap;">
        <span><svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" stroke="var(--evidence)" stroke-width="2" stroke-dasharray="4,2"/></svg> log p(x)</span>
        <span><svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" stroke="var(--elbo)" stroke-width="2"/></svg> ELBO(φ)</span>
        <span><svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" stroke="var(--kl-gap)" stroke-width="2"/></svg> KL gap(φ)</span>
      </div>
    </div>
  `;

  // Main SVG
  const svgEl = container.querySelector('#elbo-decomp-svg') as SVGSVGElement;
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = d3.select(svgEl);
  const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

  const xScale = d3.scaleLinear().domain([PHI_LO, PHI_HI]).range([0, innerW]);
  const yMin = Math.min(LOG_P_X, ...phis.map(elbo)) - 0.3;
  const yMax = LOG_P_X + 0.3;
  const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(7) as any);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(yScale).ticks(5) as any);
  g.append('text').attr('class', 'axis-label').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').text('φ_μ');
  g.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -40).attr('text-anchor', 'middle').text('value');

  // log p(x) horizontal line
  g.append('line')
    .attr('x1', 0).attr('x2', innerW)
    .attr('y1', yScale(LOG_P_X)).attr('y2', yScale(LOG_P_X))
    .attr('stroke', 'var(--evidence)').attr('stroke-width', 2).attr('stroke-dasharray', '6,3');
  g.append('text').attr('x', innerW - 4).attr('y', yScale(LOG_P_X) - 5)
    .attr('text-anchor', 'end').attr('fill', 'var(--evidence)').attr('font-size', 11).text('log p(x)');

  // ELBO curve
  const elboLine = d3.line<number>().x(p => xScale(p)).y(p => yScale(elbo(p)));
  g.append('path').attr('id', 'elbo-curve').attr('fill', 'none')
    .attr('stroke', 'var(--elbo)').attr('stroke-width', 2.5)
    .attr('d', elboLine(phis)!);

  // KL curve (plotted as log_px - kl relative to the bottom so it reads as a "gap")
  // We show KL as a separate area between the ELBO curve and log p(x) line
  const klArea = d3.area<number>()
    .x(p => xScale(p)).y0(p => yScale(elbo(p))).y1(yScale(LOG_P_X));
  g.append('path').attr('id', 'kl-area').attr('fill', 'var(--kl-gap)').attr('fill-opacity', 0.18)
    .attr('d', klArea(phis)!);

  // Current phi indicator group
  const phiGroup = g.append('g').attr('id', 'phi-group');
  const vertLine = phiGroup.append('line').attr('stroke', 'var(--ink-soft)').attr('stroke-width', 1).attr('stroke-dasharray', '3,2').attr('y1', 0).attr('y2', innerH);
  const elboDot = phiGroup.append('circle').attr('r', 6).attr('fill', 'var(--elbo)').attr('stroke', 'white').attr('stroke-width', 1.5);
  const logpDot = phiGroup.append('circle').attr('r', 6).attr('fill', 'var(--evidence)').attr('stroke', 'white').attr('stroke-width', 1.5);
  const gapBar  = phiGroup.append('line').attr('stroke', 'var(--kl-gap)').attr('stroke-width', 4);

  // Inset SVG: posterior + q
  const IW = 260, IH = 160;
  const IM = { top: 8, right: 8, bottom: 24, left: 8 };
  const insetEl = container.querySelector('#elbo-inset-svg') as SVGSVGElement;
  insetEl.setAttribute('viewBox', `0 0 ${IW} ${IH}`);
  const isvg = d3.select(insetEl);
  const ig = isvg.append('g').attr('transform', `translate(${IM.left},${IM.top})`);
  const ixW = IW - IM.left - IM.right, ixH = IH - IM.top - IM.bottom;
  const ix = d3.scaleLinear().domain([-4, 6]).range([0, ixW]);
  const pdfXs = d3.range(200).map(i => -4 + 10 * i / 199);

  // posterior curve (fixed)
  const postYs = pdfXs.map(x => gaussianPdf(x, POST_MU, Math.sqrt(POST_VAR)));
  const iy = d3.scaleLinear().domain([0, Math.max(...postYs) * 1.15]).range([ixH, 0]);

  ig.append('g').attr('class', 'axis').attr('transform', `translate(0,${ixH})`).call(d3.axisBottom(ix).ticks(5) as any);

  const postLine = d3.line<number>().x(x => ix(x)).y(x => iy(gaussianPdf(x, POST_MU, Math.sqrt(POST_VAR))));
  ig.append('path').attr('fill', 'none').attr('stroke', 'var(--posterior)').attr('stroke-width', 2.5).attr('d', postLine(pdfXs)!);
  ig.append('text').attr('x', ix(POST_MU)).attr('y', iy(gaussianPdf(POST_MU, POST_MU, 1)) - 6)
    .attr('text-anchor', 'middle').attr('fill', 'var(--posterior)').attr('font-size', 10).text('p(z|x)');

  const qPath = ig.append('path').attr('fill', 'none').attr('stroke', 'var(--variational)').attr('stroke-width', 2).attr('stroke-dasharray', '5,3');
  const qLabel = ig.append('text').attr('fill', 'var(--variational)').attr('font-size', 10);

  const slider = container.querySelector('#phi-slider') as HTMLInputElement;
  const phiVal = container.querySelector('#phi-val') as HTMLElement;
  container.querySelector('#r-logpx');  // display-only, not updated programmatically
  const rElbo  = container.querySelector('#r-elbo') as HTMLElement;
  const rKl    = container.querySelector('#r-kl') as HTMLElement;
  const rSum   = container.querySelector('#r-sum') as HTMLElement;

  let animFrame: ReturnType<typeof setTimeout> | null = null;

  function update(phi: number) {
    const e = elbo(phi), k = kl(phi);
    slider.value = String(phi);
    phiVal.textContent = phi.toFixed(2);
    rElbo.textContent = e.toFixed(4);
    rKl.textContent   = k.toFixed(4);
    rSum.textContent  = (e + k).toFixed(4);

    const cx = xScale(phi);
    vertLine.attr('x1', cx).attr('x2', cx);
    elboDot.attr('cx', cx).attr('cy', yScale(e));
    logpDot.attr('cx', cx).attr('cy', yScale(LOG_P_X));
    gapBar.attr('x1', cx + 8).attr('x2', cx + 8).attr('y1', yScale(e)).attr('y2', yScale(LOG_P_X));

    // inset q curve
    const qLine = d3.line<number>().x(x => ix(x)).y(x => iy(gaussianPdf(x, phi, 1)));
    qPath.attr('d', qLine(pdfXs)!);
    qLabel.attr('x', ix(phi)).attr('y', iy(gaussianPdf(phi, phi, 1)) - 6)
      .attr('text-anchor', 'middle').text(`q(φ=${phi.toFixed(1)})`);
  }

  slider.addEventListener('input', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    update(+slider.value);
  });

  container.querySelector('#btn-optimize')!.addEventListener('click', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    const start = +slider.value, target = POST_MU;
    const steps = 40;
    let i = 0;
    const step = () => {
      const t = i / steps;
      const phi = start + (target - start) * (1 - Math.cos(Math.PI * t)) / 2;
      update(phi);
      i++;
      if (i <= steps) animFrame = setTimeout(step, 30);
    };
    step();
  });

  container.querySelector('#btn-reset')!.addEventListener('click', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    update(2.5);
  });

  update(2.5);
}
