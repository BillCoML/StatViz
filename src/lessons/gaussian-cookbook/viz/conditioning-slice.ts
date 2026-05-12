import * as d3 from 'd3';
import { conditionalGaussian2D } from '../math/conditioning';
import { sampleMVN } from '../math/mvn';

const WJ = 440, HJ = 320;  // joint scatter
const WC = 440, HC = 160;  // conditional curve
const M  = { top: 12, right: 16, bottom: 36, left: 44 };
const DOMAIN = 4;
const N_SAMPLES = 500;

function gaussianPdf1D(x: number, mu: number, sigma2: number): number {
  return (1 / Math.sqrt(2 * Math.PI * sigma2)) * Math.exp(-0.5 * (x - mu) ** 2 / sigma2);
}

export function mountConditioningSlice(container: HTMLElement): void {
  let mu_x = 0, mu_y = 0;
  let s_xx = 1, s_xy = 0.7, s_yy = 1;
  let xSlice = 1.0;
  let samples: number[][] = [];

  function resample() {
    samples = Array.from({ length: N_SAMPLES }, () =>
      sampleMVN([mu_x, mu_y], [[s_xx, s_xy], [s_xy, s_yy]]));
  }

  container.innerHTML = `
    <div class="gauss-slice-layout">
      <div>
        <div style="text-align:center;font-size:0.8em;color:var(--ink-soft);font-family:var(--font-mono);margin-bottom:0.3rem;">
          Joint (X, Y) — drag the vertical slice
        </div>
        <svg id="cs-joint-svg" style="width:100%;max-width:${WJ}px;display:block;cursor:ew-resize;"></svg>
      </div>
      <div>
        <div style="text-align:center;font-size:0.8em;color:var(--ink-soft);font-family:var(--font-mono);margin-bottom:0.3rem;" id="cs-cond-title">Y | X = 1.00</div>
        <svg id="cs-cond-svg" style="width:100%;max-width:${WC}px;display:block;"></svg>
      </div>
      <div class="gauss-viz-controls">
        <label class="viz-label">x (slice position)
          <input type="range" id="cs-xslice" min="-3" max="3" step="0.05" value="1.0">
          <span id="v-xslice" style="font-family:var(--font-mono);">1.00</span>
        </label>
        <label class="viz-label">ρ = Σ<sub>XY</sub>
          <input type="range" id="cs-rho" min="-0.95" max="0.95" step="0.05" value="0.7">
          <span id="v-rho" style="font-family:var(--font-mono);">0.70</span>
        </label>
        <label class="viz-label">Σ<sub>XX</sub>
          <input type="range" id="cs-sxx" min="0.2" max="3" step="0.05" value="1">
          <span id="v-sxx" style="font-family:var(--font-mono);">1.00</span>
        </label>
        <label class="viz-label">Σ<sub>YY</sub>
          <input type="range" id="cs-syy" min="0.2" max="3" step="0.05" value="1">
          <span id="v-syy" style="font-family:var(--font-mono);">1.00</span>
        </label>
      </div>
      <div style="font-family:var(--font-mono);font-size:0.85em;display:flex;gap:2em;flex-wrap:wrap;">
        <span>μ<sub>Y|X</sub> = <strong id="r-mu-cond">—</strong></span>
        <span>σ²<sub>Y|X</sub> = <strong id="r-var-cond">—</strong></span>
      </div>
    </div>
  `;

  // Joint SVG
  const jointEl = container.querySelector('#cs-joint-svg') as SVGSVGElement;
  jointEl.setAttribute('viewBox', `0 0 ${WJ} ${HJ}`);
  const jsvg = d3.select(jointEl);
  const jg   = jsvg.append('g').attr('transform', `translate(${M.left},${M.top})`);
  const jiW  = WJ - M.left - M.right;
  const jiH  = HJ - M.top - M.bottom;
  const jxSc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([0, jiW]);
  const jySc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([jiH, 0]);
  jg.append('g').attr('class', 'axis').attr('transform', `translate(0,${jiH})`).call(d3.axisBottom(jxSc).ticks(5) as any);
  jg.append('g').attr('class', 'axis').call(d3.axisLeft(jySc).ticks(5) as any);
  jg.append('text').attr('class', 'axis-label').attr('x', jiW / 2).attr('y', jiH + 32).attr('text-anchor', 'middle').text('X');
  jg.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)').attr('x', -jiH / 2).attr('y', -36).attr('text-anchor', 'middle').text('Y');

  const scatterG = jg.append('g');
  const sliceLine = jg.append('line').attr('stroke', 'var(--amber)').attr('stroke-width', 2).attr('stroke-dasharray', '6,3').attr('y1', 0).attr('y2', jiH);

  // Conditional SVG
  const condEl = container.querySelector('#cs-cond-svg') as SVGSVGElement;
  condEl.setAttribute('viewBox', `0 0 ${WC} ${HC}`);
  const csvg = d3.select(condEl);
  const cg   = csvg.append('g').attr('transform', `translate(${M.left},${M.top})`);
  const ciW  = WC - M.left - M.right;
  const ciH  = HC - M.top - M.bottom;
  const cxSc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([0, ciW]);
  cg.append('g').attr('class', 'axis').attr('transform', `translate(0,${ciH})`).call(d3.axisBottom(cxSc).ticks(5) as any);
  cg.append('text').attr('class', 'axis-label').attr('x', ciW / 2).attr('y', ciH + 32).attr('text-anchor', 'middle').text('Y');
  const condPath = cg.append('path').attr('fill', 'var(--gauss-p)').attr('fill-opacity', 0.25).attr('stroke', 'var(--gauss-p)').attr('stroke-width', 2.5);
  const meanLine = cg.append('line').attr('stroke', 'var(--amber)').attr('stroke-width', 2).attr('stroke-dasharray', '4,3');

  const rMuCond   = container.querySelector('#r-mu-cond')   as HTMLElement;
  const rVarCond  = container.querySelector('#r-var-cond')  as HTMLElement;
  const condTitle = container.querySelector('#cs-cond-title') as HTMLElement;

  function render() {
    // Scatter dots
    const dots = scatterG.selectAll<SVGCircleElement, number[]>('circle').data(samples);
    dots.enter().append('circle').attr('r', 2.5)
      .merge(dots as any)
      .attr('cx', d => jxSc(d[0]))
      .attr('cy', d => jySc(d[1]))
      .attr('fill', 'var(--gauss-q)').attr('fill-opacity', 0.35);
    dots.exit().remove();

    // Vertical slice
    sliceLine.attr('x1', jxSc(xSlice)).attr('x2', jxSc(xSlice));

    // Conditional distribution
    const { mu: muCond, sigma2: sig2Cond } = conditionalGaussian2D(mu_x, mu_y, s_xx, s_xy, s_yy, xSlice);
    condTitle.textContent = `Y | X = ${xSlice.toFixed(2)}`;
    rMuCond.textContent  = muCond.toFixed(4);
    rVarCond.textContent = sig2Cond.toFixed(4);

    const ys = d3.range(201).map(i => -DOMAIN + 2 * DOMAIN * i / 200);
    const pdfVals = ys.map(y => gaussianPdf1D(y, muCond, sig2Cond));
    const maxPdf = Math.max(...pdfVals);
    const cySc = d3.scaleLinear().domain([0, maxPdf * 1.2]).range([ciH, 0]);

    const area = d3.area<number>()
      .x(y => cxSc(y))
      .y0(ciH)
      .y1(y => cySc(gaussianPdf1D(y, muCond, sig2Cond)));
    condPath.attr('d', area(ys)!);

    meanLine
      .attr('x1', cxSc(muCond)).attr('x2', cxSc(muCond))
      .attr('y1', 0).attr('y2', ciH);
  }

  // Drag slice line on joint SVG
  let dragging = false;
  jsvg.on('mousedown', () => { dragging = true; })
    .on('mousemove', (event) => {
      if (!dragging) return;
      const [rawX] = d3.pointer(event, jointEl);
      const xInG = rawX - M.left;
      const newX = Math.max(-DOMAIN, Math.min(DOMAIN, jxSc.invert(xInG)));
      xSlice = newX;
      (container.querySelector('#cs-xslice') as HTMLInputElement).value = String(newX.toFixed(2));
      (container.querySelector('#v-xslice') as HTMLElement).textContent = newX.toFixed(2);
      render();
    })
    .on('mouseup', () => { dragging = false; });

  function bindSlider(id: string, setter: (v: number) => void, dispId: string, needsResample = false) {
    const sl = container.querySelector(id) as HTMLInputElement;
    sl.addEventListener('input', () => {
      setter(+sl.value);
      (container.querySelector(dispId) as HTMLElement).textContent = (+sl.value).toFixed(2);
      if (needsResample) resample();
      render();
    });
  }

  bindSlider('#cs-xslice', v => { xSlice = v; }, '#v-xslice');
  bindSlider('#cs-rho',    v => { s_xy = v * Math.sqrt(s_xx * s_yy); }, '#v-rho', true);
  bindSlider('#cs-sxx',    v => { s_xx = v; s_xy = parseFloat((container.querySelector('#cs-rho') as HTMLInputElement).value) * Math.sqrt(v * s_yy); }, '#v-sxx', true);
  bindSlider('#cs-syy',    v => { s_yy = v; s_xy = parseFloat((container.querySelector('#cs-rho') as HTMLInputElement).value) * Math.sqrt(s_xx * v); }, '#v-syy', true);

  resample();
  render();
}
