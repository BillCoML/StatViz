import * as d3 from 'd3';
import { klMVN, klMVNTerms } from '../math/kl-mvn';
import { eigen2x2 } from '../math/mvn';

const W = 480, H = 360;
const M = { top: 16, right: 16, bottom: 36, left: 44 };
const IW = W - M.left - M.right;
const IH = H - M.top - M.bottom;
const DOMAIN = 5;

function isPosDef(s11: number, s12: number, s22: number): boolean {
  return s11 > 0 && s22 > 0 && s11 * s22 - s12 * s12 > 0;
}

function drawGaussian(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  xSc: d3.ScaleLinear<number, number>,
  ySc: d3.ScaleLinear<number, number>,
  mu: number[], s11: number, s12: number, s22: number,
  color: string,
  id: string,
) {
  g.select(`#${id}`).remove();
  if (!isPosDef(s11, s12, s22)) return;
  const ellG = g.append('g').attr('id', id);
  const { values, vectors } = eigen2x2([[s11, s12], [s12, s22]]);
  const scale = IW / (2 * DOMAIN);
  for (let k = 1; k <= 3; k++) {
    const rx = k * Math.sqrt(values[0]) * scale;
    const ry = k * Math.sqrt(values[1]) * scale;
    const angle = Math.atan2(vectors[0][1], vectors[0][0]) * 180 / Math.PI;
    const cx = xSc(mu[0]), cy = ySc(mu[1]);
    ellG.append('ellipse')
      .attr('cx', cx).attr('cy', cy)
      .attr('rx', rx).attr('ry', ry)
      .attr('transform', `rotate(${angle}, ${cx}, ${cy})`)
      .attr('fill', k === 1 ? color : 'none')
      .attr('fill-opacity', k === 1 ? 0.12 : 0)
      .attr('stroke', color)
      .attr('stroke-width', k === 1 ? 2.5 : 1.5)
      .attr('stroke-opacity', 0.9 - k * 0.2);
  }
  // mean dot
  ellG.append('circle').attr('cx', xSc(mu[0])).attr('cy', ySc(mu[1]))
    .attr('r', 5).attr('fill', color).attr('stroke', 'white').attr('stroke-width', 1.5)
    .style('cursor', 'grab');
}

export function mountKLMVNExplorer(container: HTMLElement): void {
  let pMu = [1, 1];   let pS11 = 1, pS12 = 0.5, pS22 = 1;
  let qMu = [0, 0];   let qS11 = 2, qS12 = 0,   qS22 = 2;

  container.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:0.75rem;">
      <div class="gauss-viz-row">
        <svg id="klmvn-svg" style="width:100%;max-width:${W}px;display:block;"></svg>
        <div class="gauss-readout-panel" style="min-width:200px;">
          <div><strong style="color:var(--gauss-p);">D<sub>KL</sub>(p ‖ q)</strong></div>
          <div id="r-kl-pq" style="font-size:1.15em;font-weight:700;color:var(--gauss-p);">—</div>
          <div style="margin-top:0.5rem;"><strong style="color:var(--gauss-q);">D<sub>KL</sub>(q ‖ p)</strong></div>
          <div id="r-kl-qp" style="font-size:1.15em;font-weight:700;color:var(--gauss-q);">—</div>
          <div style="margin-top:0.75rem;font-size:0.8em;color:var(--ink-soft);">Sub-terms of D(p‖q):</div>
          <div class="gauss-kl-terms" style="margin-top:0.2rem;">
            <span class="term-label">log-det</span><span id="t-logdet" class="term-val">—</span>
            <span class="term-label">−d</span><span id="t-negd" class="term-val">—</span>
            <span class="term-label">trace</span><span id="t-trace" class="term-val">—</span>
            <span class="term-label">Mahal.</span><span id="t-maha" class="term-val">—</span>
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        <div>
          <div style="font-size:0.82em;font-weight:700;color:var(--gauss-p);margin-bottom:0.3rem;">Distribution p</div>
          <div class="gauss-viz-controls">
            <label class="viz-label">μ<sub>p,1</sub> <input type="range" id="p-mu1" min="-3" max="3" step="0.1" value="1"> <span id="v-pmu1">1.0</span></label>
            <label class="viz-label">μ<sub>p,2</sub> <input type="range" id="p-mu2" min="-3" max="3" step="0.1" value="1"> <span id="v-pmu2">1.0</span></label>
            <label class="viz-label">Σ<sub>p,11</sub> <input type="range" id="p-s11" min="0.2" max="4" step="0.1" value="1"> <span id="v-ps11">1.0</span></label>
            <label class="viz-label">Σ<sub>p,22</sub> <input type="range" id="p-s22" min="0.2" max="4" step="0.1" value="1"> <span id="v-ps22">1.0</span></label>
            <label class="viz-label">Σ<sub>p,12</sub> <input type="range" id="p-s12" min="-1.9" max="1.9" step="0.1" value="0.5"> <span id="v-ps12">0.5</span></label>
          </div>
        </div>
        <div>
          <div style="font-size:0.82em;font-weight:700;color:var(--gauss-q);margin-bottom:0.3rem;">Distribution q</div>
          <div class="gauss-viz-controls">
            <label class="viz-label">μ<sub>q,1</sub> <input type="range" id="q-mu1" min="-3" max="3" step="0.1" value="0"> <span id="v-qmu1">0.0</span></label>
            <label class="viz-label">μ<sub>q,2</sub> <input type="range" id="q-mu2" min="-3" max="3" step="0.1" value="0"> <span id="v-qmu2">0.0</span></label>
            <label class="viz-label">Σ<sub>q,11</sub> <input type="range" id="q-s11" min="0.2" max="4" step="0.1" value="2"> <span id="v-qs11">2.0</span></label>
            <label class="viz-label">Σ<sub>q,22</sub> <input type="range" id="q-s22" min="0.2" max="4" step="0.1" value="2"> <span id="v-qs22">2.0</span></label>
            <label class="viz-label">Σ<sub>q,12</sub> <input type="range" id="q-s12" min="-1.9" max="1.9" step="0.1" value="0"> <span id="v-qs12">0.0</span></label>
          </div>
        </div>
      </div>
      <div class="gauss-buttons">
        <button class="viz-btn-sm" id="btn-qstandard">Set q = 𝒩(0, I)</button>
        <button class="viz-btn-sm" id="btn-matchmeans">Match means (μ_q = μ_p)</button>
        <button class="viz-btn-sm" id="btn-matchcovs">Match covariances (Σ_q = Σ_p)</button>
      </div>
      <div style="display:flex;gap:1.5em;font-size:0.8em;font-family:var(--font-mono);">
        <span><svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke="var(--gauss-p)" stroke-width="2.5"/></svg> p</span>
        <span><svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke="var(--gauss-q)" stroke-width="2.5"/></svg> q</span>
      </div>
    </div>
  `;

  const svgEl = container.querySelector('#klmvn-svg') as SVGSVGElement;
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = d3.select(svgEl);
  const gMain = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

  const xSc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([0, IW]);
  const ySc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([IH, 0]);

  gMain.append('g').attr('class', 'axis').attr('transform', `translate(0,${IH})`).call(d3.axisBottom(xSc).ticks(5) as any);
  gMain.append('g').attr('class', 'axis').call(d3.axisLeft(ySc).ticks(5) as any);
  gMain.append('text').attr('class', 'axis-label').attr('x', IW / 2).attr('y', IH + 32).attr('text-anchor', 'middle').text('x₁');

  const gaussG = gMain.append('g');

  const rKLpq = container.querySelector('#r-kl-pq') as HTMLElement;
  const rKLqp = container.querySelector('#r-kl-qp') as HTMLElement;
  const tLogdet = container.querySelector('#t-logdet') as HTMLElement;
  const tNegd   = container.querySelector('#t-negd')   as HTMLElement;
  const tTrace  = container.querySelector('#t-trace')  as HTMLElement;
  const tMaha   = container.querySelector('#t-maha')   as HTMLElement;

  function render() {
    drawGaussian(gaussG, xSc, ySc, pMu, pS11, pS12, pS22, 'var(--gauss-p)', 'gauss-p-ell');
    drawGaussian(gaussG, xSc, ySc, qMu, qS11, qS12, qS22, 'var(--gauss-q)', 'gauss-q-ell');

    const pdP = isPosDef(pS11, pS12, pS22);
    const pdQ = isPosDef(qS11, qS12, qS22);
    if (pdP && pdQ) {
      const Sigma1 = [[pS11, pS12], [pS12, pS22]];
      const Sigma2 = [[qS11, qS12], [qS12, qS22]];
      const kl1 = klMVN(pMu, Sigma1, qMu, Sigma2);
      const kl2 = klMVN(qMu, Sigma2, pMu, Sigma1);
      const terms = klMVNTerms(pMu, Sigma1, qMu, Sigma2);
      rKLpq.textContent = kl1.toFixed(4);
      rKLqp.textContent = kl2.toFixed(4);
      tLogdet.textContent = terms.logDet.toFixed(4);
      tNegd.textContent   = terms.negD.toFixed(4);
      tTrace.textContent  = terms.traceterm.toFixed(4);
      tMaha.textContent   = terms.mahalanobis.toFixed(4);
    } else {
      rKLpq.textContent = rKLqp.textContent = '—';
    }
  }

  function bindSlider(id: string, setter: (v: number) => void, dispId: string) {
    const sl = container.querySelector(id) as HTMLInputElement;
    sl.addEventListener('input', () => {
      setter(+sl.value);
      (container.querySelector(dispId) as HTMLElement).textContent = (+sl.value).toFixed(1);
      render();
    });
  }
  bindSlider('#p-mu1', v => { pMu[0] = v; }, '#v-pmu1');
  bindSlider('#p-mu2', v => { pMu[1] = v; }, '#v-pmu2');
  bindSlider('#p-s11', v => { pS11 = v; }, '#v-ps11');
  bindSlider('#p-s22', v => { pS22 = v; }, '#v-ps22');
  bindSlider('#p-s12', v => { pS12 = v; }, '#v-ps12');
  bindSlider('#q-mu1', v => { qMu[0] = v; }, '#v-qmu1');
  bindSlider('#q-mu2', v => { qMu[1] = v; }, '#v-qmu2');
  bindSlider('#q-s11', v => { qS11 = v; }, '#v-qs11');
  bindSlider('#q-s22', v => { qS22 = v; }, '#v-qs22');
  bindSlider('#q-s12', v => { qS12 = v; }, '#v-qs12');

  function syncSlider(id: string, valId: string, value: number) {
    const sl = container.querySelector(id) as HTMLInputElement;
    sl.value = String(value);
    (container.querySelector(valId) as HTMLElement).textContent = value.toFixed(1);
  }

  container.querySelector('#btn-qstandard')!.addEventListener('click', () => {
    qMu = [0, 0]; qS11 = 1; qS12 = 0; qS22 = 1;
    syncSlider('#q-mu1', '#v-qmu1', 0); syncSlider('#q-mu2', '#v-qmu2', 0);
    syncSlider('#q-s11', '#v-qs11', 1); syncSlider('#q-s22', '#v-qs22', 1); syncSlider('#q-s12', '#v-qs12', 0);
    render();
  });
  container.querySelector('#btn-matchmeans')!.addEventListener('click', () => {
    qMu = [...pMu];
    syncSlider('#q-mu1', '#v-qmu1', pMu[0]); syncSlider('#q-mu2', '#v-qmu2', pMu[1]);
    render();
  });
  container.querySelector('#btn-matchcovs')!.addEventListener('click', () => {
    qS11 = pS11; qS12 = pS12; qS22 = pS22;
    syncSlider('#q-s11', '#v-qs11', pS11); syncSlider('#q-s22', '#v-qs22', pS22); syncSlider('#q-s12', '#v-qs12', pS12);
    render();
  });

  render();
}
