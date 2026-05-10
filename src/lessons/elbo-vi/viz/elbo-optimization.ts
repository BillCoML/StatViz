import * as d3 from 'd3';
import { gaussianPdf } from '@lessons/kl-jensen/math/kl';
import { elboGaussian, logEvidence, posterior } from '../math/gaussian';

const MODEL = { tau2: 1, sigma2: 1, data: [2.5, 1.7, 3.1] };
const LOG_P_X = logEvidence(MODEL);
const POST = posterior(MODEL);   // { mu: 1.825, var_: 0.25 }

const PRIOR_MU = 0.05, PRIOR_VAR = 0.25;  // slider initial
const OPT_MU   = POST.mu, OPT_VAR = POST.var_;

export function mountELBOOptimization(container: HTMLElement): void {
  container.innerHTML = `
    <div class="elbo-two-panel">
      <div class="elbo-panel">
        <div style="font-family:var(--font-display);font-weight:600;margin-bottom:0.5rem;">Posterior &amp; Variational q</div>
        <svg id="elbo-opt-density" style="width:100%;max-width:340px;display:block;"></svg>
        <div style="margin-top:0.6rem;display:flex;flex-direction:column;gap:0.35rem;">
          <label class="viz-label" style="font-size:0.85em;">
            φ<sub>μ</sub>
            <input type="range" id="opt-phi-mu"  min="-1" max="4"   step="0.01" value="${PRIOR_MU}">
            <span id="r-phi-mu"  style="font-family:var(--font-mono);min-width:3em;">${PRIOR_MU.toFixed(2)}</span>
          </label>
          <label class="viz-label" style="font-size:0.85em;">
            φ<sub>σ²</sub>
            <input type="range" id="opt-phi-var" min="0.05" max="4" step="0.01" value="${PRIOR_VAR}">
            <span id="r-phi-var" style="font-family:var(--font-mono);min-width:3em;">${PRIOR_VAR.toFixed(2)}</span>
          </label>
        </div>
        <div class="elbo-buttons" style="margin-top:0.5rem;">
          <button class="viz-btn-sm" id="btn-set-prior">Set to prior</button>
          <button class="viz-btn-sm" id="btn-set-post">Set to posterior</button>
          <button class="viz-btn-sm" id="btn-gradient">Run gradient ascent</button>
        </div>
      </div>
      <div class="elbo-panel">
        <div style="font-family:var(--font-display);font-weight:600;margin-bottom:0.5rem;">ELBO Decomposition</div>
        <div class="elbo-readout" style="font-size:0.9em;">
          log p(x) = <strong class="elbo-readout--evidence">${LOG_P_X.toFixed(4)}</strong>
        </div>
        <div class="elbo-readout" style="font-size:0.9em;margin-top:0.3rem;">
          ELBO = <strong id="ro-elbo" class="elbo-readout--elbo">—</strong>
        </div>
        <div class="elbo-readout" style="font-size:0.9em;margin-top:0.3rem;">
          KL gap = <strong id="ro-kl" class="elbo-readout--kl-gap">—</strong>
        </div>
        <svg id="elbo-opt-bar" style="width:100%;max-width:340px;display:block;margin-top:1rem;"></svg>
        <div style="margin-top:0.75rem;font-size:0.8em;color:var(--ink-soft);">
          <span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--elbo);vertical-align:middle;margin-right:4px;"></span>ELBO
          &nbsp;&nbsp;
          <span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--kl-gap);vertical-align:middle;margin-right:4px;"></span>KL gap
        </div>
      </div>
    </div>
  `;

  // Density plot
  const DW = 340, DH = 200;
  const DM = { top: 10, right: 10, bottom: 26, left: 30 };
  const dEl = container.querySelector('#elbo-opt-density') as SVGSVGElement;
  dEl.setAttribute('viewBox', `0 0 ${DW} ${DH}`);
  const dsvg = d3.select(dEl);
  const dg = dsvg.append('g').attr('transform', `translate(${DM.left},${DM.top})`);
  const dW = DW - DM.left - DM.right, dH = DH - DM.top - DM.bottom;
  const dx = d3.scaleLinear().domain([-3, 5]).range([0, dW]);
  const pdfXs = d3.range(300).map(i => -3 + 8 * i / 299);

  // posterior curve (fixed)
  const postYs = pdfXs.map(x => gaussianPdf(x, POST.mu, Math.sqrt(POST.var_)));
  const dy = d3.scaleLinear().domain([0, Math.max(...postYs) * 1.2]).range([dH, 0]);
  dg.append('g').attr('class', 'axis').attr('transform', `translate(0,${dH})`).call(d3.axisBottom(dx).ticks(6) as any);
  dg.append('g').attr('class', 'axis').call(d3.axisLeft(dy).ticks(3) as any);

  const postLine = d3.line<number>().x(x => dx(x)).y(x => dy(gaussianPdf(x, POST.mu, Math.sqrt(POST.var_))));
  dg.append('path').attr('fill', 'none').attr('stroke', 'var(--posterior)').attr('stroke-width', 2.5).attr('d', postLine(pdfXs)!);
  dg.append('text').attr('x', dx(POST.mu)).attr('y', dy(gaussianPdf(POST.mu, POST.mu, Math.sqrt(POST.var_))) - 6)
    .attr('text-anchor', 'middle').attr('fill', 'var(--posterior)').attr('font-size', 10).text('p(z|x)');

  const qPath = dg.append('path').attr('fill', 'none').attr('stroke', 'var(--variational)').attr('stroke-width', 2).attr('stroke-dasharray', '5,3');
  const qLabel = dg.append('text').attr('fill', 'var(--variational)').attr('font-size', 10);

  // Bar chart SVG
  const BW = 340, BH = 60;
  const bEl = container.querySelector('#elbo-opt-bar') as SVGSVGElement;
  bEl.setAttribute('viewBox', `0 0 ${BW} ${BH}`);
  const bsvg = d3.select(bEl);
  const bg = bsvg.append('g').attr('transform', 'translate(10,8)');
  const barW = BW - 20;
  const elboRect = bg.append('rect').attr('y', 0).attr('height', 20).attr('fill', 'var(--elbo)').attr('rx', 3);
  const klRect   = bg.append('rect').attr('y', 0).attr('height', 20).attr('fill', 'var(--kl-gap)').attr('rx', 3);
  const totalLine = bg.append('line').attr('y1', -4).attr('y2', 24).attr('stroke', 'var(--evidence)').attr('stroke-width', 2).attr('stroke-dasharray', '3,2');
  bg.append('text').attr('x', 0).attr('y', 38).attr('font-size', 10).attr('fill', 'var(--ink-soft)').text('ELBO');
  const klBarLabel = bg.append('text').attr('y', 38).attr('font-size', 10).attr('fill', 'var(--ink-soft)');

  const sliderMu  = container.querySelector('#opt-phi-mu') as HTMLInputElement;
  const sliderVar = container.querySelector('#opt-phi-var') as HTMLInputElement;
  const rMu  = container.querySelector('#r-phi-mu') as HTMLElement;
  const rVar = container.querySelector('#r-phi-var') as HTMLElement;
  const roElbo = container.querySelector('#ro-elbo') as HTMLElement;
  const roKl   = container.querySelector('#ro-kl') as HTMLElement;

  let animFrame: ReturnType<typeof setTimeout> | null = null;

  function update(mu: number, v: number) {
    sliderMu.value  = String(mu);
    sliderVar.value = String(v);
    rMu.textContent  = mu.toFixed(2);
    rVar.textContent = v.toFixed(2);

    const e = elboGaussian(MODEL, mu, v);
    const k = LOG_P_X - e;
    roElbo.textContent = e.toFixed(4);
    roKl.textContent   = k.toFixed(4);

    // q density
    const qLine = d3.line<number>().x(x => dx(x)).y(x => dy(gaussianPdf(x, mu, Math.sqrt(v))));
    qPath.attr('d', qLine(pdfXs)!);
    const qPeak = gaussianPdf(mu, mu, Math.sqrt(v));
    const yPos = dy(qPeak) - 6;
    qLabel.attr('x', dx(mu)).attr('y', Math.max(6, yPos)).attr('text-anchor', 'middle').text(`q(φ)`);

    // bar chart: show ELBO and KL as fractions of -LOG_P_X (make everything positive for width)
    const totalNeg = -LOG_P_X;
    const elboFrac = Math.max(0, Math.min(1, -e / totalNeg));
    const klFrac   = Math.max(0, Math.min(1, k / totalNeg));
    const elboW = elboFrac * barW;
    const klW   = klFrac * barW;
    elboRect.attr('x', 0).attr('width', elboW);
    klRect.attr('x', elboW).attr('width', klW);
    totalLine.attr('x1', elboW + klW).attr('x2', elboW + klW);
    klBarLabel.attr('x', elboW + 4).text('KL gap');
  }

  sliderMu.addEventListener('input', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    update(+sliderMu.value, +sliderVar.value);
  });
  sliderVar.addEventListener('input', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    update(+sliderMu.value, +sliderVar.value);
  });

  container.querySelector('#btn-set-prior')!.addEventListener('click', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    update(0, 1);
  });
  container.querySelector('#btn-set-post')!.addEventListener('click', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    update(OPT_MU, OPT_VAR);
  });
  container.querySelector('#btn-gradient')!.addEventListener('click', () => {
    if (animFrame) { clearTimeout(animFrame); animFrame = null; }
    let mu = +sliderMu.value, v = +sliderVar.value;
    const eps = 0.002, lr = 0.12;
    const steps = 60;
    let i = 0;
    const step = () => {
      const e0  = elboGaussian(MODEL, mu, v);
      const eMu = elboGaussian(MODEL, mu + eps, v);
      const eV  = elboGaussian(MODEL, mu, v + eps);
      const gMu = (eMu - e0) / eps;
      const gV  = (eV  - e0) / eps;
      mu = mu + lr * gMu;
      v  = Math.max(0.01, v + lr * gV);
      update(mu, v);
      i++;
      if (i < steps) animFrame = setTimeout(step, 30);
    };
    step();
  });

  update(PRIOR_MU, PRIOR_VAR);
}
