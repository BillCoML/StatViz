import * as d3 from 'd3';
import { sampleMVN, eigen2x2 } from '../math/mvn';

const W = 420, H = 380;
const M = { top: 16, right: 16, bottom: 36, left: 44 };
const IW = W - M.left - M.right;
const IH = H - M.top - M.bottom;
const DOMAIN = 4.5;
const N_SAMPLES = 200;
const N_CONTOUR_LEVELS = 3; // 1, 2, 3 sigma

function isPosDef(s11: number, s12: number, s22: number): boolean {
  return s11 > 0 && s22 > 0 && s11 * s22 - s12 * s12 > 0;
}

export function mountMVNExplorer(container: HTMLElement): void {
  let mu = [0, 0];
  let s11 = 1.0, s12 = 0.4, s22 = 1.2;
  let showContours = true, showSamples = true, showEigenvectors = true;
  let samples: number[][] = [];
  container.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:0.75rem;">
      <div class="gauss-viz-row">
        <div>
          <svg id="mvn-svg" style="width:100%;max-width:${W}px;display:block;cursor:crosshair;"></svg>
          <div id="mvn-pd-warning" style="display:none;color:var(--amber);font-family:var(--font-mono);font-size:0.8em;margin-top:0.3rem;">
            ⚠ Covariance is not positive definite — adjust sliders.
          </div>
        </div>
        <div class="gauss-readout-panel">
          <div><strong>μ</strong></div>
          <div id="r-mu" style="color:var(--ink-soft);">—</div>
          <div style="margin-top:0.5rem;"><strong>Σ</strong></div>
          <div id="r-sigma" style="color:var(--ink-soft);">—</div>
          <div style="margin-top:0.5rem;"><strong>det Σ</strong></div>
          <div id="r-det" style="color:var(--ink-soft);">—</div>
          <div style="margin-top:0.5rem;"><strong>Eigenvalues</strong></div>
          <div id="r-eig" style="color:var(--ink-soft);">—</div>
          <div style="margin-top:0.5rem;"><strong>Correlation ρ</strong></div>
          <div id="r-rho" style="color:var(--ink-soft);">—</div>
        </div>
      </div>
      <div class="gauss-viz-controls">
        <label class="viz-label">Σ<sub>11</sub>
          <input type="range" id="sl-s11" min="0.1" max="4" step="0.05" value="1.0">
          <span id="v-s11" style="font-family:var(--font-mono);min-width:3em;">1.00</span>
        </label>
        <label class="viz-label">Σ<sub>22</sub>
          <input type="range" id="sl-s22" min="0.1" max="4" step="0.05" value="1.2">
          <span id="v-s22" style="font-family:var(--font-mono);min-width:3em;">1.20</span>
        </label>
        <label class="viz-label">Σ<sub>12</sub>
          <input type="range" id="sl-s12" min="-2" max="2" step="0.05" value="0.4">
          <span id="v-s12" style="font-family:var(--font-mono);min-width:3em;">0.40</span>
        </label>
        <div style="display:flex;gap:1.2em;font-size:0.85em;">
          <label><input type="checkbox" id="cb-contours" checked> Contours</label>
          <label><input type="checkbox" id="cb-samples" checked> Samples</label>
          <label><input type="checkbox" id="cb-eigenvec" checked> Eigenvectors</label>
        </div>
      </div>
    </div>
  `;

  const svgEl = container.querySelector('#mvn-svg') as SVGSVGElement;
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = d3.select(svgEl);
  const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

  const xSc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([0, IW]);
  const ySc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([IH, 0]);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${IH})`).call(d3.axisBottom(xSc).ticks(5) as any);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(ySc).ticks(5) as any);
  g.append('text').attr('class', 'axis-label').attr('x', IW / 2).attr('y', IH + 32).attr('text-anchor', 'middle').text('x₁');
  g.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)').attr('x', -IH / 2).attr('y', -36).attr('text-anchor', 'middle').text('x₂');

  const samplesG = g.append('g').attr('id', 'samples-g');
  const contoursG = g.append('g').attr('id', 'contours-g');
  const eigG = g.append('g').attr('id', 'eig-g');
  const meanDot = g.append('circle').attr('r', 6).attr('fill', 'var(--amber)').attr('stroke', 'white').attr('stroke-width', 2).style('cursor', 'grab');

  const warning = container.querySelector('#mvn-pd-warning') as HTMLElement;
  const rMu = container.querySelector('#r-mu') as HTMLElement;
  const rSigma = container.querySelector('#r-sigma') as HTMLElement;
  const rDet = container.querySelector('#r-det') as HTMLElement;
  const rEig = container.querySelector('#r-eig') as HTMLElement;
  const rRho = container.querySelector('#r-rho') as HTMLElement;

  function resample() {
    if (!isPosDef(s11, s12, s22)) { samples = []; return; }
    samples = Array.from({ length: N_SAMPLES }, () => sampleMVN(mu, [[s11, s12], [s12, s22]]));
  }

  function render() {
    const pd = isPosDef(s11, s12, s22);
    warning.style.display = pd ? 'none' : 'block';
    meanDot.attr('cx', xSc(mu[0])).attr('cy', ySc(mu[1]));

    // Readouts
    rMu.textContent = `(${mu[0].toFixed(2)}, ${mu[1].toFixed(2)})`;
    rSigma.innerHTML = `[[${s11.toFixed(2)}, ${s12.toFixed(2)}],<br>[${s12.toFixed(2)}, ${s22.toFixed(2)}]]`;
    if (pd) {
      const det = s11 * s22 - s12 * s12;
      const { values } = eigen2x2([[s11, s12], [s12, s22]]);
      const rho = s12 / Math.sqrt(s11 * s22);
      rDet.textContent = det.toFixed(4);
      rEig.textContent = `λ₁=${values[0].toFixed(3)}, λ₂=${values[1].toFixed(3)}`;
      rRho.textContent = rho.toFixed(3);
    } else {
      rDet.textContent = rEig.textContent = rRho.textContent = '—';
    }

    // Samples
    samplesG.style('display', showSamples && pd ? '' : 'none');
    if (showSamples && pd) {
      const dots = samplesG.selectAll<SVGCircleElement, number[]>('circle').data(samples);
      dots.enter().append('circle').attr('r', 2.2).attr('fill', 'var(--gauss-q)').attr('fill-opacity', 0.5)
        .merge(dots as any)
        .attr('cx', d => xSc(d[0]))
        .attr('cy', d => ySc(d[1]));
      dots.exit().remove();
    }

    // Contour ellipses
    contoursG.style('display', showContours && pd ? '' : 'none');
    contoursG.selectAll('*').remove();
    if (showContours && pd) {
      const { values, vectors } = eigen2x2([[s11, s12], [s12, s22]]);
      for (let k = 1; k <= N_CONTOUR_LEVELS; k++) {
        const rx = k * Math.sqrt(values[0]);
        const ry = k * Math.sqrt(values[1]);
        const angle = Math.atan2(vectors[0][1], vectors[0][0]) * 180 / Math.PI;
        const cx = xSc(mu[0]), cy = ySc(mu[1]);
        const rxPx = rx * (IW / (2 * DOMAIN));
        const ryPx = ry * (IH / (2 * DOMAIN));
        contoursG.append('ellipse')
          .attr('cx', cx).attr('cy', cy)
          .attr('rx', rxPx).attr('ry', ryPx)
          .attr('transform', `rotate(${angle}, ${cx}, ${cy})`)
          .attr('fill', 'none')
          .attr('stroke', 'var(--gauss-p)')
          .attr('stroke-width', k === 1 ? 2.5 : 1.5)
          .attr('stroke-opacity', 0.9 - k * 0.2);
      }
    }

    // Eigenvectors
    eigG.style('display', showEigenvectors && pd ? '' : 'none');
    eigG.selectAll('*').remove();
    if (showEigenvectors && pd) {
      const { values, vectors } = eigen2x2([[s11, s12], [s12, s22]]);
      for (let i = 0; i < 2; i++) {
        const len = Math.sqrt(values[i]);
        const scale = IW / (2 * DOMAIN);
        const dx = vectors[i][0] * len * scale;
        const dy = -vectors[i][1] * len * scale;
        const cx = xSc(mu[0]), cy = ySc(mu[1]);
        eigG.append('line')
          .attr('x1', cx - dx).attr('y1', cy - dy)
          .attr('x2', cx + dx).attr('y2', cy + dy)
          .attr('stroke', 'var(--accent-ev)').attr('stroke-width', 2.5);
        eigG.append('circle').attr('cx', cx + dx).attr('cy', cy + dy).attr('r', 4).attr('fill', 'var(--accent-ev)');
      }
    }
  }

  // Drag mean
  const drag = d3.drag<SVGCircleElement, unknown>()
    .on('start', () => { meanDot.style('cursor', 'grabbing'); })
    .on('drag', (event) => {
      mu = [xSc.invert(event.x), ySc.invert(event.y)];
      render();
    })
    .on('end', () => { meanDot.style('cursor', 'grab'); });
  meanDot.call(drag as any);

  // Sliders
  function bindSlider(id: string, setter: (v: number) => void, valId: string) {
    const sl = container.querySelector(id) as HTMLInputElement;
    sl.addEventListener('input', () => {
      setter(+sl.value);
      (container.querySelector(valId) as HTMLElement).textContent = (+sl.value).toFixed(2);
      resample();
      render();
    });
  }
  bindSlider('#sl-s11', v => { s11 = v; }, '#v-s11');
  bindSlider('#sl-s22', v => { s22 = v; }, '#v-s22');
  bindSlider('#sl-s12', v => { s12 = v; }, '#v-s12');

  const bindCb = (id: string, setter: (v: boolean) => void) => {
    const cb = container.querySelector(id) as HTMLInputElement;
    cb.addEventListener('change', () => { setter(cb.checked); render(); });
  };
  bindCb('#cb-contours', v => { showContours = v; });
  bindCb('#cb-samples',  v => { showSamples  = v; });
  bindCb('#cb-eigenvec', v => { showEigenvectors = v; });

  resample();
  render();
}
