import * as d3 from 'd3';

const W = 320, H = 280;
const M = { top: 12, right: 12, bottom: 36, left: 40 };
const IW = W - M.left - M.right;
const IH = H - M.top - M.bottom;
const DOMAIN = 4;
const N = 200;

function boxMuller(rng: () => number): number {
  let u = rng(); while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function cholesky2x2(s11: number, s12: number, s22: number): [number, number, number] {
  const L11 = Math.sqrt(s11);
  const L21 = s12 / L11;
  const L22 = Math.sqrt(Math.max(0, s22 - L21 * L21));
  return [L11, L21, L22];
}

export function mountReparamFlow(container: HTMLElement): void {
  let mu = [0.5, -0.5];
  let s11 = 2, s12 = 0.6, s22 = 1.2;
  let epsSamples: number[][] = [];
  let hoveredIdx: number | null = null;

  function generateEps(): number[][] {
    const rng = Math.random;
    return Array.from({ length: N }, () => [boxMuller(rng), boxMuller(rng)]);
  }

  function transform(eps: number[][]): number[][] {
    const [L11, L21, L22] = cholesky2x2(s11, s12, s22);
    return eps.map(([e1, e2]) => [
      mu[0] + L11 * e1,
      mu[1] + L21 * e1 + L22 * e2,
    ]);
  }

  container.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:0.75rem;">
      <div class="gauss-two-panel">
        <div>
          <div style="text-align:center;font-family:var(--font-mono);font-size:0.8em;color:var(--ink-soft);margin-bottom:0.3rem;">ε ~ 𝒩(0, I)</div>
          <svg id="rp-left-svg" style="width:100%;max-width:${W}px;display:block;"></svg>
        </div>
        <div>
          <div style="text-align:center;font-family:var(--font-mono);font-size:0.8em;color:var(--ink-soft);margin-bottom:0.3rem;">Z = μ + Lε</div>
          <svg id="rp-right-svg" style="width:100%;max-width:${W}px;display:block;"></svg>
        </div>
      </div>
      <div class="gauss-viz-controls">
        <label class="viz-label">μ₁ <input type="range" id="rp-mu1" min="-2" max="2" step="0.1" value="0.5"> <span id="rv-mu1">0.5</span></label>
        <label class="viz-label">μ₂ <input type="range" id="rp-mu2" min="-2" max="2" step="0.1" value="-0.5"> <span id="rv-mu2">−0.5</span></label>
        <label class="viz-label">Σ₁₁ <input type="range" id="rp-s11" min="0.2" max="4" step="0.1" value="2"> <span id="rv-s11">2.0</span></label>
        <label class="viz-label">Σ₂₂ <input type="range" id="rp-s22" min="0.2" max="4" step="0.1" value="1.2"> <span id="rv-s22">1.2</span></label>
        <label class="viz-label">Σ₁₂ <input type="range" id="rp-s12" min="-1.9" max="1.9" step="0.1" value="0.6"> <span id="rv-s12">0.6</span></label>
      </div>
      <div>
        <button class="viz-btn-sm" id="rp-reroll">Re-roll ε</button>
      </div>
      <div style="font-size:0.8em;color:var(--ink-soft);font-family:var(--font-serif);">
        Hover a dot to highlight its pair. The same ε-samples morph deterministically as you adjust Σ — the noise is fixed; parameters do all the work.
      </div>
    </div>
  `;

  function makeSvg(id: string) {
    const el = container.querySelector(id) as SVGSVGElement;
    el.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const s = d3.select(el);
    const g = s.append('g').attr('transform', `translate(${M.left},${M.top})`);
    const xSc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([0, IW]);
    const ySc = d3.scaleLinear().domain([-DOMAIN, DOMAIN]).range([IH, 0]);
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${IH})`).call(d3.axisBottom(xSc).ticks(5) as any);
    g.append('g').attr('class', 'axis').call(d3.axisLeft(ySc).ticks(5) as any);
    return { g, xSc, ySc };
  }

  const left  = makeSvg('#rp-left-svg');
  const right = makeSvg('#rp-right-svg');

  const leftDots  = left.g.append('g');
  const rightDots = right.g.append('g');

  // Contour ellipse on right panel
  const rightEll = right.g.append('g').attr('id', 'rp-ell');

  function renderEllipse() {
    rightEll.selectAll('*').remove();
    if (s11 * s22 - s12 * s12 <= 0) return;
    const { values, vectors } = computeEigen(s11, s12, s22);
    const scale = IW / (2 * DOMAIN);
    const cx = right.xSc(mu[0]), cy = right.ySc(mu[1]);
    for (let k = 1; k <= 2; k++) {
      const rx = k * Math.sqrt(values[0]) * scale;
      const ry = k * Math.sqrt(values[1]) * scale;
      const angle = Math.atan2(vectors[0][1], vectors[0][0]) * 180 / Math.PI;
      rightEll.append('ellipse')
        .attr('cx', cx).attr('cy', cy).attr('rx', rx).attr('ry', ry)
        .attr('transform', `rotate(${angle}, ${cx}, ${cy})`)
        .attr('fill', 'none')
        .attr('stroke', 'var(--gauss-p)').attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.6 - k * 0.1);
    }
  }

  function computeEigen(s11: number, s12: number, s22: number) {
    const trace = s11 + s22;
    const det = s11 * s22 - s12 * s12;
    const disc = Math.sqrt(Math.max(0, (trace / 2) ** 2 - det));
    const l1 = trace / 2 + disc, l2 = trace / 2 - disc;
    let v1: [number, number], v2: [number, number];
    if (Math.abs(s12) > 1e-10) {
      const n1 = Math.hypot(l1 - s11, s12); v1 = [(l1 - s11) / n1, s12 / n1];
      const n2 = Math.hypot(l2 - s11, s12); v2 = [(l2 - s11) / n2, s12 / n2];
    } else { v1 = [1, 0]; v2 = [0, 1]; }
    return { values: [l1, l2] as [number, number], vectors: [v1, v2] as [[number, number], [number, number]] };
  }

  function render() {
    const zSamples = transform(epsSamples);
    renderEllipse();

    const lData = leftDots.selectAll<SVGCircleElement, number[]>('circle').data(epsSamples);
    lData.enter().append('circle').attr('r', 3)
      .merge(lData as any)
      .attr('cx', d => left.xSc(d[0])).attr('cy', d => left.ySc(d[1]))
      .attr('fill', (_, i) => i === hoveredIdx ? 'var(--amber)' : 'var(--gauss-q)')
      .attr('fill-opacity', (_, i) => i === hoveredIdx ? 1 : 0.5)
      .attr('r', (_, i) => i === hoveredIdx ? 5 : 3)
      .on('mouseenter', (_, i) => { hoveredIdx = i as unknown as number; render(); })
      .on('mouseleave', () => { hoveredIdx = null; render(); });
    lData.exit().remove();

    const rData = rightDots.selectAll<SVGCircleElement, number[]>('circle').data(zSamples);
    rData.enter().append('circle').attr('r', 3)
      .merge(rData as any)
      .attr('cx', d => right.xSc(d[0])).attr('cy', d => right.ySc(d[1]))
      .attr('fill', (_, i) => i === hoveredIdx ? 'var(--amber)' : 'var(--gauss-p)')
      .attr('fill-opacity', (_, i) => i === hoveredIdx ? 1 : 0.5)
      .attr('r', (_, i) => i === hoveredIdx ? 5 : 3)
      .on('mouseenter', (_, i) => { hoveredIdx = i as unknown as number; render(); })
      .on('mouseleave', () => { hoveredIdx = null; render(); });
    rData.exit().remove();
  }

  function bindSlider(id: string, setter: (v: number) => void, dispId: string, fmt?: (v: number) => string) {
    const sl = container.querySelector(id) as HTMLInputElement;
    sl.addEventListener('input', () => {
      setter(+sl.value);
      (container.querySelector(dispId) as HTMLElement).textContent = fmt ? fmt(+sl.value) : (+sl.value).toFixed(1);
      render();
    });
  }
  bindSlider('#rp-mu1', v => { mu[0] = v; }, '#rv-mu1');
  bindSlider('#rp-mu2', v => { mu[1] = v; }, '#rv-mu2', v => v >= 0 ? v.toFixed(1) : `−${Math.abs(v).toFixed(1)}`);
  bindSlider('#rp-s11', v => { s11 = v; }, '#rv-s11');
  bindSlider('#rp-s22', v => { s22 = v; }, '#rv-s22');
  bindSlider('#rp-s12', v => { s12 = v; }, '#rv-s12');

  container.querySelector('#rp-reroll')!.addEventListener('click', () => {
    epsSamples = generateEps();
    render();
  });

  epsSamples = generateEps();
  render();
}
