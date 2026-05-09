import * as d3 from 'd3';
import { TRIALS } from '../em/data';

const N = 10;
const K_VALUES = d3.range(0, N + 1); // 0..10

function logFactorial(n: number): number {
  let r = 0;
  for (let i = 2; i <= n; i++) r += Math.log(i);
  return r;
}

function logBinomCoeff(n: number, k: number): number {
  return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}

function binomPMF(k: number, theta: number): number {
  if (theta <= 0) return k === 0 ? 1 : 0;
  if (theta >= 1) return k === N ? 1 : 0;
  const logp = logBinomCoeff(N, k) + k * Math.log(theta) + (N - k) * Math.log(1 - theta);
  return Math.exp(logp);
}

function mixturePMF(k: number, tA: number, tB: number): number {
  return 0.5 * binomPMF(k, tA) + 0.5 * binomPMF(k, tB);
}

const dur = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 250;

export function mountBinomialMixture(container: HTMLElement): void {
  container.innerHTML = '';

  // Sliders
  const controlsDiv = document.createElement('div');
  controlsDiv.className = 'viz-controls';
  controlsDiv.innerHTML = `
    <label class="viz-label">
      θ_A <span id="bm-tA-val">0.60</span>
      <input type="range" id="bm-tA" min="0.05" max="0.95" step="0.01" value="0.60" />
    </label>
    <label class="viz-label">
      θ_B <span id="bm-tB-val">0.50</span>
      <input type="range" id="bm-tB" min="0.05" max="0.95" step="0.01" value="0.50" />
    </label>
  `;
  container.appendChild(controlsDiv);

  // SVG dimensions
  const margin = { top: 30, right: 20, bottom: 40, left: 45 };
  const chartH = 160;
  const gap = 68;
  const svgTotalH = 3 * chartH + 2 * gap + margin.top + margin.bottom;

  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
  svgEl.setAttribute('class', 'bm-svg');
  svgEl.style.width = '100%';
  svgEl.style.height = String(svgTotalH) + 'px';
  svgEl.style.display = 'block';
  container.appendChild(svgEl);

  const svg = d3.select(svgEl);
  const fullWidth = svgEl.clientWidth || 600;
  const w = fullWidth - margin.left - margin.right;

  const xScale = d3.scaleBand<number>()
    .domain(K_VALUES)
    .range([0, w])
    .padding(0.15);

  const yScaleA = d3.scaleLinear().range([chartH, 0]).domain([0, 0.35]);
  const yScaleB = d3.scaleLinear().range([chartH, 0]).domain([0, 0.35]);
  const yScaleMix = d3.scaleLinear().range([chartH, 0]).domain([0, 0.35]);

  function makeChart(yOffset: number, label: string) {
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top + yOffset})`);
    g.append('text').attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'var(--ink-soft)').text(label);
    // x axis
    g.append('g').attr('class', 'bm-xaxis').attr('transform', `translate(0,${chartH})`)
      .call(d3.axisBottom(xScale).tickFormat((d: d3.NumberValue) => String(d)));
    // y axis placeholder — stored as a typed group
    const yAxisG = g.append<SVGGElement>('g').attr('class', 'bm-yaxis');
    g.append('text').attr('x', w / 2).attr('y', chartH + 32).attr('text-anchor', 'middle')
      .attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('k (heads)');
    return { g, yAxisG };
  }

  const { g: gA, yAxisG: yAxisA } = makeChart(0, 'Bin(10, θ_A)');
  const { g: gB, yAxisG: yAxisB } = makeChart(chartH + gap, 'Bin(10, θ_B)');
  const { g: gMix, yAxisG: yAxisMix } = makeChart(2 * (chartH + gap), 'Mixture: 0.5·Bin(10,θ_A) + 0.5·Bin(10,θ_B)');

  // Init y-axes
  yAxisA.call(d3.axisLeft(yScaleA).ticks(5).tickFormat(d3.format('.2f')));
  yAxisB.call(d3.axisLeft(yScaleB).ticks(5).tickFormat(d3.format('.2f')));
  yAxisMix.call(d3.axisLeft(yScaleMix).ticks(5).tickFormat(d3.format('.2f')));

  // Observed data dots below mixture chart
  const obsY = chartH + 12;
  const obsGroup = gMix.append('g').attr('class', 'obs-dots');
  const xValues = TRIALS.map(t => t.heads);
  const obsCounts: Map<number, number> = new Map();
  xValues.forEach(v => obsCounts.set(v, (obsCounts.get(v) ?? 0) + 1));
  obsCounts.forEach((count, val) => {
    for (let j = 0; j < count; j++) {
      obsGroup.append('circle')
        .attr('cx', (xScale(val) ?? 0) + xScale.bandwidth() / 2)
        .attr('cy', obsY + j * 9)
        .attr('r', 4)
        .attr('fill', 'var(--ink-soft)')
        .attr('stroke', 'var(--paper)')
        .attr('stroke-width', 1);
    }
  });
  gMix.append('text').attr('x', -2).attr('y', obsY + 4)
    .attr('text-anchor', 'end').attr('font-size', '10px').attr('fill', 'var(--ink-soft)').text('obs');

  function draw(tA: number, tB: number) {
    const pmfA = K_VALUES.map(k => ({ k, p: binomPMF(k, tA) }));
    const pmfB = K_VALUES.map(k => ({ k, p: binomPMF(k, tB) }));
    const pmfMix = K_VALUES.map(k => ({ k, p: mixturePMF(k, tA, tB), pA: binomPMF(k, tA), pB: binomPMF(k, tB) }));

    const maxValA = d3.max(pmfA, d => d.p) ?? 0.35;
    const maxValB = d3.max(pmfB, d => d.p) ?? 0.35;
    const maxValMix = d3.max(pmfMix, d => d.p) ?? 0.35;
    yScaleA.domain([0, Math.max(maxValA * 1.1, 0.05)]);
    yScaleB.domain([0, Math.max(maxValB * 1.1, 0.05)]);
    yScaleMix.domain([0, Math.max(maxValMix * 1.1, 0.05)]);

    const td = dur();

    // Coin A bars
    const selA = gA.selectAll<SVGRectElement, { k: number; p: number }>('rect.bar').data(pmfA, d => String(d.k));
    const entA = selA.enter().append('rect').attr('class', 'bar')
      .attr('x', d => xScale(d.k) ?? 0)
      .attr('width', xScale.bandwidth())
      .attr('y', chartH).attr('height', 0)
      .attr('fill', 'var(--coin-a)').attr('opacity', 0.8);
    entA.merge(selA).transition().duration(td)
      .attr('x', d => xScale(d.k) ?? 0)
      .attr('y', d => yScaleA(d.p))
      .attr('height', d => chartH - yScaleA(d.p));
    yAxisA.call(d3.axisLeft(yScaleA).ticks(5).tickFormat(d3.format('.2f')));

    // Coin B bars
    const selB = gB.selectAll<SVGRectElement, { k: number; p: number }>('rect.bar').data(pmfB, d => String(d.k));
    const entB = selB.enter().append('rect').attr('class', 'bar')
      .attr('x', d => xScale(d.k) ?? 0)
      .attr('width', xScale.bandwidth())
      .attr('y', chartH).attr('height', 0)
      .attr('fill', 'var(--coin-b)').attr('opacity', 0.8);
    entB.merge(selB).transition().duration(td)
      .attr('x', d => xScale(d.k) ?? 0)
      .attr('y', d => yScaleB(d.p))
      .attr('height', d => chartH - yScaleB(d.p));
    yAxisB.call(d3.axisLeft(yScaleB).ticks(5).tickFormat(d3.format('.2f')));

    // Mixture bars
    type MixDatum = { k: number; p: number; pA: number; pB: number };
    const selMix = gMix.selectAll<SVGRectElement, MixDatum>('rect.bar').data(pmfMix, d => String(d.k));
    const entMix = selMix.enter().append('rect').attr('class', 'bar')
      .attr('x', d => xScale(d.k) ?? 0)
      .attr('width', xScale.bandwidth())
      .attr('y', chartH).attr('height', 0).attr('opacity', 0.75);
    entMix.merge(selMix).transition().duration(td)
      .attr('x', d => xScale(d.k) ?? 0)
      .attr('y', d => yScaleMix(d.p))
      .attr('height', d => chartH - yScaleMix(d.p))
      .attr('fill', d => d.pA >= d.pB ? 'var(--coin-a)' : 'var(--coin-b)');
    yAxisMix.call(d3.axisLeft(yScaleMix).ticks(5).tickFormat(d3.format('.2f')));
  }

  let tA = 0.6, tB = 0.5;
  draw(tA, tB);

  const sliderA = document.getElementById('bm-tA') as HTMLInputElement | null;
  const sliderB = document.getElementById('bm-tB') as HTMLInputElement | null;
  const valA = document.getElementById('bm-tA-val');
  const valB = document.getElementById('bm-tB-val');

  sliderA?.addEventListener('input', () => {
    tA = parseFloat(sliderA.value);
    if (valA) valA.textContent = tA.toFixed(2);
    draw(tA, tB);
  });
  sliderB?.addEventListener('input', () => {
    tB = parseFloat(sliderB.value);
    if (valB) valB.textContent = tB.toFixed(2);
    draw(tA, tB);
  });
}
