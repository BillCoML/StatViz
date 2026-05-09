import * as d3 from 'd3';
import { EMState, qFunction, hFunction } from '../em/algorithm';
import { emBus } from './event-bus';

export function mountMonotonicityDemo(container: HTMLElement): void {
  container.innerHTML = '';

  const info = document.createElement('p');
  info.style.cssText = 'font-size:13px;color:var(--ink-soft);margin-bottom:8px;';
  info.textContent = 'Use the EM Simulator in §7 to generate steps. This panel updates automatically.';
  container.appendChild(info);

  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
  svgEl.style.width = '100%';
  svgEl.style.height = '200px';
  svgEl.style.display = 'block';
  container.appendChild(svgEl);

  const legend = document.createElement('div');
  legend.className = 'viz-legend';
  legend.innerHTML = `
    <span class="legend-item" style="color:var(--amber);">■ Q-gap (M-step gain ≥ 0)</span>
    <span class="legend-item" style="color:var(--sage);">■ −KL correction (≥ 0)</span>
  `;
  container.appendChild(legend);

  let currentHistory: EMState[] = [];

  function render() {
    const h = currentHistory.length;
    if (h < 2) {
      d3.select(svgEl).selectAll('*').remove();
      d3.select(svgEl).append('text').attr('x', 10).attr('y', 30)
        .attr('fill', 'var(--ink-soft)').attr('font-size', '13px')
        .text('Run at least one EM step in §7 to see the gain decomposition.');
      return;
    }

    const gains: { qGap: number; klTerm: number }[] = [];
    for (let i = 1; i < currentHistory.length; i++) {
      const prev = currentHistory[i - 1];
      const curr = currentHistory[i];
      const R_prev = prev.responsibilities;
      const qPrev = qFunction(prev.thetaA, prev.thetaB, R_prev);
      const qNext = qFunction(curr.thetaA, curr.thetaB, R_prev);
      const qGap = Math.max(0, qNext - qPrev);
      const hPrev = hFunction(prev.thetaA, prev.thetaB, R_prev);
      const hNext = hFunction(curr.thetaA, curr.thetaB, R_prev);
      const klTerm = Math.max(0, hPrev - hNext);
      gains.push({ qGap, klTerm });
    }

    const margin = { top: 20, right: 20, bottom: 36, left: 56 };
    const fullW = svgEl.clientWidth || 400;
    const w = fullW - margin.left - margin.right;
    const svgH = parseInt(svgEl.style.height || '200') - margin.top - margin.bottom;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleBand<number>()
      .domain(gains.map((_, i) => i + 1))
      .range([0, w]).padding(0.2);

    const maxVal = d3.max(gains, d => d.qGap + d.klTerm) ?? 1;
    const yScale = d3.scaleLinear().domain([0, maxVal * 1.1]).range([svgH, 0]);

    gains.forEach((d, idx) => {
      const x = xScale(idx + 1) ?? 0;
      const bw = xScale.bandwidth();

      // Q-gap (bottom)
      g.append('rect').attr('x', x).attr('y', yScale(d.qGap)).attr('width', bw)
        .attr('height', svgH - yScale(d.qGap))
        .attr('fill', 'var(--amber)').attr('opacity', 0.85);

      // KL term (stacked on top)
      g.append('rect').attr('x', x).attr('y', yScale(d.qGap + d.klTerm)).attr('width', bw)
        .attr('height', svgH - yScale(d.klTerm))
        .attr('fill', 'var(--sage)').attr('opacity', 0.85);
    });

    g.append('g').attr('transform', `translate(0,${svgH})`).call(d3.axisBottom(xScale).tickFormat(d => `t=${d}`));
    g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3f')));
    g.append('text').attr('x', w / 2).attr('y', svgH + 30).attr('text-anchor', 'middle')
      .attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('EM Step');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -svgH / 2).attr('y', -48)
      .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('Δℓ components');
  }

  emBus.on('step', (data) => {
    const { history: newHistory } = data as { history: EMState[]; currentIndex: number };
    currentHistory = newHistory;
    render();
  });

  render();
}
