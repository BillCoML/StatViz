import * as d3 from 'd3';
import { klBernoulli } from '../math/kl';

export function mountBernoulliHeatmap(container: HTMLElement): void {
  let direction: 'pq' | 'qp' = 'pq';

  container.classList.add('bern-heatmap');
  container.innerHTML = `
    <div class="bern-heatmap__controls">
      <button class="viz-btn-sm" data-dir="pq">D(p ‖ q)</button>
      <button class="viz-btn-sm" data-dir="qp">D(q ‖ p)</button>
      <span class="bern-heatmap__readout"></span>
    </div>
    <div class="bern-heatmap__chart"></div>
  `;

  const host = container.querySelector('.bern-heatmap__chart') as HTMLElement;
  const readout = container.querySelector('.bern-heatmap__readout') as HTMLElement;

  const W = 380, H = 380, M = { top: 10, right: 60, bottom: 36, left: 40 };
  const innerW = W - M.left - M.right, innerH = H - M.top - M.bottom;
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).style('width', '100%').style('max-width', `${W}px`);
  const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

  const N = 40;
  const ps = d3.range(N).map(i => 0.025 + 0.95 * i / (N - 1));

  const xs = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const ys = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).ticks(5) as any);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(ys).ticks(5) as any);
  g.append('text').attr('class', 'axis-label').attr('x', innerW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').text('p');
  g.append('text').attr('class', 'axis-label').attr('transform', `rotate(-90) translate(${-innerH / 2},${-28})`).attr('text-anchor', 'middle').text('q');

  const cellW = innerW / N, cellH = innerH / N;
  const cellsLayer = g.append('g');
  const marker = g.append('g').attr('pointer-events', 'none');
  marker.append('rect').attr('class', 'marker-rect').attr('fill', 'none').attr('stroke', 'var(--amber)').attr('stroke-width', 2).style('display', 'none');

  // colormap: viridis-like via d3.interpolateViridis
  const color = d3.scaleSequential((t: number) => d3.interpolateViridis(t)).domain([0, 2]);

  function render() {
    const data: { p: number; q: number; v: number }[] = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const p = ps[i], q = ps[j];
      const v = direction === 'pq' ? klBernoulli(p, q) : klBernoulli(q, p);
      data.push({ p, q, v: Math.min(v, 2) });
    }
    const sel = cellsLayer.selectAll('rect.cell').data(data);
    sel.enter().append('rect').attr('class', 'cell')
      .attr('x', d => xs(d.p) - cellW / 2).attr('y', d => ys(d.q) - cellH / 2)
      .attr('width', cellW + 0.5).attr('height', cellH + 0.5)
      .merge(sel as any).attr('fill', d => color(d.v));
    cellsLayer.selectAll<SVGRectElement, { p: number; q: number; v: number }>('rect.cell')
      .on('mousemove', function(_evt, d) {
        readout.innerHTML = `p=${d.p.toFixed(2)}, q=${d.q.toFixed(2)}, <strong>D=${d.v.toFixed(3)}${d.v >= 2 ? '+' : ''}</strong>`;
        const rect = marker.select('.marker-rect');
        rect.attr('x', xs(d.p) - cellW / 2).attr('y', ys(d.q) - cellH / 2)
            .attr('width', cellW).attr('height', cellH).style('display', null);
      })
      .on('mouseleave', () => { readout.textContent = ''; marker.select('.marker-rect').style('display', 'none'); });
  }

  container.querySelectorAll('button[data-dir]').forEach(btn => {
    btn.addEventListener('click', () => {
      direction = (btn as HTMLButtonElement).dataset.dir as 'pq' | 'qp';
      render();
    });
  });

  render();
}
