import * as d3 from 'd3';
import { runUntilConvergence } from '../em/algorithm';
import { getCachedGrid } from './likelihood-surface';

const GRID = 200;
const THETA_MIN = 0.05;
const THETA_MAX = 0.95;
const N_CONTOURS = 20;
const N_RESTARTS = 30;

type Mode = 'mode1' | 'mode2' | 'other';

function classifyEndpoint(tA: number, tB: number): Mode {
  if (Math.abs(tA - 0.797) < 0.05 && Math.abs(tB - 0.520) < 0.05) return 'mode1';
  if (Math.abs(tA - 0.520) < 0.05 && Math.abs(tB - 0.797) < 0.05) return 'mode2';
  return 'other';
}

const modeColors: Record<Mode, string> = {
  mode1: 'var(--coin-a)',
  mode2: 'var(--coin-b)',
  other: 'var(--ink-soft)',
};

export function mountMultiRestartGallery(container: HTMLElement): void {
  container.innerHTML = '';

  const rerollBtn = document.createElement('button');
  rerollBtn.className = 'viz-btn';
  rerollBtn.textContent = 'Re-roll 30 random starts';
  rerollBtn.style.marginBottom = '12px';
  container.appendChild(rerollBtn);

  const layout = document.createElement('div');
  layout.style.cssText = 'display:flex;gap:16px;align-items:flex-start;';
  container.appendChild(layout);

  const surfaceWrap = document.createElement('div');
  surfaceWrap.style.cssText = 'flex:1;min-width:0;';
  layout.appendChild(surfaceWrap);

  const histWrap = document.createElement('div');
  histWrap.style.cssText = 'width:140px;flex-shrink:0;';
  layout.appendChild(histWrap);

  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
  svgEl.style.width = '100%';
  svgEl.style.height = '340px';
  svgEl.style.display = 'block';
  surfaceWrap.appendChild(svgEl);

  const histEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
  histEl.style.width = '100%';
  histEl.style.height = '160px';
  histEl.style.display = 'block';
  histWrap.appendChild(histEl);

  const legend = document.createElement('div');
  legend.className = 'viz-legend';
  legend.innerHTML = `
    <span class="legend-item" style="color:var(--coin-a);">■ Mode 1 (~0.797, 0.520)</span>
    <span class="legend-item" style="color:var(--coin-b);">■ Mode 2 (~0.520, 0.797)</span>
    <span class="legend-item" style="color:var(--ink-soft);">■ Other / saddle</span>
  `;
  container.appendChild(legend);

  function generateRestarts(): { traj: { tA: number; tB: number }[]; mode: Mode }[] {
    const results = [];
    for (let r = 0; r < N_RESTARTS; r++) {
      const tA0 = THETA_MIN + Math.random() * (THETA_MAX - THETA_MIN);
      const tB0 = THETA_MIN + Math.random() * (THETA_MAX - THETA_MIN);
      const history = runUntilConvergence(tA0, tB0, 200, 1e-6);
      const last = history[history.length - 1];
      const mode = classifyEndpoint(last.thetaA, last.thetaB);
      const traj = history.map(s => ({ tA: s.thetaA, tB: s.thetaB }));
      results.push({ traj, mode });
    }
    return results;
  }

  function draw(restarts: { traj: { tA: number; tB: number }[]; mode: Mode }[]) {
    const margin = { top: 16, right: 16, bottom: 46, left: 52 };
    const fullW = svgEl.clientWidth || 400;
    const h = parseInt(svgEl.style.height) - margin.top - margin.bottom;
    const w = fullW - margin.left - margin.right;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Contour backdrop
    const grid = getCachedGrid();
    const thresholds = d3.range(N_CONTOURS).map(i => grid.min + (i / (N_CONTOURS - 1)) * (grid.max - grid.min));
    const contours = d3.contours().size([GRID, GRID]).thresholds(thresholds)(Array.from(grid.values));
    const color = d3.scaleSequential(d3.interpolatePlasma).domain([grid.min, grid.max]);
    const xScale = d3.scaleLinear().domain([0, GRID - 1]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, GRID - 1]).range([h, 0]);
    const thetaScaleX = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([0, w]);
    const thetaScaleY = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([h, 0]);
    const projection = d3.geoTransform({
      point(px, py) { this.stream.point(xScale(px), yScale(py)); },
    });
    const path = d3.geoPath(projection);
    g.selectAll('path.contour').data(contours).enter().append('path').attr('class', 'contour')
      .attr('d', path).attr('fill', d => color(d.value)).attr('stroke', 'rgba(0,0,0,0.1)').attr('stroke-width', 0.3);

    // Axes
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(thetaScaleX).ticks(5).tickFormat(d3.format('.2f')));
    g.append('g').call(d3.axisLeft(thetaScaleY).ticks(5).tickFormat(d3.format('.2f')));
    g.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('θ_A');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('θ_B');

    // Trajectories
    restarts.forEach(({ traj, mode }) => {
      const col = modeColors[mode];
      if (traj.length > 1) {
        const lineGen = d3.line<{ tA: number; tB: number }>()
          .x(d => thetaScaleX(d.tA)).y(d => thetaScaleY(d.tB));
        g.append('path').datum(traj).attr('d', lineGen)
          .attr('fill', 'none').attr('stroke', col).attr('stroke-width', 1).attr('stroke-opacity', 0.6);
      }
      const start = traj[0];
      const end = traj[traj.length - 1];
      g.append('circle').attr('cx', thetaScaleX(start.tA)).attr('cy', thetaScaleY(start.tB))
        .attr('r', 3).attr('fill', col).attr('opacity', 0.7);
      g.append('circle').attr('cx', thetaScaleX(end.tA)).attr('cy', thetaScaleY(end.tB))
        .attr('r', 5).attr('fill', col).attr('stroke', 'white').attr('stroke-width', 1.5);
    });

    // Mode histogram
    const counts: Record<Mode, number> = { mode1: 0, mode2: 0, other: 0 };
    restarts.forEach(r => { counts[r.mode]++; });
    drawHistogram(counts);
  }

  function drawHistogram(counts: Record<Mode, number>) {
    const hSvg = d3.select(histEl);
    hSvg.selectAll('*').remove();
    const margin = { top: 16, right: 10, bottom: 36, left: 10 };
    const fullW = histEl.clientWidth || 140;
    const h = parseInt(histEl.style.height) - margin.top - margin.bottom;
    const w = fullW - margin.left - margin.right;
    const g = hSvg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const modes: Mode[] = ['mode1', 'mode2', 'other'];
    const labels = ['Mode 1', 'Mode 2', 'Other'];
    const vals = modes.map(m => counts[m]);
    const maxVal = Math.max(...vals, 1);

    const xScale = d3.scaleBand<string>().domain(labels).range([0, w]).padding(0.25);
    const yScale = d3.scaleLinear().domain([0, maxVal]).range([h, 0]);

    modes.forEach((mode, i) => {
      const x = xScale(labels[i]) ?? 0;
      const bw = xScale.bandwidth();
      const v = counts[mode];
      g.append('rect').attr('x', x).attr('y', yScale(v)).attr('width', bw)
        .attr('height', h - yScale(v)).attr('fill', modeColors[mode]).attr('opacity', 0.85);
      g.append('text').attr('x', x + bw / 2).attr('y', yScale(v) - 4)
        .attr('text-anchor', 'middle').attr('font-size', '12px').attr('fill', 'var(--ink)').text(String(v));
    });

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).tickSize(0));
    g.append('g').call(d3.axisLeft(yScale).ticks(Math.min(5, maxVal)).tickFormat(d => String(Math.round(Number(d)))));
  }

  let restarts = generateRestarts();
  draw(restarts);

  rerollBtn.addEventListener('click', () => {
    restarts = generateRestarts();
    draw(restarts);
  });

  window.addEventListener('resize', () => { draw(restarts); }, { passive: true });
}
