import * as d3 from 'd3';
import { observedLogLikelihood } from '../em/algorithm';
import { TRIALS } from '../em/data';

const GRID = 200;
const THETA_MIN = 0.05;
const THETA_MAX = 0.95;
const N_CONTOURS = 22;

export interface SurfaceGrid {
  values: Float64Array;
  min: number;
  max: number;
  thetaScale: (i: number) => number;
}

export function computeSurfaceGrid(): SurfaceGrid {
  const values = new Float64Array(GRID * GRID);
  let minV = Infinity, maxV = -Infinity;
  for (let row = 0; row < GRID; row++) {
    const tA = THETA_MIN + (row / (GRID - 1)) * (THETA_MAX - THETA_MIN);
    for (let col = 0; col < GRID; col++) {
      const tB = THETA_MIN + (col / (GRID - 1)) * (THETA_MAX - THETA_MIN);
      const v = observedLogLikelihood(tA, tB);
      values[row * GRID + col] = v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }
  return {
    values,
    min: minV,
    max: maxV,
    thetaScale: (i: number) => THETA_MIN + (i / (GRID - 1)) * (THETA_MAX - THETA_MIN),
  };
}

function completeLikelihoodGrid(zAssignment: ('A' | 'B')[]): { values: Float64Array; min: number; max: number } {
  const values = new Float64Array(GRID * GRID);
  let minV = Infinity, maxV = -Infinity;
  for (let row = 0; row < GRID; row++) {
    const tA = THETA_MIN + (row / (GRID - 1)) * (THETA_MAX - THETA_MIN);
    for (let col = 0; col < GRID; col++) {
      const tB = THETA_MIN + (col / (GRID - 1)) * (THETA_MAX - THETA_MIN);
      let ll = 0;
      TRIALS.forEach((t, i) => {
        const theta = zAssignment[i] === 'A' ? tA : tB;
        const eps = 1e-12;
        const tc = Math.max(eps, Math.min(1 - eps, theta));
        ll += t.heads * Math.log(tc) + t.tails * Math.log(1 - tc);
      });
      values[row * GRID + col] = ll;
      if (ll < minV) minV = ll;
      if (ll > maxV) maxV = ll;
    }
  }
  return { values, min: minV, max: maxV };
}

let _cachedGrid: SurfaceGrid | null = null;
export function getCachedGrid(): SurfaceGrid {
  if (!_cachedGrid) _cachedGrid = computeSurfaceGrid();
  return _cachedGrid;
}

export function renderContourSVG(
  svgEl: SVGSVGElement,
  grid: { values: Float64Array; min: number; max: number },
  w: number,
  h: number,
  opts: { showStars?: boolean; showSaddle?: boolean } = {},
): void {
  const svg = d3.select(svgEl);
  svg.selectAll('*').remove();

  const thresholds = d3.range(N_CONTOURS).map(
    i => grid.min + (i / (N_CONTOURS - 1)) * (grid.max - grid.min),
  );

  const contours = d3.contours().size([GRID, GRID]).thresholds(thresholds)(Array.from(grid.values));

  const color = d3.scaleSequential(d3.interpolatePlasma).domain([grid.min, grid.max]);

  const xScale = d3.scaleLinear().domain([0, GRID - 1]).range([0, w]);
  const yScale = d3.scaleLinear().domain([0, GRID - 1]).range([h, 0]);

  const projection = d3.geoTransform({
    point(px, py) {
      this.stream.point(xScale(px), yScale(py));
    },
  });
  const path = d3.geoPath(projection);

  const g = svg.append('g');

  g.selectAll('path.contour')
    .data(contours)
    .enter()
    .append('path')
    .attr('class', 'contour')
    .attr('d', path)
    .attr('fill', d => color(d.value))
    .attr('stroke', 'rgba(0,0,0,0.15)')
    .attr('stroke-width', 0.5);

  // Axes
  const thetaScaleX = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([0, w]);
  const thetaScaleY = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([h, 0]);

  svg.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(thetaScaleX).ticks(6).tickFormat(d3.format('.2f')));
  svg.append('g').call(d3.axisLeft(thetaScaleY).ticks(6).tickFormat(d3.format('.2f')));

  svg.append('text').attr('x', w / 2).attr('y', h + 34).attr('text-anchor', 'middle').attr('class', 'axis-label').text('θ_A');
  svg.append('text').attr('transform', `rotate(-90)`).attr('x', -h / 2).attr('y', -38).attr('text-anchor', 'middle').attr('class', 'axis-label').text('θ_B');

  if (opts.showStars) {
    const stars = [
      { tA: 0.797, tB: 0.520 },
      { tA: 0.520, tB: 0.797 },
    ];
    stars.forEach(({ tA, tB }) => {
      const cx = thetaScaleX(tA);
      const cy = thetaScaleY(tB);
      svg.append('text')
        .attr('x', cx).attr('y', cy + 6)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('fill', 'white')
        .text('★');
    });
  }

  if (opts.showSaddle) {
    const cx = thetaScaleX(0.65);
    const cy = thetaScaleY(0.65);
    svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', 5)
      .attr('fill', 'none').attr('stroke', 'yellow').attr('stroke-width', 2);
    svg.append('text').attr('x', cx + 8).attr('y', cy + 4).attr('fill', 'yellow').attr('font-size', '11px').text('saddle');
  }
}

export function mountLikelihoodSurface(container: HTMLElement): void {
  container.innerHTML = '';

  const margin = { top: 20, right: 20, bottom: 50, left: 55 };
  const h = 380;

  const wrapper = document.createElement('div');
  wrapper.style.position = 'relative';
  container.appendChild(wrapper);

  const toggleBtn = document.createElement('button');
  toggleBtn.className = 'viz-btn';
  toggleBtn.textContent = 'Show complete-data version';
  toggleBtn.style.marginBottom = '8px';
  wrapper.appendChild(toggleBtn);

  const tooltip = document.createElement('div');
  tooltip.className = 'viz-tooltip';
  tooltip.style.cssText = 'position:absolute;background:var(--paper);border:1px solid var(--rule);padding:6px 10px;border-radius:4px;font-size:12px;pointer-events:none;opacity:0;transition:opacity 0.15s;z-index:10;';
  wrapper.appendChild(tooltip);

  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
  svgEl.style.width = '100%';
  svgEl.style.height = String(h + margin.top + margin.bottom) + 'px';
  svgEl.style.display = 'block';
  wrapper.appendChild(svgEl);

  const draw = (gridData: { values: Float64Array; min: number; max: number }, showStars: boolean) => {
    const fullW = svgEl.clientWidth || 600;
    const w = fullW - margin.left - margin.right;
    const svgInner = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
    svgInner.setAttribute('width', String(w));
    svgInner.setAttribute('height', String(h));

    renderContourSVG(svgInner, gridData, w, h, { showStars, showSaddle: showStars });

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Re-draw into the group
    const thresholds = d3.range(N_CONTOURS).map(
      i => gridData.min + (i / (N_CONTOURS - 1)) * (gridData.max - gridData.min),
    );
    const contours = d3.contours().size([GRID, GRID]).thresholds(thresholds)(Array.from(gridData.values));
    const color = d3.scaleSequential(d3.interpolatePlasma).domain([gridData.min, gridData.max]);

    const xScale = d3.scaleLinear().domain([0, GRID - 1]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, GRID - 1]).range([h, 0]);
    const thetaScaleX = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([0, w]);
    const thetaScaleY = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([h, 0]);

    const projection = d3.geoTransform({
      point(px, py) { this.stream.point(xScale(px), yScale(py)); },
    });
    const path = d3.geoPath(projection);

    g.selectAll('path.contour')
      .data(contours)
      .enter().append('path').attr('class', 'contour')
      .attr('d', path)
      .attr('fill', d => color(d.value))
      .attr('stroke', 'rgba(0,0,0,0.12)')
      .attr('stroke-width', 0.4);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(thetaScaleX).ticks(6).tickFormat(d3.format('.2f')));
    g.append('g').call(d3.axisLeft(thetaScaleY).ticks(6).tickFormat(d3.format('.2f')));
    g.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').attr('class', 'axis-label').text('θ_A');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle').attr('class', 'axis-label').text('θ_B');

    if (showStars) {
      [{ tA: 0.797, tB: 0.520 }, { tA: 0.520, tB: 0.797 }].forEach(({ tA, tB }) => {
        g.append('text')
          .attr('x', thetaScaleX(tA)).attr('y', thetaScaleY(tB) + 6)
          .attr('text-anchor', 'middle').attr('font-size', '18px').attr('fill', 'white').text('★');
      });
      // Saddle dot near (0.65, 0.65)
      g.append('circle').attr('cx', thetaScaleX(0.65)).attr('cy', thetaScaleY(0.65))
        .attr('r', 5).attr('fill', 'none').attr('stroke', 'yellow').attr('stroke-width', 2);
    }

    // Tooltip on mousemove
    const rect = g.append('rect')
      .attr('width', w).attr('height', h)
      .attr('fill', 'transparent')
      .style('cursor', 'crosshair');

    rect.on('mousemove', (event: MouseEvent) => {
      const [mx, my] = d3.pointer(event);
      const tA = thetaScaleX.invert(mx);
      const tB = thetaScaleY.invert(my);
      if (tA < THETA_MIN || tA > THETA_MAX || tB < THETA_MIN || tB > THETA_MAX) return;
      const ll = observedLogLikelihood(tA, tB);
      tooltip.style.opacity = '1';
      tooltip.style.left = (margin.left + mx + 12) + 'px';
      tooltip.style.top = (margin.top + my - 20) + 'px';
      tooltip.textContent = `θ_A=${tA.toFixed(3)}, θ_B=${tB.toFixed(3)}, ℓ=${ll.toFixed(3)}`;
    });
    rect.on('mouseleave', () => { tooltip.style.opacity = '0'; });
  };

  let showComplete = false;
  const incompleteGrid = getCachedGrid();
  const completeGrid = completeLikelihoodGrid(['B', 'A', 'A', 'B', 'A']);

  draw(incompleteGrid, true);

  toggleBtn.addEventListener('click', () => {
    showComplete = !showComplete;
    toggleBtn.textContent = showComplete ? 'Show incomplete-data version' : 'Show complete-data version';
    draw(showComplete ? completeGrid : incompleteGrid, !showComplete);
  });
}
