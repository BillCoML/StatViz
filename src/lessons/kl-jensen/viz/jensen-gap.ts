import * as d3 from 'd3';
import { jensenGap } from '../math/jensen';

type DistKind = 'uniform' | 'bimodal' | 'spike';

export function mountJensenGap(container: HTMLElement): void {
  let kind: DistKind = 'uniform';
  let spread = 1.0;

  container.classList.add('jensen-gap');
  container.innerHTML = `
    <div class="jensen-gap__panel">
      <div class="kl-calc__title">φ(x) = −log(x)</div>
      <div class="jensen-gap__chart-phi"></div>
      <div class="jensen-gap__readout"></div>
    </div>
    <div class="jensen-gap__panel">
      <div class="kl-calc__title">Distribution of X (on (0, 4])</div>
      <div class="viz-controls">
        <label class="viz-label">shape
          <select data-knob="kind" class="viz-btn-sm">
            <option value="uniform">uniform spread</option>
            <option value="bimodal">bimodal</option>
            <option value="spike">spike (degenerate)</option>
          </select>
        </label>
        <label class="viz-label">spread
          <input type="range" data-knob="spread" min="0.05" max="1.6" step="0.01" value="1.0">
        </label>
      </div>
      <div class="jensen-gap__chart-x"></div>
    </div>
  `;

  const phiHost = container.querySelector('.jensen-gap__chart-phi') as HTMLElement;
  const xHost   = container.querySelector('.jensen-gap__chart-x') as HTMLElement;
  const readout = container.querySelector('.jensen-gap__readout') as HTMLElement;
  const kindSel = container.querySelector('[data-knob="kind"]') as HTMLSelectElement;
  const spreadIn = container.querySelector('[data-knob="spread"]') as HTMLInputElement;

  const W = 320, H = 220, M = { top: 12, right: 12, bottom: 28, left: 36 };
  const innerW = W - M.left - M.right, innerH = H - M.top - M.bottom;
  const phiSvg = d3.select(phiHost).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).style('width', '100%').style('max-width', `${W}px`)
    .append('g').attr('transform', `translate(${M.left},${M.top})`);
  const xSvg = d3.select(xHost).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).style('width', '100%').style('max-width', `${W}px`)
    .append('g').attr('transform', `translate(${M.left},${M.top})`);

  const phi = (x: number) => -Math.log(x);

  function distribution(): { values: number[]; probs: number[] } {
    const n = 60;
    const center = 2.0;
    const grid = d3.range(n).map(i => 0.1 + (3.9 - 0.1) * i / (n - 1));
    if (kind === 'spike') {
      // Concentrated near center
      const sigma = Math.max(0.04, 0.18 * spread);
      const probs = grid.map(x => Math.exp(-((x - center) ** 2) / (2 * sigma * sigma)));
      const Z = probs.reduce((a, b) => a + b, 0);
      return { values: grid, probs: probs.map(p => p / Z) };
    }
    if (kind === 'bimodal') {
      const sigma = Math.max(0.1, 0.3 * spread);
      const probs = grid.map(x => {
        const a = Math.exp(-((x - 0.7) ** 2) / (2 * sigma * sigma));
        const b = Math.exp(-((x - 3.2) ** 2) / (2 * sigma * sigma));
        return a + b;
      });
      const Z = probs.reduce((a, b) => a + b, 0);
      return { values: grid, probs: probs.map(p => p / Z) };
    }
    // uniform: window of width determined by spread
    const half = 1.6 * spread;
    const lo = Math.max(0.1, center - half), hi = Math.min(3.9, center + half);
    const probs: number[] = grid.map(x => (x >= lo && x <= hi) ? 1 : 0);
    const Z = probs.reduce((a, b) => a + b, 0);
    return { values: grid, probs: probs.map(p => p / (Z || 1)) };
  }

  function render() {
    const { values, probs } = distribution();
    const result = jensenGap(values, probs, phi);

    // Left panel: phi
    const xs = d3.scaleLinear().domain([0.1, 4]).range([0, innerW]);
    const phiYs = values.map(phi);
    const yMin = Math.min(...phiYs, result.expectationOfPhi, result.phiOfExpectation) - 0.4;
    const yMax = Math.max(...phiYs, result.expectationOfPhi, result.phiOfExpectation) + 0.4;
    const ys = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

    phiSvg.selectAll('*').remove();
    phiSvg.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).ticks(5) as any);
    phiSvg.append('g').attr('class', 'axis').call(d3.axisLeft(ys).ticks(4) as any);
    const lineG = d3.line<number>().x(d => xs(d)).y(d => ys(phi(d)));
    phiSvg.append('path').attr('fill', 'none').attr('stroke', 'var(--ink)').attr('stroke-width', 2)
      .attr('d', lineG(d3.range(80).map(i => 0.1 + 3.9 * i / 79)));

    // Vertical at E[X]
    const eX = values.reduce((s, v, i) => s + v * probs[i], 0);
    phiSvg.append('line').attr('x1', xs(eX)).attr('x2', xs(eX)).attr('y1', 0).attr('y2', innerH)
      .attr('stroke', 'var(--ink-soft)').attr('stroke-dasharray', '3 3');
    // Horizontal at phi(E[X]) and E[phi(X)]
    phiSvg.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', ys(result.phiOfExpectation)).attr('y2', ys(result.phiOfExpectation))
      .attr('stroke', 'var(--ink-soft)').attr('stroke-dasharray', '2 4');
    phiSvg.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', ys(result.expectationOfPhi)).attr('y2', ys(result.expectationOfPhi))
      .attr('stroke', 'var(--amber)').attr('stroke-dasharray', '2 4');
    // gap
    phiSvg.append('line').attr('x1', xs(eX)).attr('x2', xs(eX))
      .attr('y1', ys(result.phiOfExpectation)).attr('y2', ys(result.expectationOfPhi))
      .attr('stroke', 'var(--warning)').attr('stroke-width', 2.5);
    phiSvg.append('text').attr('class', 'axis-label')
      .attr('x', xs(eX) + 8).attr('y', ys((result.phiOfExpectation + result.expectationOfPhi) / 2))
      .text(`gap = ${(result.gap).toFixed(3)}`);

    // Right panel: distribution histogram
    xSvg.selectAll('*').remove();
    const xs2 = d3.scaleLinear().domain([0.1, 4]).range([0, innerW]);
    const ys2 = d3.scaleLinear().domain([0, Math.max(...probs) * 1.1]).range([innerH, 0]);
    xSvg.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs2).ticks(5) as any);
    xSvg.append('g').attr('class', 'axis').call(d3.axisLeft(ys2).ticks(4) as any);
    const dx = (3.9 - 0.1) / values.length;
    xSvg.selectAll('rect').data(values).enter().append('rect')
      .attr('x', d => xs2(d) - (innerW / values.length) / 2)
      .attr('y', (_d, i) => ys2(probs[i]))
      .attr('width', innerW / values.length - 1)
      .attr('height', (_d, i) => innerH - ys2(probs[i]))
      .attr('fill', 'var(--dist-p)').attr('fill-opacity', 0.55);
    xSvg.append('line').attr('x1', xs2(eX)).attr('x2', xs2(eX)).attr('y1', 0).attr('y2', innerH)
      .attr('stroke', 'var(--ink)').attr('stroke-dasharray', '3 3');
    xSvg.append('text').attr('class', 'axis-label')
      .attr('x', xs2(eX) + 4).attr('y', 12).text(`E[X] = ${eX.toFixed(2)}`);

    void dx;

    readout.innerHTML = `
      <div>φ(E[X]) = <strong>${result.phiOfExpectation.toFixed(3)}</strong></div>
      <div>E[φ(X)] = <strong>${result.expectationOfPhi.toFixed(3)}</strong></div>
      <div>gap = <strong>${result.gap.toFixed(3)}</strong> ${Math.abs(result.gap) < 0.005 ? '— equality (X ≈ constant)' : ''}</div>
    `;
  }

  kindSel.addEventListener('change', () => { kind = kindSel.value as DistKind; render(); });
  spreadIn.addEventListener('input', () => { spread = +spreadIn.value; render(); });

  render();
}
