import * as d3 from 'd3';
import { eStep } from '../em/algorithm';
import { TRIALS } from '../em/data';

export function mountResponsibilityCalculator(container: HTMLElement): void {
  container.innerHTML = '';

  // Controls
  const controlsDiv = document.createElement('div');
  controlsDiv.className = 'viz-controls';
  controlsDiv.innerHTML = `
    <label class="viz-label">
      θ_A <span id="rc-tA-val">0.60</span>
      <input type="range" id="rc-tA" min="0.05" max="0.95" step="0.01" value="0.60" />
    </label>
    <label class="viz-label">
      θ_B <span id="rc-tB-val">0.50</span>
      <input type="range" id="rc-tB" min="0.05" max="0.95" step="0.01" value="0.50" />
    </label>
  `;
  container.appendChild(controlsDiv);

  // Sparkline SVG for log(γA/γB)
  const sparkH = 80;
  const sparkMargin = { top: 14, right: 20, bottom: 30, left: 45 };
  const sparkEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  sparkEl.style.width = '100%';
  sparkEl.style.height = String(sparkH + sparkMargin.top + sparkMargin.bottom) + 'px';
  sparkEl.style.display = 'block';
  container.appendChild(sparkEl);

  const sparkSvg = d3.select(sparkEl);

  // Stacked bars SVG
  const barH = 32;
  const barPad = 8;
  const totalH = TRIALS.length * (barH + barPad) + 50;
  const barMargin = { top: 20, right: 80, bottom: 10, left: 80 };

  const barEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  barEl.style.width = '100%';
  barEl.style.height = String(totalH + barMargin.top + barMargin.bottom) + 'px';
  barEl.style.display = 'block';
  container.appendChild(barEl);

  const barSvg = d3.select(barEl);

  function draw(tA: number, tB: number) {
    const R = eStep(tA, tB);
    const fullW = barEl.clientWidth || 500;
    const bw = fullW - barMargin.left - barMargin.right;

    const xScale = d3.scaleLinear().domain([0, 1]).range([0, bw]);

    // Stacked bars
    barSvg.selectAll('*').remove();
    const g = barSvg.append('g').attr('transform', `translate(${barMargin.left},${barMargin.top})`);

    // Header
    g.append('text').attr('x', bw / 2).attr('y', -6).attr('text-anchor', 'middle')
      .attr('class', 'chart-title').text('Responsibilities γᵢᴬ (red) and γᵢᴮ (blue)');

    TRIALS.forEach((trial, i) => {
      const { gammaA, gammaB } = R[i];
      const y = i * (barH + barPad);

      // Label
      g.append('text').attr('x', -6).attr('y', y + barH / 2 + 4)
        .attr('text-anchor', 'end').attr('font-size', '13px')
        .attr('fill', 'var(--ink)').text(`Trial ${trial.id} (x=${trial.heads})`);

      // Background
      g.append('rect').attr('x', 0).attr('y', y).attr('width', bw).attr('height', barH)
        .attr('fill', 'var(--paper-soft)').attr('rx', 3);

      // γA bar
      const barA = g.append('rect').attr('class', `bar-a-${i}`)
        .attr('x', 0).attr('y', y).attr('height', barH).attr('rx', 3)
        .attr('fill', 'var(--coin-a)').attr('opacity', 0.85);
      barA.attr('width', xScale(gammaA));

      // γB bar
      const barB = g.append('rect').attr('class', `bar-b-${i}`)
        .attr('x', xScale(gammaA)).attr('y', y).attr('height', barH)
        .attr('fill', 'var(--coin-b)').attr('opacity', 0.85);
      barB.attr('width', xScale(gammaB));

      // Labels on bars
      if (gammaA > 0.08) {
        g.append('text').attr('x', xScale(gammaA) / 2).attr('y', y + barH / 2 + 4)
          .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'white')
          .text(gammaA.toFixed(4));
      }
      if (gammaB > 0.08) {
        g.append('text').attr('x', xScale(gammaA) + xScale(gammaB) / 2).attr('y', y + barH / 2 + 4)
          .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'white')
          .text(gammaB.toFixed(4));
      }

      // Numeric labels outside bars
      g.append('text').attr('x', bw + 4).attr('y', y + barH / 2 + 4)
        .attr('font-size', '11px').attr('fill', 'var(--ink-soft)')
        .text(`${gammaA.toFixed(4)} / ${gammaB.toFixed(4)}`);
    });

    // Sparkline: log(γA/γB)
    const logRatios = R.map((r, i) => ({
      i,
      v: Math.log((r.gammaA + 1e-12) / (r.gammaB + 1e-12)),
    }));

    const sparkW = sparkEl.clientWidth || 500;
    const sw = sparkW - sparkMargin.left - sparkMargin.right;

    sparkSvg.selectAll('*').remove();
    const sg = sparkSvg.append('g').attr('transform', `translate(${sparkMargin.left},${sparkMargin.top})`);
    sg.append('text').attr('x', sw / 2).attr('y', -4).attr('text-anchor', 'middle')
      .attr('class', 'chart-title').attr('font-size', '12px').text('log(γᵢᴬ / γᵢᴮ) — positive ↔ coin A preferred');

    const maxAbs = Math.max(0.1, d3.max(logRatios, d => Math.abs(d.v)) ?? 1);
    const syScale = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([sparkH, 0]);
    const sxScale = d3.scaleBand<number>().domain(TRIALS.map((_, i) => i)).range([0, sw]).padding(0.3);

    sg.append('g').attr('transform', `translate(0,${sparkH})`).call(d3.axisBottom(sxScale).tickFormat(i => `T${Number(i) + 1}`));
    sg.append('g').call(d3.axisLeft(syScale).ticks(3).tickFormat(d3.format('.2f')));

    // Zero line
    sg.append('line').attr('x1', 0).attr('x2', sw).attr('y1', syScale(0)).attr('y2', syScale(0))
      .attr('stroke', 'var(--rule)').attr('stroke-dasharray', '4,3');

    logRatios.forEach(d => {
      sg.append('rect')
        .attr('x', sxScale(d.i) ?? 0)
        .attr('width', sxScale.bandwidth())
        .attr('y', d.v >= 0 ? syScale(d.v) : syScale(0))
        .attr('height', Math.abs(syScale(d.v) - syScale(0)))
        .attr('fill', d.v >= 0 ? 'var(--coin-a)' : 'var(--coin-b)')
        .attr('opacity', 0.8);
    });
  }

  let tA = 0.6, tB = 0.5;
  draw(tA, tB);

  const sliderA = document.getElementById('rc-tA') as HTMLInputElement;
  const sliderB = document.getElementById('rc-tB') as HTMLInputElement;
  const valA = document.getElementById('rc-tA-val')!;
  const valB = document.getElementById('rc-tB-val')!;

  sliderA?.addEventListener('input', () => {
    tA = parseFloat(sliderA.value);
    valA.textContent = tA.toFixed(2);
    draw(tA, tB);
  });
  sliderB?.addEventListener('input', () => {
    tB = parseFloat(sliderB.value);
    valB.textContent = tB.toFixed(2);
    draw(tA, tB);
  });
}
