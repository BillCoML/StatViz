import * as d3 from 'd3';
import { eStep, mStep, observedLogLikelihood, qFunction, hFunction } from '../em/algorithm';

const THETA_MIN = 0.05;
const THETA_MAX = 0.95;
const GRID_N = 200;

function thetaGrid(): number[] {
  return d3.range(GRID_N).map(i => THETA_MIN + (i / (GRID_N - 1)) * (THETA_MAX - THETA_MIN));
}

export function mountELBODiagram(container: HTMLElement): void {
  container.innerHTML = '';

  const margin = { top: 30, right: 30, bottom: 50, left: 60 };
  const h = 300;

  const btns = document.createElement('div');
  btns.className = 'viz-controls';
  btns.innerHTML = `
    <button class="viz-btn" id="elbo-step-e">Step E</button>
    <button class="viz-btn" id="elbo-step-m">Step M</button>
    <button class="viz-btn" id="elbo-reset">Reset</button>
    <span id="elbo-status" style="font-size:13px;color:var(--ink-soft);margin-left:12px;"></span>
  `;
  container.appendChild(btns);

  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
  svgEl.style.width = '100%';
  svgEl.style.height = String(h + margin.top + margin.bottom) + 'px';
  svgEl.style.display = 'block';
  container.appendChild(svgEl);

  const legend = document.createElement('div');
  legend.className = 'viz-legend';
  legend.innerHTML = `
    <span class="legend-item"><svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" stroke="var(--coin-b)" stroke-width="2.5"/></svg> ℓ(θ_A|x) — observed log-lik</span>
    <span class="legend-item"><svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" stroke="var(--amber)" stroke-width="2" stroke-dasharray="5,3"/></svg> Q(θ|θ^(t)) − H^(t) — lower bound</span>
  `;
  container.appendChild(legend);

  // State
  let tA = 0.6, tB = 0.5;
  let R = eStep(tA, tB);
  let eStepDone = true;

  const thetas = thetaGrid();

  function computeLLCurve(): { x: number; y: number }[] {
    return thetas.map(t => ({ x: t, y: observedLogLikelihood(t, tB) }));
  }

  function computeLBCurve(tAVal: number, R: { gammaA: number; gammaB: number }[]): { x: number; y: number }[] {
    const H0 = hFunction(tAVal, tB, R);
    return thetas.map(t => ({ x: t, y: qFunction(t, tB, R) - H0 }));
  }

  const svg = d3.select(svgEl);

  function render() {
    const fullW = svgEl.clientWidth || 600;
    const w = fullW - margin.left - margin.right;

    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const llCurve = computeLLCurve();
    const lbCurve = computeLBCurve(tA, R);

    const allY = [...llCurve.map(d => d.y), ...lbCurve.map(d => d.y)];
    const yMin = (d3.min(allY) ?? -60);
    const yMax = (d3.max(allY) ?? -30);
    const yPad = (yMax - yMin) * 0.1;

    const xScale = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([0, w]);
    const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);

    const line = d3.line<{ x: number; y: number }>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveCatmullRom.alpha(0.5));

    // LL curve
    g.append('path').datum(llCurve).attr('d', line).attr('fill', 'none')
      .attr('stroke', 'var(--coin-b)').attr('stroke-width', 2.5);

    // Lower bound curve
    g.append('path').datum(lbCurve).attr('d', line).attr('fill', 'none')
      .attr('stroke', 'var(--amber)').attr('stroke-width', 2).attr('stroke-dasharray', '6,3');

    // Current point on LL curve
    const llAtCurrent = observedLogLikelihood(tA, tB);
    g.append('circle').attr('cx', xScale(tA)).attr('cy', yScale(llAtCurrent))
      .attr('r', 7).attr('fill', 'var(--coin-b)').attr('stroke', 'var(--paper)').attr('stroke-width', 2);

    // Label
    g.append('text').attr('x', xScale(tA) + 10).attr('y', yScale(llAtCurrent) - 8)
      .attr('font-size', '12px').attr('fill', 'var(--ink)')
      .text(`θ_A^(t)=${tA.toFixed(3)}`);

    // Axes
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(8).tickFormat(d3.format('.2f')));
    g.append('g').call(d3.axisLeft(yScale).ticks(6).tickFormat(d3.format('.1f')));
    g.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').attr('class', 'axis-label').text('θ_A');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -48).attr('text-anchor', 'middle').attr('class', 'axis-label').text('log-likelihood');

    const status = document.getElementById('elbo-status');
    if (status) {
      status.textContent = `θ_A=${tA.toFixed(4)}, θ_B=${tB.toFixed(4)}, ℓ=${llAtCurrent.toFixed(4)}`;
    }
  }

  render();

  document.getElementById('elbo-step-e')?.addEventListener('click', () => {
    R = eStep(tA, tB);
    eStepDone = true;
    render();
  });

  document.getElementById('elbo-step-m')?.addEventListener('click', () => {
    if (!eStepDone) R = eStep(tA, tB);
    const next = mStep(R);
    tA = next.thetaA;
    tB = next.thetaB;
    R = eStep(tA, tB);
    eStepDone = true;
    render();
  });

  document.getElementById('elbo-reset')?.addEventListener('click', () => {
    tA = 0.6; tB = 0.5;
    R = eStep(tA, tB);
    eStepDone = true;
    render();
  });

  window.addEventListener('resize', () => { render(); }, { passive: true });
}
