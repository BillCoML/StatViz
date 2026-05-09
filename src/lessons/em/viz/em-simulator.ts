import * as d3 from 'd3';
import { EMState, initialState, eStep, mStep, runEMStep, observedLogLikelihood } from '../em/algorithm';
import { emBus } from './event-bus';
import { getCachedGrid } from './likelihood-surface';
import { TRIALS } from '../em/data';

const GRID = 200;
const THETA_MIN = 0.05;
const THETA_MAX = 0.95;
const N_CONTOURS = 20;

const redMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function mountEMSimulator(container: HTMLElement): void {
  container.innerHTML = '';

  // ---- Controls ----
  const ctrlDiv = document.createElement('div');
  ctrlDiv.className = 'simulator-controls';
  ctrlDiv.innerHTML = `
    <div class="sim-sliders">
      <label class="viz-label">θ_A<sup>(0)</sup> <span id="sim-tA-val">0.60</span>
        <input type="range" id="sim-tA" min="0.1" max="0.9" step="0.01" value="0.60" />
      </label>
      <label class="viz-label">θ_B<sup>(0)</sup> <span id="sim-tB-val">0.50</span>
        <input type="range" id="sim-tB" min="0.1" max="0.9" step="0.01" value="0.50" />
      </label>
    </div>
    <div class="sim-buttons">
      <button class="viz-btn" id="sim-step-e">Step E</button>
      <button class="viz-btn" id="sim-step-m">Step M</button>
      <button class="viz-btn" id="sim-step-both">Step Both</button>
      <button class="viz-btn sim-btn-play" id="sim-play">▶ Play</button>
      <button class="viz-btn" id="sim-reset">⟲ Reset</button>
    </div>
    <div class="sim-presets">
      <span style="font-size:12px;color:var(--ink-soft);">Presets:</span>
      <button class="viz-btn-sm" id="preset-sym">Symmetric (0.5, 0.5)</button>
      <button class="viz-btn-sm" id="preset-asym">Asymmetric (0.8, 0.2)</button>
      <button class="viz-btn-sm" id="preset-adv">Adversarial (0.3, 0.7)</button>
      <button class="viz-btn-sm" id="preset-rand">Random</button>
    </div>
    <div id="sim-status" class="sim-status">Iteration 0 &nbsp;|&nbsp; ‖Δθ‖ = — &nbsp;|&nbsp; ℓ = —</div>
  `;
  container.appendChild(ctrlDiv);

  // ---- 4-panel grid ----
  const gridDiv = document.createElement('div');
  gridDiv.className = 'sim-grid';
  gridDiv.innerHTML = `
    <div class="sim-panel">
      <div class="sim-panel-title">Responsibilities</div>
      <svg id="sim-resp" style="width:100%;display:block;"></svg>
    </div>
    <div class="sim-panel">
      <div class="sim-panel-title">Trajectory on Likelihood Surface</div>
      <svg id="sim-traj" style="width:100%;display:block;"></svg>
    </div>
    <div class="sim-panel">
      <div class="sim-panel-title">Parameter Evolution</div>
      <svg id="sim-params" style="width:100%;display:block;"></svg>
    </div>
    <div class="sim-panel">
      <div class="sim-panel-title">Log-Likelihood (monotone)</div>
      <svg id="sim-ll" style="width:100%;display:block;"></svg>
    </div>
  `;
  container.appendChild(gridDiv);

  // ---- State ----
  let history: EMState[] = [];
  let pendingR: { gammaA: number; gammaB: number }[] | null = null;
  let playTimer: ReturnType<typeof setInterval> | null = null;
  let initThetaA = 0.6;
  let initThetaB = 0.5;

  function reset(tA = initThetaA, tB = initThetaB) {
    if (playTimer !== null) { clearInterval(playTimer); playTimer = null; }
    history = [initialState(tA, tB)];
    pendingR = eStep(tA, tB);
    renderAll();
    emBus.emit('step', { history: [...history], currentIndex: 0 });
  }

  function doStepE() {
    const last = history[history.length - 1];
    pendingR = eStep(last.thetaA, last.thetaB);
    renderResponsibilities(pendingR);
    updateStatus(last, null);
  }

  function doStepM() {
    const last = history[history.length - 1];
    const R = pendingR ?? eStep(last.thetaA, last.thetaB);
    const { thetaA, thetaB } = mStep(R);
    const newR = eStep(thetaA, thetaB);
    const next: EMState = {
      thetaA,
      thetaB,
      iteration: last.iteration + 1,
      responsibilities: newR,
      observedLogLikelihood: observedLogLikelihood(thetaA, thetaB),
    };
    history.push(next);
    pendingR = next.responsibilities;
    renderAll();
    emBus.emit('step', { history: [...history], currentIndex: history.length - 1 });
  }

  function doStepBoth() {
    doStepE();
    setTimeout(() => doStepM(), redMotion() ? 0 : 400);
  }

  function doPlay() {
    if (playTimer !== null) {
      clearInterval(playTimer);
      playTimer = null;
      const btn = document.getElementById('sim-play');
      if (btn) btn.textContent = '▶ Play';
      return;
    }
    const btn = document.getElementById('sim-play');
    if (btn) btn.textContent = '⏸ Pause';
    playTimer = setInterval(() => {
      const last = history[history.length - 1];
      const prev = history.length >= 2 ? history[history.length - 2] : null;
      const delta = prev ? Math.abs(last.thetaA - prev.thetaA) + Math.abs(last.thetaB - prev.thetaB) : 1;
      if (delta < 1e-6 || history.length > 100) {
        clearInterval(playTimer!);
        playTimer = null;
        const b = document.getElementById('sim-play');
        if (b) b.textContent = '▶ Play';
        return;
      }
      const next = runEMStep(last);
      history.push(next);
      pendingR = next.responsibilities;
      renderAll();
      emBus.emit('step', { history: [...history], currentIndex: history.length - 1 });
    }, redMotion() ? 100 : 600);
  }

  // ---- Panel renders ----
  function renderResponsibilities(R: { gammaA: number; gammaB: number }[]) {
    const svgEl = document.getElementById('sim-resp') as SVGSVGElement | null;
    if (!svgEl) return;
    const margin = { top: 16, right: 10, bottom: 20, left: 70 };
    const barH = 22, barPad = 6;
    const totalH = TRIALS.length * (barH + barPad) + 4;
    svgEl.style.height = String(totalH + margin.top + margin.bottom) + 'px';
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    const fullW = svgEl.clientWidth || 300;
    const w = fullW - margin.left - margin.right;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, w]);

    TRIALS.forEach((trial, i) => {
      const { gammaA, gammaB } = R[i];
      const y = i * (barH + barPad);
      g.append('text').attr('x', -4).attr('y', y + barH / 2 + 4).attr('text-anchor', 'end')
        .attr('font-size', '12px').attr('fill', 'var(--ink)').text(`T${trial.id} x=${trial.heads}`);
      g.append('rect').attr('x', 0).attr('y', y).attr('width', w).attr('height', barH)
        .attr('fill', 'var(--paper-soft)').attr('rx', 2);
      g.append('rect').attr('x', 0).attr('y', y).attr('height', barH).attr('rx', 2)
        .attr('fill', 'var(--coin-a)').attr('opacity', 0.85)
        .attr('width', xScale(gammaA));
      g.append('rect').attr('x', xScale(gammaA)).attr('y', y).attr('height', barH)
        .attr('fill', 'var(--coin-b)').attr('opacity', 0.85)
        .attr('width', xScale(gammaB));
      if (gammaA > 0.1) {
        g.append('text').attr('x', xScale(gammaA / 2)).attr('y', y + barH / 2 + 4)
          .attr('text-anchor', 'middle').attr('font-size', '10px').attr('fill', 'white')
          .text(gammaA.toFixed(3));
      }
      if (gammaB > 0.1) {
        g.append('text').attr('x', xScale(gammaA) + xScale(gammaB / 2)).attr('y', y + barH / 2 + 4)
          .attr('text-anchor', 'middle').attr('font-size', '10px').attr('fill', 'white')
          .text(gammaB.toFixed(3));
      }
    });
  }

  function renderTrajectory() {
    const svgEl = document.getElementById('sim-traj') as SVGSVGElement | null;
    if (!svgEl) return;
    const margin = { top: 16, right: 16, bottom: 46, left: 52 };
    const h = 240;
    svgEl.style.height = String(h + margin.top + margin.bottom) + 'px';
    const fullW = svgEl.clientWidth || 300;
    const w = fullW - margin.left - margin.right;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const grid = getCachedGrid();
    const thetaScaleX = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([0, w]);
    const thetaScaleY = d3.scaleLinear().domain([THETA_MIN, THETA_MAX]).range([h, 0]);
    const xScale = d3.scaleLinear().domain([0, GRID - 1]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, GRID - 1]).range([h, 0]);

    const thresholds = d3.range(N_CONTOURS).map(i => grid.min + (i / (N_CONTOURS - 1)) * (grid.max - grid.min));
    const contours = d3.contours().size([GRID, GRID]).thresholds(thresholds)(Array.from(grid.values));
    const color = d3.scaleSequential(d3.interpolatePlasma).domain([grid.min, grid.max]);
    const projection = d3.geoTransform({
      point(px, py) { this.stream.point(xScale(px), yScale(py)); },
    });
    const path = d3.geoPath(projection);

    g.selectAll('path.contour').data(contours).enter().append('path').attr('class', 'contour')
      .attr('d', path).attr('fill', d => color(d.value)).attr('stroke', 'rgba(0,0,0,0.1)').attr('stroke-width', 0.3);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(thetaScaleX).ticks(5).tickFormat(d3.format('.2f')));
    g.append('g').call(d3.axisLeft(thetaScaleY).ticks(5).tickFormat(d3.format('.2f')));
    g.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('θ_A');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('θ_B');

    // Trajectory polyline
    if (history.length > 1) {
      const lineGen = d3.line<EMState>()
        .x(d => thetaScaleX(d.thetaA))
        .y(d => thetaScaleY(d.thetaB));
      g.append('path').datum(history).attr('d', lineGen)
        .attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 1.5).attr('stroke-opacity', 0.85);
    }

    // Points
    history.forEach((s, i) => {
      const isLast = i === history.length - 1;
      g.append('circle')
        .attr('cx', thetaScaleX(s.thetaA)).attr('cy', thetaScaleY(s.thetaB))
        .attr('r', isLast ? 6 : 3)
        .attr('fill', isLast ? 'var(--amber)' : 'white')
        .attr('stroke', 'var(--paper)').attr('stroke-width', isLast ? 2 : 1)
        .attr('opacity', 0.9);
      if (isLast) {
        g.append('text').attr('x', thetaScaleX(s.thetaA) + 8).attr('y', thetaScaleY(s.thetaB) + 4)
          .attr('font-size', '10px').attr('fill', 'var(--amber)').text(`t=${s.iteration}`);
      }
    });
  }

  function renderParams() {
    const svgEl = document.getElementById('sim-params') as SVGSVGElement | null;
    if (!svgEl) return;
    const margin = { top: 16, right: 16, bottom: 46, left: 52 };
    const h = 200;
    svgEl.style.height = String(h + margin.top + margin.bottom) + 'px';
    const fullW = svgEl.clientWidth || 300;
    const w = fullW - margin.left - margin.right;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, Math.max(1, history.length - 1)]).range([0, w]);
    const allTheta = history.flatMap(s => [s.thetaA, s.thetaB]);
    const yScale = d3.scaleLinear().domain([
      Math.max(0, (d3.min(allTheta) ?? 0) - 0.05),
      Math.min(1, (d3.max(allTheta) ?? 1) + 0.05),
    ]).range([h, 0]);

    const lineA = d3.line<EMState>().x((_, i) => xScale(i)).y(d => yScale(d.thetaA));
    const lineB = d3.line<EMState>().x((_, i) => xScale(i)).y(d => yScale(d.thetaB));

    g.append('path').datum(history).attr('d', lineA).attr('fill', 'none').attr('stroke', 'var(--coin-a)').attr('stroke-width', 2);
    g.append('path').datum(history).attr('d', lineB).attr('fill', 'none').attr('stroke', 'var(--coin-b)').attr('stroke-width', 2);

    // Last point dots
    const last = history[history.length - 1];
    g.append('circle').attr('cx', xScale(history.length - 1)).attr('cy', yScale(last.thetaA)).attr('r', 4).attr('fill', 'var(--coin-a)');
    g.append('circle').attr('cx', xScale(history.length - 1)).attr('cy', yScale(last.thetaB)).attr('r', 4).attr('fill', 'var(--coin-b)');

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(Math.min(10, history.length)).tickFormat(d => String(Math.round(Number(d)))));
    g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.2f')));
    g.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('Iteration');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('θ');

    // Legend
    g.append('circle').attr('cx', w - 80).attr('cy', 10).attr('r', 5).attr('fill', 'var(--coin-a)');
    g.append('text').attr('x', w - 72).attr('y', 14).attr('font-size', '11px').attr('fill', 'var(--coin-a)').text('θ_A');
    g.append('circle').attr('cx', w - 80).attr('cy', 26).attr('r', 5).attr('fill', 'var(--coin-b)');
    g.append('text').attr('x', w - 72).attr('y', 30).attr('font-size', '11px').attr('fill', 'var(--coin-b)').text('θ_B');
  }

  function renderLL() {
    const svgEl = document.getElementById('sim-ll') as SVGSVGElement | null;
    if (!svgEl) return;
    const margin = { top: 16, right: 16, bottom: 46, left: 56 };
    const h = 200;
    svgEl.style.height = String(h + margin.top + margin.bottom) + 'px';
    const fullW = svgEl.clientWidth || 300;
    const w = fullW - margin.left - margin.right;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, Math.max(1, history.length - 1)]).range([0, w]);
    const llVals = history.map(s => s.observedLogLikelihood);
    const yScale = d3.scaleLinear().domain([
      (d3.min(llVals) ?? -50) - 1,
      (d3.max(llVals) ?? -30) + 1,
    ]).range([h, 0]);

    const lineLL = d3.line<number>().x((_, i) => xScale(i)).y(d => yScale(d));
    g.append('path').datum(llVals).attr('d', lineLL).attr('fill', 'none').attr('stroke', 'var(--sage)').attr('stroke-width', 2.5);

    // Dots
    llVals.forEach((v, i) => {
      g.append('circle').attr('cx', xScale(i)).attr('cy', yScale(v)).attr('r', i === llVals.length - 1 ? 5 : 3)
        .attr('fill', 'var(--sage)').attr('stroke', 'var(--paper)').attr('stroke-width', 1);
    });

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(Math.min(10, history.length)).tickFormat(d => String(Math.round(Number(d)))));
    g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.1f')));
    g.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('Iteration');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -48).attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', 'var(--ink-soft)').text('ℓ(θ|x)');
  }

  function updateStatus(s: EMState, prev: EMState | null) {
    const el = document.getElementById('sim-status');
    if (!el) return;
    const delta = prev ? (Math.abs(s.thetaA - prev.thetaA) + Math.abs(s.thetaB - prev.thetaB)).toFixed(6) : '—';
    el.innerHTML = `Iteration ${s.iteration} &nbsp;|&nbsp; ‖Δθ‖ = ${delta} &nbsp;|&nbsp; ℓ = ${s.observedLogLikelihood.toFixed(4)}`;
  }

  function renderAll() {
    const last = history[history.length - 1];
    const prev = history.length >= 2 ? history[history.length - 2] : null;
    const R = pendingR ?? last.responsibilities;
    renderResponsibilities(R);
    renderTrajectory();
    renderParams();
    renderLL();
    updateStatus(last, prev);
  }

  // ---- Wire buttons ----
  document.getElementById('sim-step-e')?.addEventListener('click', doStepE);
  document.getElementById('sim-step-m')?.addEventListener('click', doStepM);
  document.getElementById('sim-step-both')?.addEventListener('click', doStepBoth);
  document.getElementById('sim-play')?.addEventListener('click', doPlay);
  document.getElementById('sim-reset')?.addEventListener('click', () => reset(initThetaA, initThetaB));

  const sliderA = document.getElementById('sim-tA') as HTMLInputElement;
  const sliderB = document.getElementById('sim-tB') as HTMLInputElement;
  sliderA?.addEventListener('input', () => {
    initThetaA = parseFloat(sliderA.value);
    const v = document.getElementById('sim-tA-val');
    if (v) v.textContent = initThetaA.toFixed(2);
    reset(initThetaA, initThetaB);
  });
  sliderB?.addEventListener('input', () => {
    initThetaB = parseFloat(sliderB.value);
    const v = document.getElementById('sim-tB-val');
    if (v) v.textContent = initThetaB.toFixed(2);
    reset(initThetaA, initThetaB);
  });

  const setPreset = (tA: number, tB: number) => {
    initThetaA = tA; initThetaB = tB;
    const sA = document.getElementById('sim-tA') as HTMLInputElement;
    const sB = document.getElementById('sim-tB') as HTMLInputElement;
    if (sA) { sA.value = String(tA); const v = document.getElementById('sim-tA-val'); if (v) v.textContent = tA.toFixed(2); }
    if (sB) { sB.value = String(tB); const v = document.getElementById('sim-tB-val'); if (v) v.textContent = tB.toFixed(2); }
    reset(tA, tB);
  };

  document.getElementById('preset-sym')?.addEventListener('click', () => setPreset(0.5, 0.5));
  document.getElementById('preset-asym')?.addEventListener('click', () => setPreset(0.8, 0.2));
  document.getElementById('preset-adv')?.addEventListener('click', () => setPreset(0.3, 0.7));
  document.getElementById('preset-rand')?.addEventListener('click', () => {
    const tA = Math.round((0.1 + Math.random() * 0.8) * 100) / 100;
    const tB = Math.round((0.1 + Math.random() * 0.8) * 100) / 100;
    setPreset(tA, tB);
  });

  window.addEventListener('resize', () => { renderAll(); }, { passive: true });

  // ---- Initialize ----
  reset(initThetaA, initThetaB);
}
