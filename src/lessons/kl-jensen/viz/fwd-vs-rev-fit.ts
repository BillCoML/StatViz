import * as d3 from 'd3';
import { gaussianPdf } from '../math/kl';

interface TargetParams {
  delta: number;   // mode separation (each mode at ±delta)
  pi: number;      // weight on left mode
  sigma: number;   // mode width
}

interface QParams { mu: number; sigma: number }

function targetPdf(x: number, t: TargetParams): number {
  return t.pi * gaussianPdf(x, -t.delta, t.sigma) + (1 - t.pi) * gaussianPdf(x, t.delta, t.sigma);
}

function targetMoments(t: TargetParams): { mu: number; sigma: number } {
  const mu = t.pi * (-t.delta) + (1 - t.pi) * t.delta;
  // Var = E[X^2] - mu^2; E[X^2] for each component = sigma^2 + center^2.
  const ex2 = t.pi * (t.sigma ** 2 + t.delta ** 2) + (1 - t.pi) * (t.sigma ** 2 + t.delta ** 2);
  return { mu, sigma: Math.sqrt(Math.max(1e-6, ex2 - mu * mu)) };
}

function buildGrid(low: number, high: number, n: number): { xs: number[]; dx: number } {
  const xs = d3.range(n).map(i => low + (high - low) * i / (n - 1));
  return { xs, dx: (high - low) / (n - 1) };
}

function reverseKL(q: QParams, t: TargetParams, xs: number[], dx: number): number {
  let s = 0;
  for (const x of xs) {
    const qx = gaussianPdf(x, q.mu, q.sigma);
    const px = targetPdf(x, t);
    if (qx < 1e-12) continue;
    if (px < 1e-300) return Infinity;
    s += qx * Math.log(qx / px) * dx;
  }
  return s;
}

function forwardKL(q: QParams, t: TargetParams, xs: number[], dx: number): number {
  let s = 0;
  for (const x of xs) {
    const px = targetPdf(x, t);
    if (px < 1e-12) continue;
    const qx = gaussianPdf(x, q.mu, q.sigma);
    if (qx < 1e-300) return Infinity;
    s += px * Math.log(px / qx) * dx;
  }
  return s;
}

/** Numerically minimize reverse KL over (mu, sigma). 50-step gradient descent on log-sigma. */
function reverseKLOptimize(t: TargetParams, init: QParams,
                            xs: number[], dx: number,
                            steps = 50, lr = 0.18): { history: QParams[]; final: QParams } {
  let mu = init.mu, logSigma = Math.log(Math.max(0.05, init.sigma));
  const history: QParams[] = [{ mu, sigma: Math.exp(logSigma) }];
  const eps = 1e-3;
  for (let s = 0; s < steps; s++) {
    const k0 = reverseKL({ mu, sigma: Math.exp(logSigma) }, t, xs, dx);
    const kMu = reverseKL({ mu: mu + eps, sigma: Math.exp(logSigma) }, t, xs, dx);
    const kLs = reverseKL({ mu, sigma: Math.exp(logSigma + eps) }, t, xs, dx);
    if (!isFinite(k0)) break;
    const gMu = (kMu - k0) / eps;
    const gLs = (kLs - k0) / eps;
    mu -= lr * gMu;
    logSigma -= lr * 0.5 * gLs;
    if (logSigma < Math.log(0.2)) logSigma = Math.log(0.2);
    if (logSigma > Math.log(8))  logSigma = Math.log(8);
    history.push({ mu, sigma: Math.exp(logSigma) });
  }
  return { history, final: { mu, sigma: Math.exp(logSigma) } };
}

export function mountForwardVsReverseFit(container: HTMLElement): void {
  let target: TargetParams = { delta: 3.0, pi: 0.5, sigma: 1.0 };
  let revInitMu = 3.0;

  container.classList.add('fr-fit');
  container.innerHTML = `
    <div class="fr-fit__target">
      <div class="kl-calc__title">Target distribution p(x)</div>
      <div class="viz-controls">
        <label class="viz-label">mode separation Δ
          <input type="range" data-knob="delta" min="0.5" max="4.5" step="0.05" value="3.0">
          <span data-r="delta" style="min-width: 2.5em; font-family: var(--font-mono);">3.00</span>
        </label>
        <label class="viz-label">mode weight π
          <input type="range" data-knob="pi" min="0.05" max="0.95" step="0.01" value="0.50">
          <span data-r="pi" style="min-width: 2.5em; font-family: var(--font-mono);">0.50</span>
        </label>
        <label class="viz-label">mode width σ
          <input type="range" data-knob="sigma" min="0.4" max="1.6" step="0.05" value="1.0">
          <span data-r="sigma" style="min-width: 2.5em; font-family: var(--font-mono);">1.00</span>
        </label>
      </div>
      <div class="fr-fit__chart-target"></div>
    </div>
    <div class="fr-fit__pair">
      <div class="fr-fit__panel">
        <div class="fr-fit__panel-title">Forward KL fit — q minimizing D(p‖q)</div>
        <div class="fr-fit__chart-fwd"></div>
        <div class="fr-fit__readout" data-readout="fwd">—</div>
        <div class="fr-fit__caption">Mass-covering: q stretches to cover both modes.</div>
        <div class="fr-fit__buttons">
          <button class="viz-btn-sm" data-action="watch-fwd">Watch the fit</button>
        </div>
      </div>
      <div class="fr-fit__panel">
        <div class="fr-fit__panel-title">Reverse KL fit — q minimizing D(q‖p)</div>
        <div class="fr-fit__chart-rev"></div>
        <div class="fr-fit__readout" data-readout="rev">—</div>
        <div class="fr-fit__caption">Mode-seeking: q sits on one mode, ignores the other.</div>
        <div class="fr-fit__buttons">
          <button class="viz-btn-sm" data-action="watch-rev">Watch the fit</button>
          <button class="viz-btn-sm" data-action="other-mode">Try other init</button>
        </div>
        <div class="fr-fit__note" data-note></div>
      </div>
    </div>
  `;

  const knobs = ['delta', 'pi', 'sigma'].map(k => container.querySelector(`[data-knob="${k}"]`) as HTMLInputElement);
  const readouts = {
    delta: container.querySelector('[data-r="delta"]') as HTMLElement,
    pi:    container.querySelector('[data-r="pi"]') as HTMLElement,
    sigma: container.querySelector('[data-r="sigma"]') as HTMLElement,
  };
  const note = container.querySelector('[data-note]') as HTMLElement;

  function makeChart(host: HTMLElement, w = 380, h = 200) {
    const M = { top: 8, right: 8, bottom: 26, left: 30 };
    const innerW = w - M.left - M.right, innerH = h - M.top - M.bottom;
    const svg = d3.select(host).append('svg')
      .attr('viewBox', `0 0 ${w} ${h}`).style('width', '100%').style('max-width', `${w}px`);
    const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
    return { g, innerW, innerH };
  }

  const targetChart = makeChart(container.querySelector('.fr-fit__chart-target') as HTMLElement, 720, 180);
  const fwdChart = makeChart(container.querySelector('.fr-fit__chart-fwd') as HTMLElement);
  const revChart = makeChart(container.querySelector('.fr-fit__chart-rev') as HTMLElement);

  const xLow = -7, xHigh = 7;
  const grid = buildGrid(xLow, xHigh, 401);

  function redraw(animate: { fwd?: QParams[]; rev?: QParams[] } = {}) {
    target = {
      delta: +knobs[0].value,
      pi:    +knobs[1].value,
      sigma: +knobs[2].value,
    };
    readouts.delta.textContent = target.delta.toFixed(2);
    readouts.pi.textContent    = target.pi.toFixed(2);
    readouts.sigma.textContent = target.sigma.toFixed(2);

    // Auto-note when modes are close
    note.textContent = (target.delta < 1.5 * target.sigma)
      ? 'When the modes are close, forward and reverse KL agree.' : '';

    const xs = grid.xs;
    const pxs = xs.map(x => targetPdf(x, target));

    const fwdQ: QParams = (() => {
      const m = targetMoments(target);
      return { mu: m.mu, sigma: m.sigma };
    })();

    const revFinal = reverseKLOptimize(target,
      { mu: revInitMu, sigma: 1.5 }, xs, grid.dx, 60, 0.20);

    drawPanel(targetChart, xs, pxs, [], 0);
    if (animate.fwd) animateFit(fwdChart, xs, pxs, animate.fwd, target, 'fwd');
    else drawPanel(fwdChart, xs, pxs, xs.map(x => gaussianPdf(x, fwdQ.mu, fwdQ.sigma)), 0, fwdQ, 'fwd');

    if (animate.rev) animateFit(revChart, xs, pxs, animate.rev, target, 'rev');
    else drawPanel(revChart, xs, pxs, xs.map(x => gaussianPdf(x, revFinal.final.mu, revFinal.final.sigma)), 0, revFinal.final, 'rev');

    // Readouts
    const fwdKL = forwardKL(fwdQ, target, xs, grid.dx);
    const revKLval = reverseKL(revFinal.final, target, xs, grid.dx);
    (container.querySelector('[data-readout="fwd"]') as HTMLElement).innerHTML =
      `μ = <strong>${fwdQ.mu.toFixed(2)}</strong>, σ² = <strong>${(fwdQ.sigma ** 2).toFixed(2)}</strong> &nbsp;|&nbsp; D(p‖q) = <strong>${fwdKL.toFixed(2)}</strong>`;
    (container.querySelector('[data-readout="rev"]') as HTMLElement).innerHTML =
      `μ = <strong>${revFinal.final.mu.toFixed(2)}</strong>, σ² = <strong>${(revFinal.final.sigma ** 2).toFixed(2)}</strong> &nbsp;|&nbsp; D(q‖p) = <strong>${revKLval.toFixed(2)}</strong>`;
  }

  function drawPanel(
    chart: { g: d3.Selection<SVGGElement, unknown, null, undefined>; innerW: number; innerH: number },
    xs: number[], pxs: number[], qxs: number[],
    _step: number,
    q?: QParams, kind?: 'fwd' | 'rev',
  ) {
    chart.g.selectAll('*').remove();
    const xScale = d3.scaleLinear().domain([xLow, xHigh]).range([0, chart.innerW]);
    const yMax = Math.max(...pxs, ...(qxs.length ? qxs : [0])) * 1.1;
    const yScale = d3.scaleLinear().domain([0, yMax || 0.5]).range([chart.innerH, 0]);

    chart.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${chart.innerH})`).call(d3.axisBottom(xScale).ticks(7) as any);
    chart.g.append('g').attr('class', 'axis').call(d3.axisLeft(yScale).ticks(4) as any);

    // p
    const pLine = d3.line<number>().x((_d, i) => xScale(xs[i])).y(d => yScale(d));
    const pArea = d3.area<number>().x((_d, i) => xScale(xs[i])).y0(chart.innerH).y1(d => yScale(d));
    chart.g.append('path').attr('class', 'kl-curve--p-fill').attr('d', pArea(pxs)!);
    chart.g.append('path').attr('fill', 'none').attr('stroke', 'var(--dist-p)').attr('stroke-width', 1.5).attr('d', pLine(pxs)!);

    if (qxs.length) {
      const qArea = d3.area<number>().x((_d, i) => xScale(xs[i])).y0(chart.innerH).y1(d => yScale(d));
      const qLine = d3.line<number>().x((_d, i) => xScale(xs[i])).y(d => yScale(d));
      chart.g.append('path').attr('class', 'kl-curve--q-fill').attr('d', qArea(qxs)!);
      chart.g.append('path').attr('fill', 'none').attr('stroke', 'var(--dist-q)').attr('stroke-width', 2).attr('d', qLine(qxs)!);
    }

    if (q) {
      chart.g.append('text').attr('class', 'axis-label').attr('x', 8).attr('y', 14)
        .text(`${kind === 'fwd' ? 'q^fwd' : 'q^rev'}: μ=${q.mu.toFixed(2)}, σ=${q.sigma.toFixed(2)}`);
    }
  }

  function animateFit(
    chart: { g: d3.Selection<SVGGElement, unknown, null, undefined>; innerW: number; innerH: number },
    xs: number[], pxs: number[], history: QParams[], _t: TargetParams, kind: 'fwd' | 'rev',
  ) {
    let i = 0;
    const step = () => {
      const q = history[i];
      const qxs = xs.map(x => gaussianPdf(x, q.mu, q.sigma));
      drawPanel(chart, xs, pxs, qxs, i, q, kind);
      i++;
      if (i < history.length) setTimeout(step, 30);
    };
    step();
  }

  knobs.forEach(k => k.addEventListener('input', () => redraw()));
  container.querySelector('[data-action="watch-fwd"]')!.addEventListener('click', () => {
    const m = targetMoments(target);
    // synthesize a "trajectory" from a wide init to moment-match
    const steps = 40;
    const startMu = -1.5 + Math.random() * 3, startSigma = 0.5 + Math.random() * 1.5;
    const history: QParams[] = d3.range(steps + 1).map(s => ({
      mu:    startMu + (m.mu - startMu) * s / steps,
      sigma: startSigma + (m.sigma - startSigma) * s / steps,
    }));
    redraw({ fwd: history });
  });
  container.querySelector('[data-action="watch-rev"]')!.addEventListener('click', () => {
    const xs = grid.xs;
    const opt = reverseKLOptimize(target, { mu: revInitMu, sigma: 1.5 }, xs, grid.dx, 60, 0.20);
    redraw({ rev: opt.history });
  });
  container.querySelector('[data-action="other-mode"]')!.addEventListener('click', () => {
    revInitMu = -revInitMu;
    redraw();
  });

  redraw();
}
