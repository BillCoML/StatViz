import * as d3 from 'd3';

interface FunctionDef {
  label: string;
  fn: (x: number) => number;
  domain: [number, number];
  kind: 'convex' | 'concave' | 'mixed';
  /** restrict draggable x range to keep things interesting */
  xRange: [number, number];
}

const FUNCS: Record<string, FunctionDef> = {
  'x^2':     { label: 'φ(x) = x²',      fn: x => x * x,       domain: [-2.5, 2.5], kind: 'convex',  xRange: [-2, 2]  },
  'e^x':     { label: 'φ(x) = eˣ',      fn: x => Math.exp(x), domain: [-2, 2],     kind: 'convex',  xRange: [-1.6, 1.6] },
  '-log(x)': { label: 'φ(x) = −log x',  fn: x => -Math.log(x), domain: [0.05, 4],  kind: 'convex',  xRange: [0.2, 3.5] },
  'log(x)':  { label: 'φ(x) = log x',   fn: x => Math.log(x), domain: [0.05, 4],   kind: 'concave', xRange: [0.2, 3.5] },
  '|x|':     { label: 'φ(x) = |x|',     fn: x => Math.abs(x), domain: [-2.5, 2.5], kind: 'convex',  xRange: [-2, 2]  },
  'x^3':     { label: 'φ(x) = x³',      fn: x => x * x * x,   domain: [-2, 2],     kind: 'mixed',   xRange: [-1.7, 1.7] },
};

export function mountConvexExplorer(container: HTMLElement): void {
  let key: keyof typeof FUNCS = 'x^2';

  container.classList.add('convex-explorer');
  container.innerHTML = `
    <div class="convex-explorer__pick">
      <span style="font-family: var(--font-mono); font-size: 0.85em; color: var(--ink-soft);">Choose φ:</span>
      ${Object.keys(FUNCS).map(k =>
        `<button class="viz-btn-sm" data-key="${k}">${FUNCS[k].label}</button>`).join('')}
      <span class="convex-explorer__badge"></span>
    </div>
    <div class="convex-explorer__chart"></div>
    <div class="convex-explorer__sliders viz-controls">
      <label class="viz-label">x₁ <input type="range" data-knob="x1"></label>
      <label class="viz-label">x₂ <input type="range" data-knob="x2"></label>
      <label class="viz-label">λ <input type="range" data-knob="lam" min="0" max="1" step="0.01"></label>
    </div>
    <div class="convex-explorer__readout" style="font-family: var(--font-mono); font-size: 0.88em; color: var(--ink-soft);"></div>
  `;

  const chart = container.querySelector('.convex-explorer__chart') as HTMLElement;
  const badge = container.querySelector('.convex-explorer__badge') as HTMLElement;
  const readout = container.querySelector('.convex-explorer__readout') as HTMLElement;
  const knobs = {
    x1:  container.querySelector('[data-knob="x1"]') as HTMLInputElement,
    x2:  container.querySelector('[data-knob="x2"]') as HTMLInputElement,
    lam: container.querySelector('[data-knob="lam"]') as HTMLInputElement,
  };

  const W = 560, H = 280, M = { top: 20, right: 20, bottom: 30, left: 40 };
  const svg = d3.select(chart).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .style('width', '100%').style('height', 'auto')
    .style('max-width', `${W}px`);
  const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;

  const xAxisG = g.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`);
  const yAxisG = g.append('g').attr('class', 'axis');
  const curve = g.append('path').attr('fill', 'none').attr('stroke', 'var(--ink)').attr('stroke-width', 2);
  const chordLine = g.append('line').attr('stroke', 'var(--amber)').attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');
  const gapLine = g.append('line').attr('stroke', 'var(--warning)').attr('stroke-width', 2);
  const ptCurve = g.append('circle').attr('r', 5).attr('fill', 'var(--ink)');
  const ptChord = g.append('circle').attr('r', 5).attr('fill', 'var(--amber)');
  const ptX1 = g.append('circle').attr('r', 4).attr('fill', 'var(--dist-p)');
  const ptX2 = g.append('circle').attr('r', 4).attr('fill', 'var(--dist-q)');
  const gapText = g.append('text').attr('class', 'axis-label').attr('text-anchor', 'middle');

  function configure() {
    const f = FUNCS[key];
    knobs.x1.min = String(f.xRange[0]); knobs.x1.max = String(f.xRange[1]); knobs.x1.step = '0.01';
    knobs.x2.min = String(f.xRange[0]); knobs.x2.max = String(f.xRange[1]); knobs.x2.step = '0.01';
    if (key === 'log(x)' || key === '-log(x)') {
      knobs.x1.value = '0.6'; knobs.x2.value = '3.0';
    } else {
      knobs.x1.value = String(f.xRange[0] + (f.xRange[1] - f.xRange[0]) * 0.2);
      knobs.x2.value = String(f.xRange[0] + (f.xRange[1] - f.xRange[0]) * 0.8);
    }
    knobs.lam.value = '0.5';
    const tag = f.kind === 'convex' ? 'convex' : f.kind === 'concave' ? 'concave' : 'neither';
    badge.className = `convex-explorer__badge convex-explorer__badge--${tag === 'neither' ? '' : tag}`;
    badge.textContent = tag;
  }

  function render() {
    const f = FUNCS[key];
    const x1 = +knobs.x1.value, x2 = +knobs.x2.value, lam = +knobs.lam.value;
    const xLam = lam * x1 + (1 - lam) * x2;
    const yCurveLam = f.fn(xLam);
    const yChordLam = lam * f.fn(x1) + (1 - lam) * f.fn(x2);

    const samples = d3.range(200).map(i => f.domain[0] + (f.domain[1] - f.domain[0]) * i / 199);
    const ys = samples.map(f.fn);
    const yMin = d3.min(ys)!, yMax = d3.max(ys)!;
    const pad = (yMax - yMin) * 0.1 || 1;

    const xs = d3.scaleLinear().domain(f.domain).range([0, innerW]);
    const ysc = d3.scaleLinear().domain([yMin - pad, yMax + pad]).range([innerH, 0]);

    xAxisG.call(d3.axisBottom(xs).ticks(6) as any);
    yAxisG.call(d3.axisLeft(ysc).ticks(5) as any);

    const lineGen = d3.line<number>().x(d => xs(d)).y(d => ysc(f.fn(d)));
    curve.attr('d', lineGen(samples));

    chordLine.attr('x1', xs(x1)).attr('y1', ysc(f.fn(x1)))
             .attr('x2', xs(x2)).attr('y2', ysc(f.fn(x2)));
    gapLine.attr('x1', xs(xLam)).attr('y1', ysc(yChordLam))
           .attr('x2', xs(xLam)).attr('y2', ysc(yCurveLam));
    ptX1.attr('cx', xs(x1)).attr('cy', ysc(f.fn(x1)));
    ptX2.attr('cx', xs(x2)).attr('cy', ysc(f.fn(x2)));
    ptCurve.attr('cx', xs(xLam)).attr('cy', ysc(yCurveLam));
    ptChord.attr('cx', xs(xLam)).attr('cy', ysc(yChordLam));

    const gap = yChordLam - yCurveLam;
    gapText.attr('x', xs(xLam) + 8)
           .attr('y', ysc((yCurveLam + yChordLam) / 2))
           .text(`gap = ${gap.toFixed(3)}`);

    readout.innerHTML = `φ(λx₁+(1−λ)x₂) = <strong>${yCurveLam.toFixed(3)}</strong> &nbsp;|&nbsp; λφ(x₁)+(1−λ)φ(x₂) = <strong>${yChordLam.toFixed(3)}</strong>`;
  }

  container.querySelectorAll('button[data-key]').forEach(btn => {
    btn.addEventListener('click', () => {
      key = (btn as HTMLButtonElement).dataset.key as keyof typeof FUNCS;
      configure();
      render();
    });
  });
  Object.values(knobs).forEach(k => k.addEventListener('input', render));

  configure();
  render();
}
