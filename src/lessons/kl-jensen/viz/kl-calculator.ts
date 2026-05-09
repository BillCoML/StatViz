import * as d3 from 'd3';
import { klDiscrete } from '../math/kl';

const N = 6;

interface KLCalculatorOptions {
  /** Add a banner across the top reminding the user KL ≥ 0. */
  showGibbsBanner?: boolean;
}

export function mountKLCalculator(container: HTMLElement, opts: KLCalculatorOptions = {}): void {
  let p = [0.1, 0.15, 0.25, 0.2, 0.18, 0.12];
  let q = [0.18, 0.18, 0.18, 0.18, 0.18, 0.10];

  const lowestSeen = { value: Infinity };

  container.classList.add('kl-calc');
  if (opts.showGibbsBanner) {
    const banner = document.createElement('div');
    banner.style.cssText = 'grid-column: 1 / -1; padding: 0.5rem 0.75rem; background: color-mix(in srgb, var(--sage) 12%, var(--paper)); border: 1px solid color-mix(in srgb, var(--sage) 40%, var(--rule)); border-radius: 6px; font-size: 0.9rem;';
    banner.innerHTML = `<strong>Verify Gibbs' inequality:</strong> this number is always ≥ 0. <span class="kl-calc__lowest" style="float: right; font-family: var(--font-mono); color: var(--ink-soft);"></span>`;
    container.appendChild(banner);
  }

  const panels = ['p', 'q'].map(which => {
    const div = document.createElement('div');
    div.className = `kl-calc__panel kl-calc__panel--${which}`;
    div.innerHTML = `
      <div class="kl-calc__title kl-calc__title--${which}">Distribution ${which}</div>
      <div class="kl-calc__chart"></div>
      <div class="kl-calc__sliders"></div>
    `;
    container.appendChild(div);
    return div;
  });

  const readout = document.createElement('div');
  readout.className = 'kl-calc__readouts';
  readout.style.gridColumn = '1 / -1';
  readout.innerHTML = `
    <div class="kl-calc__readout"><strong>D(p ‖ q):</strong> <span class="kl-pq">—</span></div>
    <div class="kl-calc__readout"><strong>D(q ‖ p):</strong> <span class="kl-qp">—</span></div>
    <div class="kl-calc__readout"><strong>ratio D(p‖q)/D(q‖p):</strong> <span class="kl-ratio">—</span></div>
    <div class="kl-calc__decomp" style="grid-column: 1 / -1;">
      <div style="font-family: var(--font-mono); font-size: 0.82em; color: var(--ink-soft); margin-bottom: 0.25em;">Per-outcome contribution to D(p ‖ q): pᵢ log(pᵢ/qᵢ)</div>
      <div class="kl-decomp"></div>
    </div>
    <div class="kl-calc__try" style="grid-column: 1 / -1;">
      <button class="viz-btn-sm" data-preset="zero-q-6">Try: set q(6) = 0 → ∞</button>
      <button class="viz-btn-sm" data-preset="reset">Reset</button>
    </div>
  `;
  container.appendChild(readout);

  const W = 280, H = 140, M = { top: 8, right: 8, bottom: 24, left: 26 };
  const innerW = W - M.left - M.right, innerH = H - M.top - M.bottom;

  function makeChart(host: HTMLElement, color: string) {
    const svg = d3.select(host).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`).style('width', '100%').style('max-width', `${W}px`);
    const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
    const xs = d3.scaleBand<number>().domain(d3.range(1, N + 1)).range([0, innerW]).padding(0.15);
    const ys = d3.scaleLinear().domain([0, 0.5]).range([innerH, 0]);
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).tickFormat(d => String(d)) as any);
    const yAxisG = g.append('g').attr('class', 'axis');
    const update = (vals: number[]) => {
      const max = Math.max(...vals, 0.5);
      ys.domain([0, max * 1.1]);
      yAxisG.call(d3.axisLeft(ys).ticks(4) as any);
      const sel = g.selectAll('.bar').data(vals);
      sel.enter().append('rect').attr('class', 'bar')
        .attr('x', (_d, i) => xs(i + 1)!)
        .attr('width', xs.bandwidth())
        .attr('fill', color)
        .merge(sel as any)
        .attr('y', d => ys(d as number))
        .attr('height', d => innerH - ys(d as number));
      sel.exit().remove();
    };
    return update;
  }

  const updateP = makeChart(panels[0].querySelector('.kl-calc__chart') as HTMLElement, 'var(--dist-p)');
  const updateQ = makeChart(panels[1].querySelector('.kl-calc__chart') as HTMLElement, 'var(--dist-q)');

  function makeSliders(host: HTMLElement, getVals: () => number[], setVals: (v: number[]) => void) {
    host.innerHTML = '';
    getVals().forEach((_v, i) => {
      const row = document.createElement('div');
      row.className = 'kl-calc__slider-row';
      row.innerHTML = `<span>${i + 1}</span><input type="range" min="0" max="1" step="0.01"><span class="value" style="min-width: 3em; text-align: right;"></span>`;
      const input = row.querySelector('input') as HTMLInputElement;
      const valueSpan = row.querySelector('.value') as HTMLSpanElement;
      input.value = String(getVals()[i]);
      valueSpan.textContent = (+input.value).toFixed(3);
      input.addEventListener('input', () => {
        const vals = getVals();
        vals[i] = +input.value;
        setVals(vals);
        valueSpan.textContent = (+input.value).toFixed(3);
        renormalizeAndRender();
      });
      host.appendChild(row);
    });
  }

  function renormalizeAndRender() {
    const sP = p.reduce((a, b) => a + b, 0);
    const sQ = q.reduce((a, b) => a + b, 0);
    const pn = sP > 0 ? p.map(x => x / sP) : Array(N).fill(1 / N);
    const qn = sQ > 0 ? q.map(x => x / sQ) : Array(N).fill(1 / N);
    updateP(pn); updateQ(qn);

    const dPQ = klDiscrete(pn, qn);
    const dQP = klDiscrete(qn, pn);
    (container.querySelector('.kl-pq') as HTMLElement).textContent = isFinite(dPQ) ? dPQ.toFixed(4) : '∞';
    (container.querySelector('.kl-qp') as HTMLElement).textContent = isFinite(dQP) ? dQP.toFixed(4) : '∞';
    const ratio = (isFinite(dPQ) && isFinite(dQP) && dQP > 1e-12) ? (dPQ / dQP).toFixed(3) : '—';
    (container.querySelector('.kl-ratio') as HTMLElement).textContent = ratio;

    if (isFinite(dPQ)) {
      lowestSeen.value = Math.min(lowestSeen.value, dPQ);
      const lowest = container.querySelector('.kl-calc__lowest') as HTMLElement | null;
      if (lowest) lowest.textContent = `lowest D(p‖q) seen: ${lowestSeen.value.toFixed(4)}`;
    }

    // Decomposition bars
    const decomp = container.querySelector('.kl-decomp') as HTMLElement;
    decomp.innerHTML = '';
    const contribs = pn.map((pi, i) => (pi === 0 || qn[i] === 0) ? 0 : pi * Math.log(pi / qn[i]));
    const maxAbs = Math.max(...contribs.map(Math.abs), 0.01);
    contribs.forEach((c, i) => {
      const inf = (qn[i] === 0 && pn[i] > 0);
      const sign = c >= 0 ? 'pos' : 'neg';
      const w = Math.min(95, Math.abs(c) / maxAbs * 50);
      const bar = document.createElement('div');
      bar.style.cssText = `display: flex; align-items: center; gap: 0.5em; font-family: var(--font-mono); font-size: 0.78em; margin: 0.15em 0;`;
      bar.innerHTML = `
        <span style="min-width: 1.5em; color: var(--ink-soft);">${i + 1}</span>
        <span style="display: inline-block; width: 50%; height: 14px; position: relative;">
          <span style="position: absolute; ${sign === 'pos' ? 'left: 50%' : `right: 50%`}; top: 0; bottom: 0; width: ${w}%;
            background: ${sign === 'pos' ? 'var(--dist-p)' : 'var(--dist-q)'};
            opacity: 0.7; border-radius: 2px;"></span>
          <span style="position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--ink-soft);"></span>
        </span>
        <span style="min-width: 6em;">${inf ? '+∞' : c.toFixed(4)}</span>
      `;
      decomp.appendChild(bar);
    });
  }

  makeSliders(panels[0].querySelector('.kl-calc__sliders') as HTMLElement, () => p, v => { p = v; });
  makeSliders(panels[1].querySelector('.kl-calc__sliders') as HTMLElement, () => q, v => { q = v; });

  container.querySelectorAll('button[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => {
      const preset = (btn as HTMLButtonElement).dataset.preset!;
      if (preset === 'zero-q-6') {
        q = [0.22, 0.22, 0.18, 0.2, 0.18, 0];
      } else {
        p = [0.1, 0.15, 0.25, 0.2, 0.18, 0.12];
        q = [0.18, 0.18, 0.18, 0.18, 0.18, 0.10];
        lowestSeen.value = Infinity;
      }
      makeSliders(panels[0].querySelector('.kl-calc__sliders') as HTMLElement, () => p, v => { p = v; });
      makeSliders(panels[1].querySelector('.kl-calc__sliders') as HTMLElement, () => q, v => { q = v; });
      renormalizeAndRender();
    });
  });

  renormalizeAndRender();
}
