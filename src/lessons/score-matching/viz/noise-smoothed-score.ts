import * as d3 from 'd3';
import { scoreSmoothedGMM } from '../math/score';

const GRID_N = 18;
const MODES: [number, number][] = [[2, 0], [-2, 0]];
const DOMAIN: [number, number] = [-4.5, 4.5];

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Noise-Smoothed Score Field</div>
      <div class="viz-controls">
        <label class="viz-label">
          σ = <span id="nss-sigma-val">0.2</span>
          <input type="range" id="nss-sigma" min="-2" max="1.1" step="0.05" value="-1.6" style="width:160px;">
        </label>
        <span id="nss-level-label" style="font-family:var(--font-mono);font-size:0.8rem;color:var(--noise-level);"></span>
      </div>
      <svg id="nss-svg" style="width:100%;height:380px;display:block;"></svg>
      <div class="viz-caption">
        The score of the noise-smoothed GMM <em>p</em><sub>σ</sub>.
        At small σ the field is sharp and multimodal; at large σ it collapses to a single broad Gaussian.
      </div>
    </div>
  `;

  const svgEl    = container.querySelector<SVGSVGElement>('#nss-svg')!;
  const slider   = container.querySelector<HTMLInputElement>('#nss-sigma')!;
  const sigLabel = container.querySelector<HTMLElement>('#nss-sigma-val')!;
  const levLabel = container.querySelector<HTMLElement>('#nss-level-label')!;
  const svg      = d3.select(svgEl);

  const pis    = [0.5, 0.5];
  const mus    = MODES.map(m => Array.from(m));
  const Sigmas = MODES.map(() => [[0.2, 0], [0, 0.2]]);

  const STOPS = [
    { v: -2.0, label: '← data noise' },
    { v: -0.3, label: 'diffusion training' },
    { v:  1.1, label: 'prior noise →' },
  ];

  function getLevelLabel(logSigma: number): string {
    if (logSigma < -1.4) return '← data noise level';
    if (logSigma > 0.6)  return 'prior noise level →';
    return 'diffusion training range';
  }

  function render(logSigma: number) {
    const sigma = Math.exp(logSigma);
    sigLabel.textContent = sigma.toFixed(3);
    levLabel.textContent = getLevelLabel(logSigma);

    const W = svgEl.clientWidth || 600;
    const H = 380;
    const xS = d3.scaleLinear(DOMAIN, [40, W - 20]);
    const yS = d3.scaleLinear(DOMAIN, [H - 30, 20]);

    svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();

    const step = (DOMAIN[1] - DOMAIN[0]) / GRID_N;
    const arrows: { x: number; y: number; sx: number; sy: number }[] = [];
    let maxLen = 0;

    for (let i = 0; i <= GRID_N; i++) {
      for (let j = 0; j <= GRID_N; j++) {
        const gx = DOMAIN[0] + i * step;
        const gy = DOMAIN[0] + j * step;
        const s  = scoreSmoothedGMM([gx, gy], pis, mus, Sigmas, sigma);
        const len = Math.hypot(s[0], s[1]);
        if (len > maxLen) maxLen = len;
        arrows.push({ x: gx, y: gy, sx: s[0], sy: s[1] });
      }
    }

    const maxPx = (W * 0.055);
    const scale = maxLen > 0 ? maxPx / maxLen : 1;

    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'nss-arrow').attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('refX', 5).attr('refY', 3).attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L0,6 L6,3 z').attr('fill', '#2c5f8d').attr('opacity', 0.7);

    svg.selectAll('.nss-line')
      .data(arrows).join('line').attr('class', 'nss-line')
      .attr('x1', d => xS(d.x)).attr('y1', d => yS(d.y))
      .attr('x2', d => xS(d.x) + d.sx * scale).attr('y2', d => yS(d.y) - d.sy * scale)
      .attr('stroke', '#2c5f8d').attr('stroke-width', 1.3).attr('stroke-opacity', 0.65)
      .attr('marker-end', 'url(#nss-arrow)');

    // Mode markers
    svg.selectAll('.nss-mode')
      .data(MODES).join('circle').attr('class', 'nss-mode')
      .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
      .attr('r', 5).attr('fill', '#d4a437').attr('stroke', '#fff').attr('stroke-width', 1.5);

    // Sigma level indicators
    STOPS.forEach(s => {
      if (Math.abs(logSigma - s.v) < 0.45) {
        svg.append('text').attr('x', W/2).attr('y', H - 6)
          .attr('text-anchor', 'middle').attr('font-size', '0.75rem').attr('fill', '#6b3a8c')
          .text(s.label);
      }
    });

    svg.append('g').attr('transform', `translate(0,${H-30})`).call(d3.axisBottom(xS).ticks(5))
       .selectAll('text').attr('font-size', '0.7rem');
    svg.append('g').attr('transform', 'translate(40,0)').call(d3.axisLeft(yS).ticks(5))
       .selectAll('text').attr('font-size', '0.7rem');
  }

  slider.addEventListener('input', () => render(parseFloat(slider.value)));
  render(parseFloat(slider.value));
  window.addEventListener('resize', () => render(parseFloat(slider.value)));
}
