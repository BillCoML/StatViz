import * as d3 from 'd3';
import { renderMath } from '@shared/ui';

const SIGMA_DEFAULT = 0.3;
const DATA_PT: [number, number] = [0, 0];
const DOMAIN: [number, number] = [-3, 3];

// Arrow length is capped at 80px and scales with displacement magnitude
export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container" style="max-width:480px;">
      <div class="viz-title">DSM Target Illustration</div>
      <div class="viz-controls">
        <label class="viz-label">σ = <span id="dsmt-sigma-val">${SIGMA_DEFAULT}</span>
          <input type="range" id="dsmt-sigma" min="0.05" max="1.5" step="0.05" value="${SIGMA_DEFAULT}" style="width:120px;">
        </label>
      </div>
      <svg id="dsmt-svg" style="width:100%;max-width:420px;height:320px;display:block;cursor:crosshair;"></svg>
      <div class="viz-caption" id="dsmt-caption">Click anywhere to place the noisy point $\\tilde{x}$.</div>
    </div>
  `;

  renderMath(container.querySelector('#dsmt-caption')!);

  const svgEl    = container.querySelector<SVGSVGElement>('#dsmt-svg')!;
  const slider   = container.querySelector<HTMLInputElement>('#dsmt-sigma')!;
  const sigLabel = container.querySelector<HTMLElement>('#dsmt-sigma-val')!;
  const caption  = container.querySelector<HTMLElement>('#dsmt-caption')!;
  const svg      = d3.select(svgEl);

  let noisyPt: [number, number] | null = null;
  let sigma = SIGMA_DEFAULT;

  const W = 420, H = 320;
  const xS = d3.scaleLinear(DOMAIN, [40, W - 20]);
  const yS = d3.scaleLinear(DOMAIN, [H - 30, 20]);

  function render() {
    svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();

    // Background grid
    svg.append('g').attr('transform', `translate(0,${H-30})`).call(d3.axisBottom(xS).ticks(5))
       .selectAll('text').attr('font-size', '0.7rem');
    svg.append('g').attr('transform', 'translate(40,0)').call(d3.axisLeft(yS).ticks(5))
       .selectAll('text').attr('font-size', '0.7rem');

    // Data point x (clean)
    svg.append('circle')
      .attr('cx', xS(DATA_PT[0])).attr('cy', yS(DATA_PT[1]))
      .attr('r', 9).attr('fill', '#b8651a').attr('stroke', '#fff').attr('stroke-width', 2);
    svg.append('text')
      .attr('x', xS(DATA_PT[0]) + 14).attr('y', yS(DATA_PT[1]) + 4)
      .attr('font-family', 'var(--font-mono)').attr('font-size', '0.8rem').text('x (clean)');

    if (!noisyPt) return;

    const [nx, ny] = noisyPt;
    const displacement: [number, number] = [nx - DATA_PT[0], ny - DATA_PT[1]];
    const sigma2 = sigma * sigma;
    const target: [number, number] = [-displacement[0] / sigma2, -displacement[1] / sigma2];
    const tLen = Math.hypot(target[0], target[1]);
    const tNorm = tLen > 0 ? [target[0] / tLen, target[1] / tLen] : [0, 0];
    const arrowPixels = Math.min(tLen * 30, 80);  // cap pixel length

    const arrowEnd: [number, number] = [
      xS(nx) + tNorm[0] * arrowPixels,
      yS(ny) - tNorm[1] * arrowPixels,
    ];

    // Dashed line from clean to noisy
    svg.append('line')
      .attr('x1', xS(DATA_PT[0])).attr('y1', yS(DATA_PT[1]))
      .attr('x2', xS(nx)).attr('y2', yS(ny))
      .attr('stroke', '#aaa').attr('stroke-width', 1.5).attr('stroke-dasharray', '5,4');

    // Noisy point
    svg.append('circle')
      .attr('cx', xS(nx)).attr('cy', yS(ny))
      .attr('r', 7).attr('fill', '#2c5f8d').attr('stroke', '#fff').attr('stroke-width', 2);
    svg.append('text')
      .attr('x', xS(nx) + 12).attr('y', yS(ny) + 4)
      .attr('font-family', 'var(--font-mono)').attr('font-size', '0.8rem')
      .attr('fill', '#2c5f8d').text('x̃ (noisy)');

    // DSM target arrow
    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'dsmt-head').attr('markerWidth', 7).attr('markerHeight', 7)
      .attr('refX', 6).attr('refY', 3.5).attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L0,7 L7,3.5 z').attr('fill', '#c87f3b');

    svg.append('line')
      .attr('x1', xS(nx)).attr('y1', yS(ny))
      .attr('x2', arrowEnd[0]).attr('y2', arrowEnd[1])
      .attr('stroke', '#c87f3b').attr('stroke-width', 2.5)
      .attr('marker-end', 'url(#dsmt-head)');

    svg.append('text')
      .attr('x', arrowEnd[0] + 6).attr('y', arrowEnd[1] + 4)
      .attr('font-family', 'var(--font-mono)').attr('font-size', '0.78rem').attr('fill', '#c87f3b')
      .text('−(x̃−x)/σ²');

    caption.innerHTML = `
      $\\tilde{x} = (${nx.toFixed(2)},\\, ${ny.toFixed(2)})$,
      $\\sigma = ${sigma.toFixed(2)}$,
      DSM target $= (${target[0].toFixed(2)},\\, ${target[1].toFixed(2)})$
    `;
    renderMath(caption);
  }

  svgEl.addEventListener('click', (evt) => {
    const rect = svgEl.getBoundingClientRect();
    const scaleX = W / rect.width;
    const scaleY = H / rect.height;
    const px = (evt.clientX - rect.left) * scaleX;
    const py = (evt.clientY - rect.top)  * scaleY;
    noisyPt = [xS.invert(px), yS.invert(py)];
    render();
  });

  slider.addEventListener('input', () => {
    sigma = parseFloat(slider.value);
    sigLabel.textContent = sigma.toFixed(2);
    if (noisyPt) render();
  });

  render();
}
