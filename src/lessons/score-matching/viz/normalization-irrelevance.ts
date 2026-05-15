import * as d3 from 'd3';

const W = 300, H = 260, PAD = 30;

function renderPanel(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  title: string,
  valueFn: (x: number) => number,
  yLabel: string,
  color: string,
  range: [number, number],
) {
  svg.selectAll('*').remove();
  const xS = d3.scaleLinear([-3.5, 3.5], [PAD, W - 10]);
  const yS = d3.scaleLinear(range, [H - PAD, 20]);
  svg.attr('viewBox', `0 0 ${W} ${H}`);

  const xs = d3.range(-3.5, 3.5, 0.05);
  const line = d3.line<number>()
    .x(x => xS(x))
    .y(x => yS(valueFn(x)));

  svg.append('text')
    .attr('x', W / 2).attr('y', 14).attr('text-anchor', 'middle')
    .attr('font-size', '0.8rem').attr('fill', '#555').text(title);

  svg.append('path')
    .datum(xs)
    .attr('fill', 'none')
    .attr('stroke', color)
    .attr('stroke-width', 2)
    .attr('d', line);

  svg.append('g').attr('transform', `translate(0,${H-PAD})`).call(d3.axisBottom(xS).ticks(5))
    .selectAll('text').attr('font-size', '0.7rem');
  svg.append('g').attr('transform', `translate(${PAD},0)`).call(d3.axisLeft(yS).ticks(4))
    .selectAll('text').attr('font-size', '0.7rem');

  svg.append('text')
    .attr('x', W/2).attr('y', H - 4).attr('text-anchor', 'middle')
    .attr('font-size', '0.75rem').attr('fill', '#888').text(yLabel);
}

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Score is Invariant to Normalization</div>
      <div style="display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap;">
        <div style="flex:1;min-width:220px;">
          <label class="viz-label" style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.5rem;">
            Z = <span id="norm-z-val">1.0</span>
            <input type="range" id="norm-z" min="0.5" max="10" step="0.1" value="1" style="width:140px;">
          </label>
          <svg id="norm-density-svg" style="width:100%;max-width:300px;"></svg>
          <p style="font-size:0.8rem;color:var(--ink-soft);text-align:center;">p(x) = exp(-x²/2) / Z</p>
        </div>
        <div style="flex:1;min-width:220px;">
          <p style="font-size:0.85rem;font-weight:600;text-align:center;margin-bottom:0.5rem;">Score (unchanged)</p>
          <svg id="norm-score-svg" style="width:100%;max-width:300px;"></svg>
          <p style="font-size:0.8rem;color:var(--ink-soft);text-align:center;">s(x) = -x</p>
        </div>
      </div>
      <p class="viz-caption"><strong>Score is invariant to normalization.</strong>
      Drag the slider to change Z — the density rescales, the score stays still.</p>
    </div>
  `;

  const densitySvg = d3.select<SVGSVGElement, unknown>(container.querySelector('#norm-density-svg') as SVGSVGElement);
  const scoreSvg   = d3.select<SVGSVGElement, unknown>(container.querySelector('#norm-score-svg') as SVGSVGElement);
  const slider     = container.querySelector<HTMLInputElement>('#norm-z')!;
  const zLabel     = container.querySelector<HTMLElement>('#norm-z-val')!;

  function draw(Z: number) {
    renderPanel(densitySvg, `Density (Z = ${Z.toFixed(1)})`,
      x => Math.exp(-x * x / 2) / Z,
      'p(x)', '#b8651a', [0, 0.42]);
    renderPanel(scoreSvg, 'Score (always −x)',
      x => -x,
      's(x)', '#2c5f8d', [-3.6, 3.6]);
  }

  slider.addEventListener('input', () => {
    const Z = parseFloat(slider.value);
    zLabel.textContent = Z.toFixed(1);
    draw(Z);
  });

  draw(1.0);
}
