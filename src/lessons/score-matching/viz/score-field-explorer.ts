import * as d3 from 'd3';
import { scoreGaussian, scoreGMM, scoreBanana, scoreRing } from '../math/score';
import { Matrix, inverse } from 'ml-matrix';

// ── Distributions ─────────────────────────────────────────────────────────────

type DistId = 'normal' | 'gaussian' | 'gmm' | 'banana' | 'ring';

interface DistInfo {
  label: string;
  scoreFn: (x: [number, number]) => [number, number];
  logpFn:  (x: [number, number]) => number;
  domain:  [number, number];
}

function makeGaussianDist(mu: [number, number], Sigma: number[][]): DistInfo {
  const inv = inverse(new Matrix(Sigma));
  return {
    label: 'General Gaussian',
    scoreFn: ([x0, x1]) => {
      const s = scoreGaussian([x0, x1], mu, Sigma);
      return [s[0], s[1]];
    },
    logpFn: ([x0, x1]) => {
      const diff = Matrix.columnVector([x0 - mu[0], x1 - mu[1]]);
      return -0.5 * diff.transpose().mmul(inv).mmul(diff).get(0, 0);
    },
    domain: [-4, 4],
  };
}

function makeDists(): Record<DistId, DistInfo> {
  return {
    normal: {
      label: 'Standard normal N(0, I)',
      scoreFn: ([x, y]) => [-x, -y],
      logpFn:  ([x, y]) => -0.5 * (x*x + y*y),
      domain: [-3.5, 3.5],
    },
    gaussian: makeGaussianDist([0.5, -0.3], [[1.5, 0.8], [0.8, 0.7]]),
    gmm: {
      label: 'Gaussian mixture (2 modes)',
      scoreFn: ([x, y]) => {
        const s = scoreGMM([x, y], [0.5, 0.5], [[2, 0], [-2, 0]],
          [[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]]);
        return [s[0], s[1]];
      },
      logpFn: ([x, y]) => {
        const l1 = -((x-2)**2 + y**2) - Math.log(2 * Math.PI);
        const l2 = -((x+2)**2 + y**2) - Math.log(2 * Math.PI);
        const m = Math.max(l1, l2);
        return m + Math.log(0.5 * Math.exp(l1 - m) + 0.5 * Math.exp(l2 - m));
      },
      domain: [-4.5, 4.5],
    },
    banana: {
      label: 'Banana (Rosenbrock)',
      scoreFn: ([x, y]) => { const s = scoreBanana([x, y]); return [s[0], s[1]]; },
      logpFn:  ([x, y]) => -((1 - x)**2 + 10 * (y - x*x)**2),
      domain: [-2.5, 2.5],
    },
    ring: {
      label: 'Ring distribution',
      scoreFn: ([x, y]) => { const s = scoreRing([x, y]); return [s[0], s[1]]; },
      logpFn:  ([x, y]) => { const r = Math.hypot(x, y); return -0.5 * ((r - 2) / 0.3)**2; },
      domain: [-3.5, 3.5],
    },
  };
}

// ── Main component ────────────────────────────────────────────────────────────

const GRID_N = 20;

export function mount(container: HTMLElement): void {
  const dists  = makeDists();
  let activeDist: DistId = 'normal';

  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Score Field Explorer</div>
      <div class="viz-controls">
        <label class="viz-label">Distribution:
          <select id="sf-dist-select">
            <option value="normal">Standard normal</option>
            <option value="gaussian">General Gaussian</option>
            <option value="gmm">Gaussian mixture (2 modes)</option>
            <option value="banana">Banana / Rosenbrock</option>
            <option value="ring">Ring distribution</option>
          </select>
        </label>
      </div>
      <svg id="sf-svg" style="width:100%;height:420px;display:block;"></svg>
      <div class="viz-caption" id="sf-tooltip">Hover an arrow to see x, s(x), log p(x)</div>
    </div>
  `;

  const select  = container.querySelector<HTMLSelectElement>('#sf-dist-select')!;
  const tooltip = container.querySelector<HTMLElement>('#sf-tooltip')!;
  const svgEl   = container.querySelector<SVGSVGElement>('#sf-svg')!;
  const svg     = d3.select(svgEl);

  function render() {
    const dist = dists[activeDist];
    const W = svgEl.clientWidth || 600;
    const H = 420;
    const [lo, hi] = dist.domain;
    const xScale = d3.scaleLinear([lo, hi], [40, W - 20]);
    const yScale = d3.scaleLinear([lo, hi], [H - 30, 20]);

    svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();

    // Background density heatmap
    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = W; imgCanvas.height = H;
    const ctx = imgCanvas.getContext('2d')!;
    const imgData = ctx.createImageData(W, H);
    const logPs: number[] = [];
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const x: [number, number] = [xScale.invert(px), yScale.invert(py)];
        logPs.push(dist.logpFn(x));
      }
    }
    const [lpMin, lpMax] = d3.extent(logPs) as [number, number];
    const lpRange = lpMax - lpMin || 1;
    for (let i = 0; i < logPs.length; i++) {
      const t = (logPs[i] - lpMin) / lpRange;
      const v = Math.round(255 - t * 60);
      imgData.data[i*4+0] = v;
      imgData.data[i*4+1] = v;
      imgData.data[i*4+2] = Math.round(v * 0.97);
      imgData.data[i*4+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    svg.append('image')
      .attr('x', 0).attr('y', 0).attr('width', W).attr('height', H)
      .attr('href', imgCanvas.toDataURL());

    // Compute score vectors on a 20×20 grid
    const step = (hi - lo) / GRID_N;
    const arrows: { x: number; y: number; sx: number; sy: number }[] = [];
    let maxLen = 0;
    for (let i = 0; i <= GRID_N; i++) {
      for (let j = 0; j <= GRID_N; j++) {
        const gx = lo + i * step;
        const gy = lo + j * step;
        const [sx, sy] = dist.scoreFn([gx, gy]);
        const len = Math.hypot(sx, sy);
        if (len > maxLen) maxLen = len;
        arrows.push({ x: gx, y: gy, sx, sy });
      }
    }

    // Scale so longest arrow is 5% of canvas width
    const maxPixels = W * 0.05;
    const scale = maxLen > 0 ? maxPixels / maxLen : 1;

    const g = svg.append('g');
    const defs = svg.append('defs');
    defs.append('marker')
      .attr('id', 'sf-arrowhead')
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('refX', 5).attr('refY', 3)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L0,6 L6,3 z')
      .attr('fill', '#2c5f8d').attr('opacity', 0.75);

    g.selectAll('.sf-arrow')
      .data(arrows)
      .join('line')
        .attr('class', 'sf-arrow')
        .attr('x1', d => xScale(d.x))
        .attr('y1', d => yScale(d.y))
        .attr('x2', d => xScale(d.x) + d.sx * scale)
        .attr('y2', d => yScale(d.y) - d.sy * scale)
        .attr('stroke', '#2c5f8d')
        .attr('stroke-width', 1.2)
        .attr('stroke-opacity', 0.7)
        .attr('marker-end', 'url(#sf-arrowhead)')
        .on('mouseover', function(_event, d) {
          d3.select(this).attr('stroke', '#c87f3b').attr('stroke-opacity', 1);
          const lp = dist.logpFn([d.x, d.y]);
          tooltip.textContent =
            `x = (${d.x.toFixed(2)}, ${d.y.toFixed(2)})  ` +
            `s(x) = (${d.sx.toFixed(3)}, ${d.sy.toFixed(3)})  ` +
            `log p(x) = ${lp.toFixed(3)}`;
        })
        .on('mouseout', function() {
          d3.select(this).attr('stroke', '#2c5f8d').attr('stroke-opacity', 0.7);
          tooltip.textContent = 'Hover an arrow to see x, s(x), log p(x)';
        });

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);
    svg.append('g').attr('transform', `translate(0,${H-30})`).call(xAxis)
       .selectAll('text').attr('font-size', '0.75rem');
    svg.append('g').attr('transform', 'translate(40,0)').call(yAxis)
       .selectAll('text').attr('font-size', '0.75rem');
  }

  select.addEventListener('change', () => {
    activeDist = select.value as DistId;
    render();
  });

  render();
  window.addEventListener('resize', render);
}
