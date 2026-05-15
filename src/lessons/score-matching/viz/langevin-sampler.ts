import * as d3 from 'd3';
import { scoreGaussian, scoreGMM, scoreRing } from '../math/score';
import { langevinStep } from '../math/langevin';

const DOMAIN: [number, number] = [-5, 5];

type DistId = 'gaussian' | 'mixture' | 'ring';

interface Dist {
  label: string;
  scoreFn: (x: number[]) => number[];
}

const DISTS: Record<DistId, Dist> = {
  gaussian: {
    label: '2D Gaussian N(0, I)',
    scoreFn: x => scoreGaussian(x, [0, 0], [[1, 0], [0, 1]]),
  },
  mixture: {
    label: 'Mixture (2 modes)',
    scoreFn: x => scoreGMM(x, [0.5, 0.5], [[2, 0], [-2, 0]],
      [[[0.4, 0], [0, 0.4]], [[0.4, 0], [0, 0.4]]]),
  },
  ring: {
    label: 'Ring distribution',
    scoreFn: x => scoreRing(x),
  },
};

function seededRng(seed: number): () => number {
  let s = seed;
  return () => { s = (Math.imul(1664525, s) + 1013904223) | 0; return (s >>> 0) / 0x100000000; };
}

function boxMuller(rng: () => number): number {
  let u = rng(); while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

const MAX_TRAIL = 20;

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Langevin Sampler (Analytical Score)</div>
      <div class="viz-controls">
        <label class="viz-label">Distribution:
          <select id="ls-dist">
            <option value="gaussian">2D Gaussian</option>
            <option value="mixture">Gaussian mixture (2 modes)</option>
            <option value="ring">Ring distribution</option>
          </select>
        </label>
        <label class="viz-label">N particles:
          <select id="ls-nparts">
            <option value="1">1</option>
            <option value="10">10</option>
            <option value="100" selected>100</option>
          </select>
        </label>
        <label class="viz-label">η = <span id="ls-eta-val">0.05</span>
          <input type="range" id="ls-eta" min="0.005" max="0.2" step="0.005" value="0.05" style="width:100px;">
        </label>
      </div>
      <svg id="ls-svg" style="width:100%;height:380px;display:block;"></svg>
      <div class="viz-controls" style="margin-top:0.5rem;">
        <button class="viz-btn" id="ls-step">Step</button>
        <button class="viz-btn" id="ls-play">Play</button>
        <button class="viz-btn" id="ls-pause" disabled>Pause</button>
        <button class="viz-btn" id="ls-reset">Reset</button>
        <span id="ls-step-count" style="font-family:var(--font-mono);font-size:0.82rem;color:var(--ink-soft);">step 0</span>
      </div>
      <div class="viz-caption">Particles initialized at N(0, 9I). Drift toward high-density; noise prevents mode collapse.</div>
    </div>
  `;

  const svgEl    = container.querySelector<SVGSVGElement>('#ls-svg')!;
  const distSel  = container.querySelector<HTMLSelectElement>('#ls-dist')!;
  const npartsSel = container.querySelector<HTMLSelectElement>('#ls-nparts')!;
  const etaSlider = container.querySelector<HTMLInputElement>('#ls-eta')!;
  const etaLabel  = container.querySelector<HTMLElement>('#ls-eta-val')!;
  const stepBtn   = container.querySelector<HTMLButtonElement>('#ls-step')!;
  const playBtn   = container.querySelector<HTMLButtonElement>('#ls-play')!;
  const pauseBtn  = container.querySelector<HTMLButtonElement>('#ls-pause')!;
  const resetBtn  = container.querySelector<HTMLButtonElement>('#ls-reset')!;
  const stepCount = container.querySelector<HTMLElement>('#ls-step-count')!;
  const svg       = d3.select(svgEl);

  const W = svgEl.clientWidth || 600;
  const H = 380;
  const xS = d3.scaleLinear(DOMAIN, [40, W - 20]);
  const yS = d3.scaleLinear(DOMAIN, [H - 30, 20]);

  let rng = seededRng(42);
  let particles: number[][] = [];
  let trails: number[][][] = [];
  let stepN = 0;
  let animId: number | null = null;
  let activeDist: DistId = 'gaussian';

  function initParticles() {
    const n = parseInt(npartsSel.value);
    rng = seededRng(42);
    particles = Array.from({ length: n }, () => [
      3 * boxMuller(rng), 3 * boxMuller(rng),
    ]);
    trails = particles.map(p => [[p[0], p[1]]]);
    stepN = 0;
  }

  function doStep() {
    const eta = parseFloat(etaSlider.value);
    const dist = DISTS[activeDist];
    particles = particles.map((p, i) => {
      const score = dist.scoreFn(p);
      const eps   = [boxMuller(rng), boxMuller(rng)];
      const next  = langevinStep(p, score, eta, eps);
      trails[i].push([next[0], next[1]]);
      if (trails[i].length > MAX_TRAIL) trails[i].shift();
      return next;
    });
    stepN++;
    stepCount.textContent = `step ${stepN}`;
    draw();
  }

  function draw() {
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${W} ${H}`);

    // Trail lines
    for (let i = 0; i < particles.length; i++) {
      const trail = trails[i];
      if (trail.length < 2) continue;
      for (let j = 1; j < trail.length; j++) {
        const opacity = (j / trail.length) * 0.5;
        svg.append('line')
          .attr('x1', xS(trail[j-1][0])).attr('y1', yS(trail[j-1][1]))
          .attr('x2', xS(trail[j][0])).attr('y2', yS(trail[j][1]))
          .attr('stroke', '#5a8a6a').attr('stroke-width', 1.2)
          .attr('stroke-opacity', opacity);
      }
    }

    // Particles
    svg.selectAll('.ls-particle')
      .data(particles).join('circle').attr('class', 'ls-particle')
      .attr('cx', d => xS(d[0])).attr('cy', d => yS(d[1]))
      .attr('r', particles.length === 1 ? 6 : 3)
      .attr('fill', '#b8651a').attr('fill-opacity', 0.8);

    svg.append('g').attr('transform', `translate(0,${H-30})`).call(d3.axisBottom(xS).ticks(5))
       .selectAll('text').attr('font-size', '0.7rem');
    svg.append('g').attr('transform', 'translate(40,0)').call(d3.axisLeft(yS).ticks(5))
       .selectAll('text').attr('font-size', '0.7rem');
  }

  function stopAnim() {
    if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
    playBtn.disabled = false;
    pauseBtn.disabled = true;
  }

  function startAnim() {
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    const loop = () => { doStep(); animId = requestAnimationFrame(loop); };
    animId = requestAnimationFrame(loop);
  }

  stepBtn.addEventListener('click', () => { stopAnim(); doStep(); });
  playBtn.addEventListener('click', startAnim);
  pauseBtn.addEventListener('click', stopAnim);
  resetBtn.addEventListener('click', () => { stopAnim(); initParticles(); draw(); });
  distSel.addEventListener('change', () => { activeDist = distSel.value as DistId; stopAnim(); initParticles(); draw(); });
  npartsSel.addEventListener('change', () => { stopAnim(); initParticles(); draw(); });
  etaSlider.addEventListener('input', () => { etaLabel.textContent = parseFloat(etaSlider.value).toFixed(3); });

  initParticles();
  draw();
}
