/**
 * §3: Figure 2 recreation — directed graphical model x_T → ... → x_0 with overlaid
 * forward arrows q. Static layout.
 */
import { browserSched } from './_shared';

export function mount(container: HTMLElement): void {
  const T = browserSched.T;
  const positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0].map(f => Math.round(f * (T - 1)));
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">Graphical model — forward q and reverse p_θ</div>
      <svg id="gm-svg" width="800" height="240" viewBox="0 0 800 240" style="width:100%;height:auto;display:block;background:var(--paper);border-radius:6px;"></svg>
      <div class="viz-caption">Solid blue arrows: learned reverse process p_θ(x_{t−1} | x_t). Dashed sage arrows: fixed forward process q(x_t | x_{t−1}). Hover a node for its marginal.</div>
    </div>`;

  const svg = container.querySelector('#gm-svg')!;
  const cx = (i: number) => 80 + i * 130;
  const cy = 120;
  const r = 32;

  positions.forEach((t, i) => {
    const tx = cx(i);
    const ab = browserSched.alpha_bars[t];
    const sA = Math.sqrt(ab).toFixed(2);
    const sN = Math.sqrt(1 - ab).toFixed(2);
    // Reverse arrow (right → left)
    if (i < positions.length - 1) {
      const x1 = cx(i + 1) - r;
      const x2 = cx(i) + r;
      svg.insertAdjacentHTML('beforeend',
        `<line x1="${x1}" y1="${cy - 8}" x2="${x2}" y2="${cy - 8}" stroke="#2c5f8d" stroke-width="2" marker-end="url(#arr-blue)"/>`);
      svg.insertAdjacentHTML('beforeend',
        `<text x="${(x1+x2)/2}" y="${cy - 18}" text-anchor="middle" fill="#2c5f8d" font-family="JetBrains Mono" font-size="10">p_θ</text>`);
      // Forward dashed arrow (left → right, lower)
      const xa = cx(i) + r, xb = cx(i + 1) - r;
      svg.insertAdjacentHTML('beforeend',
        `<line x1="${xa}" y1="${cy + 8}" x2="${xb}" y2="${cy + 8}" stroke="#5a8a6a" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arr-sage)"/>`);
      svg.insertAdjacentHTML('beforeend',
        `<text x="${(xa+xb)/2}" y="${cy + 24}" text-anchor="middle" fill="#5a8a6a" font-family="JetBrains Mono" font-size="10">q</text>`);
    }
    // Node circle
    svg.insertAdjacentHTML('beforeend',
      `<circle cx="${tx}" cy="${cy}" r="${r}" fill="#fafaf6" stroke="#1c1c1c" stroke-width="1.5"/>
       <text x="${tx}" y="${cy + 4}" text-anchor="middle" font-family="Fraunces" font-size="14">x<tspan baseline-shift="sub" font-size="10">${t}</tspan></text>
       <title>t=${t}: √ᾱ=${sA}, √(1−ᾱ)=${sN}</title>`);
  });

  svg.insertAdjacentHTML('afterbegin',
    `<defs>
      <marker id="arr-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L10,5 L0,10 z" fill="#2c5f8d"/>
      </marker>
      <marker id="arr-sage" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L10,5 L0,10 z" fill="#5a8a6a"/>
      </marker>
    </defs>`);
}
