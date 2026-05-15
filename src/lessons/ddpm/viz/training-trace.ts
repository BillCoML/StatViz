/**
 * §8: six-panel storyboard of one training step.
 */
import { browserSched, mkRng, gauss2, DATA_CENTERS } from './_shared';
import { forwardSample } from '../math/forward-process';

const T = browserSched.T;

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container">
      <div class="viz-title">One training step, six panels</div>
      <div id="tt-grid" style="display:grid;grid-template-columns:repeat(3, 1fr);gap:0.5rem;"></div>
      <div class="viz-controls" style="margin-top:0.6rem;">
        <button class="viz-btn" id="tt-step">New step</button>
        <span class="ddpm-readout" id="tt-readout"></span>
      </div>
      <div class="viz-caption">Each click samples a new (x₀, t, ε), runs the closed-form jump to x_t, and shows the loss.</div>
    </div>`;

  const grid = container.querySelector('#tt-grid') as HTMLElement;
  const rng = mkRng(Math.floor(Math.random() * 1e6));

  function step() {
    const cIdx = Math.floor(rng() * 4);
    const c = DATA_CENTERS[cIdx];
    const jit = gauss2(rng); const x0 = [c[0] + 0.2*jit[0], c[1] + 0.2*jit[1]];
    const t = Math.floor(rng() * T);
    const eps = gauss2(rng);
    const xt = forwardSample(x0, t, eps, browserSched);
    const ab = browserSched.alpha_bars[t];
    const sA = Math.sqrt(ab), sN = Math.sqrt(1-ab);

    grid.innerHTML = `
      ${panel('1. Sample x_0', `(${x0[0].toFixed(2)}, ${x0[1].toFixed(2)})`, 'cluster '+cIdx)}
      ${panel('2. Sample t', `t = ${t}`, `√ᾱ = ${sA.toFixed(3)}, √(1−ᾱ) = ${sN.toFixed(3)}`)}
      ${panel('3. Sample ε', `(${eps[0].toFixed(2)}, ${eps[1].toFixed(2)})`, '∼ 𝒩(0, I)')}
      ${panel('4. Closed-form jump', `x_t = (${xt[0].toFixed(2)}, ${xt[1].toFixed(2)})`, '= √ᾱ·x_0 + √(1−ᾱ)·ε')}
      ${panel('5. Network forward', `ε_θ(x_t, t) ≈ ε`, 'target: predict the noise')}
      ${panel('6. Loss + backprop', `‖ε − ε_θ‖²`, 'MSE on noise prediction')}`;

    (container.querySelector('#tt-readout') as HTMLElement).textContent =
      `Step: x_0=(${x0[0].toFixed(2)},${x0[1].toFixed(2)}) t=${t} → x_t=(${xt[0].toFixed(2)},${xt[1].toFixed(2)})`;
  }

  function panel(title: string, big: string, sub: string) {
    return `<div style="background:var(--paper-darker,#efece2);padding:0.5rem;border-radius:6px;">
      <div style="font-family:JetBrains Mono;font-size:0.78rem;color:var(--ink-soft);">${title}</div>
      <div style="font-family:Fraunces;font-size:1.1rem;margin-top:0.3rem;">${big}</div>
      <div style="font-family:JetBrains Mono;font-size:0.76rem;color:var(--ink-soft);margin-top:0.2rem;">${sub}</div>
    </div>`;
  }

  container.querySelector('#tt-step')!.addEventListener('click', step);
  step();
}
