import { renderMath } from '@shared/ui';
import { mountEMSimulator } from '../viz/em-simulator';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-7';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§7</div>
    <h2>The Full Algorithm</h2>
    <div class="prose">
      <p>Putting both steps together, the EM algorithm for our two-coin problem is
      remarkably compact. It alternates between computing responsibilities (E-step)
      and updating parameters (M-step) until the parameters stop changing.</p>
    </div>
    <div class="pseudocode"><span class="kw">Input:</span>  observed head counts $x_1, \\ldots, x_5$; initial guess $\\theta^{(0)} = (\\theta_A^{(0)}, \\theta_B^{(0)})$
<span class="kw">Initialize:</span> $t \\leftarrow 0$
<span class="kw">Repeat:</span>
  <span class="cm">// E-step: compute responsibilities</span>
  <strong>for</strong> $i = 1$ <strong>to</strong> $5$:
    $\\gamma_i^A \\leftarrow \\dfrac{(\\theta_A^{(t)})^{x_i}(1-\\theta_A^{(t)})^{10-x_i}}{(\\theta_A^{(t)})^{x_i}(1-\\theta_A^{(t)})^{10-x_i} + (\\theta_B^{(t)})^{x_i}(1-\\theta_B^{(t)})^{10-x_i}}$
    $\\gamma_i^B \\leftarrow 1 - \\gamma_i^A$

  <span class="cm">// M-step: update parameters</span>
  $\\theta_A^{(t+1)} \\leftarrow \\dfrac{\\sum_{i} \\gamma_i^A \\cdot x_i}{\\sum_{i} \\gamma_i^A \\cdot 10}$
  $\\theta_B^{(t+1)} \\leftarrow \\dfrac{\\sum_{i} \\gamma_i^B \\cdot x_i}{\\sum_{i} \\gamma_i^B \\cdot 10}$

  $t \\leftarrow t + 1$
<strong>until</strong>  $|\\theta_A^{(t)} - \\theta_A^{(t-1)}| + |\\theta_B^{(t)} - \\theta_B^{(t-1)}| < \\varepsilon$
<span class="kw">Return:</span> $\\hat{\\theta} = (\\theta_A^{(t)}, \\theta_B^{(t)})$</div>
    <div class="prose">
      <p>The simulator below lets you control every step. You can choose initial values,
      step through E and M individually, or watch the algorithm run to convergence.
      The four panels track the responsibilities, the trajectory on the likelihood
      surface, and the evolution of both parameters and the log-likelihood over time.</p>
    </div>
    <div id="two-coins-simulator" class="viz-wide viz-full"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountEMSimulator(sec.querySelector('#two-coins-simulator') as HTMLElement);
}
