import { renderMath } from '@shared/ui';
import { callout } from '@shared/ui';
import { mountResponsibilityCalculator } from '../viz/responsibility-calculator';
import { eStep } from '../em/algorithm';
import { TRIALS } from '../em/data';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-5';
  sec.className = 'section';

  // Compute responsibilities from algorithm
  const R = eStep(0.6, 0.5);

  // Build responsibility table rows
  const tableRows = TRIALS.map((t, i) => {
    const { gammaA, gammaB } = R[i];
    return `<tr>
      <td>${t.id}</td>
      <td>${t.heads}</td>
      <td>${gammaA.toFixed(4)}</td>
      <td>${gammaB.toFixed(4)}</td>
    </tr>`;
  }).join('');

  // Compute gamma_1^A manually for the worked example
  // theta_A=0.6, theta_B=0.5, trial 1: heads=5, tails=5
  const tA = 0.6, tB = 0.5;
  const h1 = 5, t1 = 5;
  const lA1 = Math.log(0.5) + h1 * Math.log(tA) + t1 * Math.log(1 - tA);
  const lB1 = Math.log(0.5) + h1 * Math.log(tB) + t1 * Math.log(1 - tB);
  const m1 = Math.max(lA1, lB1);
  const a1 = Math.exp(lA1 - m1);
  const b1 = Math.exp(lB1 - m1);
  const gamma1A = a1 / (a1 + b1);

  sec.innerHTML = `
    <div class="section-label">§5</div>
    <h2>The E-Step</h2>
    <div class="prose">
      <p>The E-step computes, for each trial $i$, the probability that coin A
      (or B) was used, given the observed heads count $x_i$ and the current
      parameter estimate $\\theta^{(t)}$. These probabilities are called
      <strong>responsibilities</strong>.</p>

      <p>By Bayes' rule:</p>
      $$\\gamma_i^A \\;=\\; P(z_i = A \\mid x_i, \\theta^{(t)}) \\;=\\;
      \\frac{P(x_i \\mid z_i=A,\\, \\theta^{(t)}) \\cdot P(z_i=A)}{\\sum_{c \\in \\{A,B\\}} P(x_i \\mid z_i=c,\\,\\theta^{(t)}) \\cdot P(z_i=c)}$$

      <p>With equal priors and binomial likelihoods, this simplifies to:</p>
      $$\\boxed{\\; \\gamma_i^A \\;=\\;
      \\frac{(\\theta_A^{(t)})^{x_i}\\,(1-\\theta_A^{(t)})^{10-x_i}}
           {(\\theta_A^{(t)})^{x_i}\\,(1-\\theta_A^{(t)})^{10-x_i} + (\\theta_B^{(t)})^{x_i}\\,(1-\\theta_B^{(t)})^{10-x_i}} \\;}$$
      <p>and $\\gamma_i^B = 1 - \\gamma_i^A$.</p>

      <p>Substituting into the Q-function:</p>
      $$Q(\\theta \\mid \\theta^{(t)}) = \\sum_{i=1}^{5} \\Bigl[
        \\gamma_i^A \\bigl( x_i \\log\\theta_A + (10-x_i)\\log(1-\\theta_A) \\bigr)
        + \\gamma_i^B \\bigl( x_i \\log\\theta_B + (10-x_i)\\log(1-\\theta_B) \\bigr)
      \\Bigr]$$
      <p>This is a weighted sum of binomial log-likelihoods — no log of a sum in sight.</p>
    </div>

    <div class="worked-example">
      <div class="worked-example-title">Worked Example — $\\gamma_1^A$ at $\\theta^{(0)} = (0.6,\\, 0.5)$</div>
      <div class="prose">
        <p>Trial 1 has $x_1 = 5$ heads (and 5 tails). With $\\theta_A^{(0)} = 0.6$,
        $\\theta_B^{(0)} = 0.5$:</p>
        $$\\text{Numerator: } 0.6^5 \\times 0.4^5 = ${(Math.pow(0.6,5)*Math.pow(0.4,5)).toFixed(6)}$$
        $$\\text{Denominator: } 0.6^5 \\times 0.4^5 + 0.5^5 \\times 0.5^5 = ${(Math.pow(0.6,5)*Math.pow(0.4,5) + Math.pow(0.5,10)).toFixed(6)}$$
        $$\\gamma_1^A = \\frac{${(Math.pow(0.6,5)*Math.pow(0.4,5)).toFixed(6)}}{${(Math.pow(0.6,5)*Math.pow(0.4,5) + Math.pow(0.5,10)).toFixed(6)}} \\approx ${gamma1A.toFixed(4)}$$
      </div>
    </div>

    <div class="prose">
      <p>Applying the same formula to all five trials at $\\theta^{(0)} = (0.6, 0.5)$:</p>
    </div>
    <table class="data-table">
      <thead>
        <tr>
          <th>Trial $i$</th>
          <th>Heads $x_i$</th>
          <th>$\\gamma_i^A$</th>
          <th>$\\gamma_i^B$</th>
        </tr>
      </thead>
      <tbody>
        ${tableRows}
      </tbody>
    </table>
    <div class="prose">
      <p>Notice how trials 2 and 3 (with many heads) assign higher responsibility to
      coin A ($\\theta_A = 0.6 > \\theta_B = 0.5$), while trial 4 (few heads) assigns
      more responsibility to coin B.</p>
    </div>
    ${callout('info', 'Numerical stability',
    `<p>In practice, we compute responsibilities in log-space using the log-sum-exp
    trick to avoid floating-point underflow. The algorithm module already does this.</p>`,
  )}
    <div id="viz-responsibility-calc" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountResponsibilityCalculator(sec.querySelector('#viz-responsibility-calc') as HTMLElement);
}
