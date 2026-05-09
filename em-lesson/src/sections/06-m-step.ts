import { renderMath } from '../katex-render';
import { callout } from '../ui/callout';
import { eStep, runUntilConvergence, observedLogLikelihood } from '../em/algorithm';
import { TRIALS } from '../em/data';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-6';
  sec.className = 'section';

  // Compute E-step values
  const R = eStep(0.6, 0.5);

  // Compute sums for worked example
  let sumGammaA_x = 0, sumGammaA_n = 0;
  let sumGammaB_x = 0, sumGammaB_n = 0;
  R.forEach(({ gammaA, gammaB }, i) => {
    sumGammaA_x += gammaA * TRIALS[i].heads;
    sumGammaA_n += gammaA * 10;
    sumGammaB_x += gammaB * TRIALS[i].heads;
    sumGammaB_n += gammaB * 10;
  });
  const thetaA1 = sumGammaA_x / sumGammaA_n;
  const thetaB1 = sumGammaB_x / sumGammaB_n;

  // Run to convergence for trajectory table
  const history = runUntilConvergence(0.6, 0.5);

  // Build trajectory table (show iterations 0 through min(10, history.length-1))
  const showRows = Math.min(11, history.length);
  const trajRows = history.slice(0, showRows).map(s =>
    `<tr>
      <td>${s.iteration}</td>
      <td>${s.thetaA.toFixed(4)}</td>
      <td>${s.thetaB.toFixed(4)}</td>
      <td>${s.observedLogLikelihood.toFixed(4)}</td>
    </tr>`,
  ).join('');

  const finalState = history[history.length - 1];

  sec.innerHTML = `
    <div class="section-label">§6</div>
    <h2>The M-Step</h2>
    <div class="prose">
      <p>Given the responsibilities $\\{\\gamma_i^A, \\gamma_i^B\\}$ from the E-step,
      the M-step maximizes $Q(\\theta \\mid \\theta^{(t)})$ over $\\theta$.</p>

      <p>Because $Q$ separates into a $\\theta_A$-part and a $\\theta_B$-part, we can
      maximize each independently:</p>
      $$Q(\\theta \\mid \\theta^{(t)}) = \\underbrace{\\sum_{i} \\gamma_i^A \\bigl[ x_i \\log\\theta_A + (10-x_i)\\log(1-\\theta_A) \\bigr]}_{\\text{depends only on } \\theta_A}
      + \\underbrace{\\sum_{i} \\gamma_i^B \\bigl[ x_i \\log\\theta_B + (10-x_i)\\log(1-\\theta_B) \\bigr]}_{\\text{depends only on } \\theta_B}$$

      <p>Taking the derivative with respect to $\\theta_A$ and setting it to zero:</p>
      $$\\frac{\\partial Q}{\\partial \\theta_A} = \\sum_{i} \\gamma_i^A \\left[ \\frac{x_i}{\\theta_A} - \\frac{10 - x_i}{1 - \\theta_A} \\right] = 0$$
      $$\\Longrightarrow \\quad (1-\\theta_A) \\sum_i \\gamma_i^A x_i = \\theta_A \\sum_i \\gamma_i^A (10 - x_i)$$
      $$\\Longrightarrow \\quad \\sum_i \\gamma_i^A x_i = \\theta_A \\sum_i \\gamma_i^A \\cdot 10$$

      <p>Solving:</p>
      $$\\boxed{\\;\\; \\theta_A^{(t+1)} \\;=\\; \\frac{\\displaystyle \\sum_{i=1}^{5} \\gamma_i^A \\cdot x_i}{\\displaystyle \\sum_{i=1}^{5} \\gamma_i^A \\cdot 10} \\;\\;}$$

      <p>And symmetrically:</p>
      $$\\theta_B^{(t+1)} \\;=\\; \\frac{\\displaystyle \\sum_{i=1}^{5} \\gamma_i^B \\cdot x_i}{\\displaystyle \\sum_{i=1}^{5} \\gamma_i^B \\cdot 10}$$
    </div>

    ${callout('tip', 'Interpret the formula',
    `<p>$\\theta_A^{(t+1)}$ is a weighted average of heads rates, where trial $i$
    gets weight $\\gamma_i^A$ — the probability that trial $i$ used coin A.
    Trials that probably came from coin A contribute more to the coin A estimate.
    This is exactly the "soft" version of pooling known-A trials.</p>`,
  )}

    <div class="worked-example">
      <div class="worked-example-title">Worked Example — M-step at iteration 1</div>
      <div class="prose">
        <p>Using the responsibilities from §5 at $\\theta^{(0)} = (0.6, 0.5)$:</p>
        $$\\sum_i \\gamma_i^A \\cdot x_i = ${sumGammaA_x.toFixed(4)}, \\qquad
        \\sum_i \\gamma_i^A \\cdot 10 = ${sumGammaA_n.toFixed(4)}$$
        $$\\theta_A^{(1)} = \\frac{${sumGammaA_x.toFixed(4)}}{${sumGammaA_n.toFixed(4)}} = ${thetaA1.toFixed(4)}$$
        $$\\sum_i \\gamma_i^B \\cdot x_i = ${sumGammaB_x.toFixed(4)}, \\qquad
        \\sum_i \\gamma_i^B \\cdot 10 = ${sumGammaB_n.toFixed(4)}$$
        $$\\theta_B^{(1)} = \\frac{${sumGammaB_x.toFixed(4)}}{${sumGammaB_n.toFixed(4)}} = ${thetaB1.toFixed(4)}$$
        <p>The log-likelihood at iteration 1 is
        $\\ell(\\theta^{(1)}) = ${observedLogLikelihood(thetaA1, thetaB1).toFixed(4)}$,
        up from $${observedLogLikelihood(0.6, 0.5).toFixed(4)}$ at iteration 0. ✓</p>
      </div>
    </div>

    <div class="prose">
      <p>Running until convergence (starting from $\\theta^{(0)} = (0.6, 0.5)$) gives
      the trajectory below. EM converges in about ${history.length - 1} iterations
      to $\\hat{\\theta}_A \\approx ${finalState.thetaA.toFixed(4)}$,
      $\\hat{\\theta}_B \\approx ${finalState.thetaB.toFixed(4)}$.</p>
    </div>

    <table class="data-table">
      <thead>
        <tr>
          <th>Iteration $t$</th>
          <th>$\\theta_A^{(t)}$</th>
          <th>$\\theta_B^{(t)}$</th>
          <th>$\\ell(\\theta^{(t)} \\mid x)$</th>
        </tr>
      </thead>
      <tbody>
        ${trajRows}
        ${history.length > showRows ? `<tr><td colspan="4" style="text-align:center;font-style:italic;">… converges at iteration ${history.length - 1}</td></tr>` : ''}
      </tbody>
    </table>
  `;

  container.appendChild(sec);
  renderMath(sec);
}
