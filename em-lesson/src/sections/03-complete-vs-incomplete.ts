import { renderMath } from '../katex-render';
import { callout } from '../ui/callout';
import { mountLikelihoodSurface } from '../viz/likelihood-surface';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-3';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§3</div>
    <h2>Complete vs Incomplete Data</h2>
    <div class="prose">
      <p>Let's be precise about what we mean by <em>complete</em> and <em>incomplete</em> data.</p>
      <p>The <strong>complete data</strong> would be the pair $(x, z)$ where
      $x = (x_1, \\ldots, x_5)$ are the observed head counts and
      $z = (z_1, \\ldots, z_5)$ are the hidden coin identities, $z_i \\in \\{A, B\\}$.</p>
      <p>The <strong>incomplete data</strong> is just $x$ — we only see the head counts,
      not which coin was used.</p>

      <h3>Complete-data likelihood</h3>
      <p>If we knew $z$, the likelihood would factor nicely:</p>
      $$L_c(\\theta \\mid x, z) \\;=\\; \\prod_{i=1}^{5} \\binom{10}{x_i} \\, \\theta_{z_i}^{x_i} (1 - \\theta_{z_i})^{10 - x_i}$$
      <p>Taking logs and dropping the constant binomial coefficients:</p>
      $$\\ell_c(\\theta \\mid x, z) = \\sum_{i : z_i = A} \\bigl[ x_i \\log \\theta_A + (10 - x_i) \\log(1-\\theta_A) \\bigr]
      + \\sum_{i : z_i = B} \\bigl[ x_i \\log \\theta_B + (10 - x_i) \\log(1-\\theta_B) \\bigr]$$
      <p>This separates into two independent terms — one depending only on $\\theta_A$,
      one on $\\theta_B$. Maximizing each term separately gives the MLE:</p>
      $$\\hat{\\theta}_A = \\frac{\\sum_{i: z_i=A} x_i}{\\sum_{i: z_i=A} 10}, \\qquad
      \\hat{\\theta}_B = \\frac{\\sum_{i: z_i=B} x_i}{\\sum_{i: z_i=B} 10}$$

      <h3>Incomplete-data likelihood</h3>
      <p>Without knowing $z$, we must marginalize over all possible coin assignments:</p>
      $$L(\\theta \\mid x) = \\prod_{i=1}^{5} \\sum_{c \\in \\{A,B\\}} P(z_i = c) \\cdot \\binom{10}{x_i} \\theta_c^{x_i} (1-\\theta_c)^{10-x_i}$$
      <p>With equal priors $P(z_i = A) = P(z_i = B) = \\tfrac{1}{2}$, the log-likelihood is:</p>
      $$\\ell(\\theta \\mid x) = \\sum_{i=1}^{5} \\log \\left[
        \\tfrac{1}{2} \\theta_A^{x_i}(1-\\theta_A)^{10-x_i}
        + \\tfrac{1}{2} \\theta_B^{x_i}(1-\\theta_B)^{10-x_i}
      \\right] + \\text{const}$$
      <p>The sum appears <em>inside</em> the log. There is no closed-form solution for
      $\\nabla_\\theta \\ell = 0$. The contour plot below shows the resulting rugged,
      symmetric landscape.</p>
    </div>
    <div id="viz-likelihood-surface" class="viz-wide"></div>
    ${callout('warning', 'The obstacle',
    `<p>Setting the gradient of $\\ell(\\theta \\mid x)$ to zero leads to implicit equations
    that cannot be solved in closed form. Numerical maximization is possible but
    ignores the rich structure of the problem. EM exploits that structure instead.</p>`,
  )}
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountLikelihoodSurface(sec.querySelector('#viz-likelihood-surface') as HTMLElement);
}
