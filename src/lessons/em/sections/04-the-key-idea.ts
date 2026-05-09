import { renderMath } from '@shared/ui';
import { callout } from '@shared/ui';
import { mountELBODiagram } from '../viz/elbo-diagram';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-4';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§4</div>
    <h2>The Key Idea</h2>
    <div class="prose">
      <p>Here is the trick. Instead of maximizing the incomplete-data log-likelihood
      $\\ell(\\theta \\mid x)$ directly — which is hard — EM iteratively maximizes a
      <em>surrogate</em> function that is always a lower bound on $\\ell$ and always
      touches $\\ell$ at the current parameter estimate.</p>

      <p id="q-function">Given a current estimate $\\theta^{(t)}$, define the <strong>Q-function</strong>
      as the expected complete-data log-likelihood under the posterior distribution
      of the hidden variables:</p>

      $$\\boxed{\\; Q(\\theta \\mid \\theta^{(t)}) \\;:=\\; \\mathbb{E}_{Z \\,\\sim\\, k(\\cdot \\mid x, \\theta^{(t)})} \\big[\\, \\ell_c(\\theta \\mid x, Z) \\,\\big] \\;}$$

      <p>where $k(z \\mid x, \\theta^{(t)}) = P(Z = z \\mid X = x, \\theta^{(t)})$ is the
      posterior over hidden coin assignments given current parameters.</p>

      <p>The <strong>E-step</strong> (Expectation step) computes $Q$ by computing these
      posterior probabilities. The <strong>M-step</strong> (Maximization step) then
      sets:</p>

      $$\\boxed{\\; \\theta^{(t+1)} \\;:=\\; \\arg\\max_\\theta \\; Q(\\theta \\mid \\theta^{(t)}) \\;}$$

      <p>Crucially, because the complete-data log-likelihood separates into independent
      terms for $\\theta_A$ and $\\theta_B$, maximizing $Q$ has a closed-form solution —
      unlike maximizing $\\ell$ directly.</p>

      <p>The key insight is that EM turns a hard optimization problem (maximize
      $\\ell$ with a sum inside a log) into a sequence of easy optimization problems
      (maximize $Q$, which has no log of a sum).</p>
    </div>
    ${callout('tip', 'Read this twice',
    `<p>The Q-function is <em>not</em> the incomplete-data log-likelihood. It is the
    expected complete-data log-likelihood — averaged over the hidden variables using
    the current parameter estimate. This expectation is exactly what the E-step computes.</p>`,
  )}
    <div id="viz-elbo" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountELBODiagram(sec.querySelector('#viz-elbo') as HTMLElement);
}
