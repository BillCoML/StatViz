import { renderMath, crosslinkBack } from '@shared/ui';
import { mountKLCalculator } from '../viz/kl-calculator';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-5';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§5</div>
    <h2>Non-Negativity (Gibbs' Inequality)</h2>
    <div class="prose" id="gibbs-inequality">
      <h3>Theorem (Gibbs' inequality)</h3>
      <p>For any two probability distributions $p, q$ on the same space,</p>
      $$\\boxed{\\;\\; D_{\\mathrm{KL}}(p \\,\\|\\, q) \\;\\geq\\; 0 \\;\\;}$$
      <p>with equality if and only if $p = q$ almost everywhere (with respect to $p$).</p>
      <p>This is the back-pocket inequality of probabilistic ML. Whenever a proof "needs
      an inequality," chances are this is the one.</p>

      <h3>Proof</h3>
      <p>By definition,</p>
      $$D_{\\mathrm{KL}}(p \\,\\|\\, q) \\;=\\; \\mathbb{E}_p\\!\\left[\\log \\frac{p(X)}{q(X)}\\right]
        \\;=\\; -\\mathbb{E}_p\\!\\left[\\log \\frac{q(X)}{p(X)}\\right]$$
      <p>Since $\\log$ is <strong>concave</strong>, Jensen's inequality with
      $\\varphi = \\log$ reverses, giving</p>
      $$\\mathbb{E}_p\\!\\left[\\log \\frac{q(X)}{p(X)}\\right]
        \\;\\leq\\; \\log \\mathbb{E}_p\\!\\left[\\frac{q(X)}{p(X)}\\right]$$
      <p>Now compute the right-hand side directly:</p>
      $$\\mathbb{E}_p\\!\\left[\\frac{q(X)}{p(X)}\\right]
        \\;=\\; \\int p(x) \\cdot \\frac{q(x)}{p(x)} \\, dx
        \\;=\\; \\int q(x) \\, dx \\;=\\; 1$$
      <p>So $\\mathbb{E}_p[\\log(q/p)] \\leq \\log 1 = 0$, which gives
      $D_{\\mathrm{KL}}(p \\,\\|\\, q) = -\\mathbb{E}_p[\\log(q/p)] \\geq 0$.</p>
      <p>For the <strong>equality condition</strong>: $\\log$ is <em>strictly</em> concave,
      so Jensen is strict unless $q(X)/p(X)$ is constant almost surely under $p$. Combined
      with the constraint that both $p$ and $q$ integrate to 1, the constant must be 1,
      i.e. $p = q$ almost everywhere. $\\blacksquare$</p>
    </div>

    ${crosslinkBack({
      toLesson: 'em',
      toAnchor: 'monotonicity',
      toAnchorLabel: '8',
      body: `<p>The EM convergence proof relied on the inequality
        $\\mathrm{KL}(k(\\cdot \\mid x, \\theta^{(t)}) \\,\\|\\, k(\\cdot \\mid x, \\theta^{(t+1)})) \\geq 0$
        applied to the conditional distribution of the missing data. That inequality is
        exactly Gibbs' inequality, which we just proved. The EM lesson called it
        "Gibbs' inequality" and asked you to take it on trust. <strong>It's no longer on trust.</strong></p>`,
    })}

    <div class="worked-example">
      <div class="worked-example-title">Sanity checks</div>
      <p>Verify on the Bernoulli example from §4:
      $D(\\mathrm{Bern}(0.7) \\,\\|\\, \\mathrm{Bern}(0.5)) \\approx 0.0823 > 0$
      and $D(\\mathrm{Bern}(0.5) \\,\\|\\, \\mathrm{Bern}(0.7)) \\approx 0.0872 > 0$.
      Both positive, both nonzero (since the distributions differ), as required.</p>
    </div>

    <div id="viz-gibbs" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountKLCalculator(sec.querySelector('#viz-gibbs') as HTMLElement, { showGibbsBanner: true });
}
