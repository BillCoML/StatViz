import { renderMath, callout, crosslinkBack } from '@shared/ui';
import { mountEMTrajectory } from '../viz/em-trajectory';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-7';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§7</div>
    <h2>EM Through the ELBO Lens</h2>
    <div class="prose" id="em-as-elbo-ascent">
      <p>Now we'll show that the EM algorithm — derived two lessons ago for the two-coins
      problem — is <strong>literally coordinate ascent on the ELBO</strong>. This
      re-framing replaces the EM proof with one line and generalizes EM to settings where
      the E-step can't be done exactly (which is, well, most of modern ML).</p>

      <h3>Setup</h3>
      <p>Now we have <strong>two</strong> sets of unknowns: the variational distribution
      $q$ and the model parameters $\\theta$. The joint depends on $\\theta$:
      $p_\\theta(x, z) = p_\\theta(z) p_\\theta(x \\mid z)$. The ELBO depends on both:</p>
      $$\\mathcal{L}(q, \\theta) \\;:=\\; \\mathbb{E}_q[\\log p_\\theta(x, Z)] - \\mathbb{E}_q[\\log q(Z)]$$
      <p>The fundamental identity becomes:</p>
      $$\\log p_\\theta(x) \\;=\\; \\mathcal{L}(q, \\theta) \\;+\\; D_{\\mathrm{KL}}(q(z) \\,\\|\\, p_\\theta(z \\mid x))$$

      <h3>Coordinate ascent</h3>
      <p>Maximize $\\mathcal{L}$ alternately, one variable at a time:</p>
      <p><strong>Step E</strong> (optimize over $q$, holding $\\theta$ fixed):</p>
      $$q^{(t+1)} \\;=\\; \\arg\\max_q \\; \\mathcal{L}(q, \\theta^{(t)})$$
      <p>If we allow $q$ to be <em>any</em> distribution (no restriction to a tractable
      family), the maximum is at $q = p_{\\theta^{(t)}}(z \\mid x)$ — the <strong>exact
      posterior</strong> — because that's where the KL gap closes. At the optimum,</p>
      $$\\mathcal{L}(q^{(t+1)}, \\theta^{(t)}) \\;=\\; \\log p_{\\theta^{(t)}}(x)$$
      <p><strong>Step M</strong> (optimize over $\\theta$, holding $q$ fixed):</p>
      $$\\theta^{(t+1)} \\;=\\; \\arg\\max_\\theta \\; \\mathcal{L}(q^{(t+1)}, \\theta)
      \\;=\\; \\arg\\max_\\theta \\; \\mathbb{E}_{q^{(t+1)}}[\\log p_\\theta(x, Z)]$$
      <p>(The entropy $-\\mathbb{E}_{q}[\\log q(Z)]$ doesn't depend on $\\theta$, so it
      drops.)</p>

      <h3>These are exactly the EM steps</h3>
      <ul>
        <li>The EM E-step computes responsibilities $\\gamma_i^A = P(Z_i = A \\mid x_i, \\theta^{(t)})$.
        <strong>That's $q^{(t+1)}(z) = p_{\\theta^{(t)}}(z \\mid x)$ for the two-coins
        discrete case.</strong> The "responsibilities" <em>are</em> the exact posterior.</li>
        <li>The EM M-step maximizes $Q(\\theta \\mid \\theta^{(t)}) = \\mathbb{E}_{q^{(t+1)}}[\\log p_\\theta(x, Z)]$.
        <strong>That's the ELBO M-step.</strong> The Q function is the ELBO with the
        entropy term dropped (legal because it's constant in $\\theta$).</li>
      </ul>
      <p>So EM is coordinate ascent on $\\mathcal{L}(q, \\theta)$.</p>

      <h3>And the monotonicity proof becomes one line</h3>
      <p>Coordinate ascent on a function never decreases its value. So
      $\\mathcal{L}(q^{(t+1)}, \\theta^{(t+1)}) \\geq \\mathcal{L}(q^{(t)}, \\theta^{(t)})$.
      But after each E-step the bound is tight:
      $\\mathcal{L}(q^{(t+1)}, \\theta^{(t)}) = \\log p_{\\theta^{(t)}}(x)$.
      Combined:</p>
      $$\\log p_{\\theta^{(t+1)}}(x) \\;\\geq\\; \\mathcal{L}(q^{(t+1)}, \\theta^{(t+1)})
      \\;\\geq\\; \\mathcal{L}(q^{(t+1)}, \\theta^{(t)}) \\;=\\; \\log p_{\\theta^{(t)}}(x)$$
      <p>The middle step is the M-step's monotonicity ($\\theta^{(t+1)}$ maximizes
      $\\mathcal{L}(q^{(t+1)}, \\cdot)$). The first step is the fundamental identity
      ($\\log p \\geq \\mathcal{L}$ always). The last equality is the post-E-step tightness.
      Three lines, no Gibbs' inequality needed at the level of $\\theta$.
      <strong>The EM lesson's monotonicity theorem is now this one-paragraph argument.</strong></p>
    </div>

    ${crosslinkBack({
      toLesson: 'em',
      toAnchor: 'monotonicity',
      toAnchorLabel: '8',
      body: `<p><strong>Re-derivation of: EM §8 monotonicity proof.</strong> The original EM
        proof factored $\\ell(\\theta^{(t+1)}) - \\ell(\\theta^{(t)})$ into a Q-gap (M-step)
        and a KL-gap (Gibbs' inequality on the conditional density). That proof is correct
        but elaborate. The ELBO view collapses it to "coordinate ascent on $\\mathcal{L}$
        with tight bound after each E-step."</p>`,
    })}

    <div class="prose">
      <h3>Worked numbers — re-running the 2-coins ELBO</h3>
      <p>Initialize $\\theta^{(0)} = (0.6, 0.5)$ and $q^{(0)}$ uninformative
      ($\\gamma_i = 0.5$ for all $i$).</p>

      <table>
        <thead>
          <tr>
            <th>step</th>
            <th>what happened</th>
            <th>$\\mathcal{L}(q, \\theta)$</th>
            <th>$\\log p_\\theta(x)$</th>
            <th>KL gap</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>0</td><td>initialized</td><td>$-33.5458$</td><td>$-33.0939$</td><td>$0.4519$</td></tr>
          <tr><td>0+E</td><td>E-step at $\\theta^{(0)}$</td><td>$-33.0939$</td><td>$-33.0939$</td><td>$0.0000$</td></tr>
          <tr><td>0+M</td><td>M-step → $\\theta^{(1)} \\approx (0.713, 0.581)$</td><td>$-31.9972$</td><td>$-31.8593$</td><td>$0.1380$</td></tr>
          <tr><td>1+E</td><td>E-step at $\\theta^{(1)}$</td><td>$-31.8593$</td><td>$-31.8593$</td><td>$0.0000$</td></tr>
        </tbody>
      </table>

      <p>Read the table: at every E-step the ELBO <strong>snaps up to meet
      $\\log p(x)$</strong>. At every M-step both rise, but the ELBO trails
      $\\log p(x)$ slightly because $q$ is no longer the exact posterior for the new
      $\\theta$. The dance continues until convergence.</p>
    </div>

    <div id="viz-em-trajectory" class="viz-wide"></div>

    ${callout('tip', 'When you can\'t do the E-step exactly',
      `<p>If the variational family $\\mathcal{Q}$ doesn't contain the exact posterior,
      the E-step is "best $q$ in $\\mathcal{Q}$" — the bound doesn't fully tighten.
      This is <strong>variational EM</strong> (or, if generalized further, just plain VI).
      Modern deep generative models (VAE, DDPM) are in this regime. The KL gap is the
      price of tractability.</p>`
    )}
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountEMTrajectory(sec.querySelector('#viz-em-trajectory') as HTMLElement);
}
