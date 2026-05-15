import { renderMath, crosslinkBack, proofToggle } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-5';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'forward-posterior');
  sec.innerHTML = `
    <div class="section-label">§5</div>
    <h2 id="forward-posterior">The Tractable Forward Posterior</h2>
    <div class="prose">
      <p>The key new object is $q(x_{t-1} \\mid x_t, x_0)$ — the distribution of
      $x_{t-1}$ conditional on both $x_t$ (where we are now in the chain) and
      $x_0$ (the original clean data).</p>

      <p>Why include $x_0$? Without it, $q(x_{t-1} \\mid x_t)$ is the distribution
      we'd love to compute — the "true reverse step." But it's intractable: it's a
      function of the entire data distribution.</p>

      <p><strong>With $x_0$ included, the posterior is Gaussian and closed-form.</strong></p>

      <h3>The closed-form posterior</h3>
      <p>Conditional on $x_0$ and $x_t$, the joint $(x_{t-1}, x_t \\mid x_0)$ is
      jointly Gaussian (because both marginals and the conditional are Gaussian —
      chained Gaussians). By the conditioning identity:</p>

      <div class="formula-box">$$q(x_{t-1} \\mid x_t, x_0) \\;=\\; \\mathcal{N}\\!\\big(x_{t-1}; \\; \\tilde\\mu_t(x_t, x_0), \\; \\tilde\\beta_t I\\big)$$</div>

      <p>where</p>

      $$\\tilde\\mu_t(x_t, x_0) \\;=\\; \\frac{\\sqrt{\\bar\\alpha_{t-1}} \\beta_t}{1 - \\bar\\alpha_t} x_0 \\;+\\; \\frac{\\sqrt{\\alpha_t}(1 - \\bar\\alpha_{t-1})}{1 - \\bar\\alpha_t} x_t$$

      $$\\tilde\\beta_t \\;=\\; \\frac{1 - \\bar\\alpha_{t-1}}{1 - \\bar\\alpha_t} \\beta_t$$

      ${proofToggle('Derivation: $\\tilde\\mu_t$ and $\\tilde\\beta_t$ from Gaussian conditioning', `
          <p>Joint distribution of $(x_{t-1}, x_t)$ given $x_0$. Both are Gaussian:</p>
          <ul>
            <li>$x_{t-1} \\mid x_0 \\sim \\mathcal{N}(\\sqrt{\\bar\\alpha_{t-1}} x_0, (1 - \\bar\\alpha_{t-1}) I)$.</li>
            <li>$x_t \\mid x_{t-1} \\sim \\mathcal{N}(\\sqrt{\\alpha_t} x_{t-1}, \\beta_t I)$.</li>
          </ul>
          <p>The joint covariance has block structure with cross-covariance
          $\\sqrt{\\alpha_t}(1 - \\bar\\alpha_{t-1}) I$. Plugging into the conditional-Gaussian
          formula $\\mu_{a|b} = \\mu_a + \\Sigma_{ab}\\Sigma_{bb}^{-1}(x_b - \\mu_b)$:</p>
          $$\\tilde\\mu_t \\;=\\; \\sqrt{\\bar\\alpha_{t-1}} x_0 \\;+\\; \\frac{\\sqrt{\\alpha_t}(1 - \\bar\\alpha_{t-1})}{1 - \\bar\\alpha_t}\\!\\left(x_t - \\sqrt{\\bar\\alpha_t} x_0\\right)$$
          <p>Expanding and using $\\bar\\alpha_t = \\alpha_t \\bar\\alpha_{t-1}$:</p>
          $$\\tilde\\mu_t \\;=\\; \\frac{\\sqrt{\\bar\\alpha_{t-1}}\\beta_t}{1 - \\bar\\alpha_t} x_0 \\;+\\; \\frac{\\sqrt{\\alpha_t}(1 - \\bar\\alpha_{t-1})}{1 - \\bar\\alpha_t} x_t.$$
          <p>The conditional variance is
          $\\Sigma_{aa} - \\Sigma_{ab}\\Sigma_{bb}^{-1}\\Sigma_{ba} = \\tilde\\beta_t I$ with
          $\\tilde\\beta_t = (1 - \\bar\\alpha_{t-1})/(1 - \\bar\\alpha_t) \\cdot \\beta_t$.
          $\\blacksquare$</p>
        `, true)}

      <h3>Reading the formulas</h3>
      <p>The posterior mean $\\tilde\\mu_t$ is a <strong>convex combination</strong>
      of $x_0$ and $x_t$. The weights interpolate:</p>
      <ul>
        <li>At <strong>small $t$</strong>: the $x_t$ weight dominates. The
        posterior mean is essentially $x_t / \\sqrt{\\alpha_t}$. We trust the
        current noisy sample, since it has very little noise.</li>
        <li>At <strong>large $t$</strong>: the $x_0$ weight dominates. The
        posterior mean is pulled toward $x_0$. The current noisy sample is
        essentially pure noise; we have to lean on the clean data.</li>
        <li>At <strong>intermediate $t$</strong>: a graded mixture.</li>
      </ul>

      <p>The posterior variance $\\tilde\\beta_t$ is a <strong>rescaled version</strong>
      of the forward step variance $\\beta_t$:</p>

      $$\\tilde\\beta_t \\;\\leq\\; \\beta_t$$

      <p>Why ≤? Because conditioning on $x_0$ resolves uncertainty about $x_{t-1}$
      — the conditional variance is smaller than the marginal step variance. At
      $t = 1$ (lots of resolution from $x_0$),
      $\\tilde\\beta_1 / \\beta_1 \\approx 0.45$. At large $t$ (where $x_0$ tells
      us little), $\\tilde\\beta_t / \\beta_t \\to 1$.</p>

      <h3>What this gives us</h3>
      <p>The forward posterior is the <strong>target distribution</strong> that
      the reverse process $p_\\theta(x_{t-1} \\mid x_t)$ tries to match. The
      $L_{t-1}$ term in the loss is:</p>

      $$L_{t-1} \\;=\\; D_{\\mathrm{KL}}\\!\\big(q(x_{t-1} \\mid x_t, x_0) \\,\\|\\, p_\\theta(x_{t-1} \\mid x_t)\\big)$$

      <p>Both distributions are Gaussian. If we parameterize $p_\\theta$ with the
      same variance as $q$ (i.e., $\\Sigma_\\theta = \\tilde\\beta_t I$ or
      $\\beta_t I$), then by the shared-covariance special case of Gaussian KL:</p>

      $$L_{t-1} \\;=\\; \\frac{1}{2 \\sigma_t^2} \\|\\tilde\\mu_t(x_t, x_0) - \\mu_\\theta(x_t, t)\\|^2 \\;+\\; \\text{const}$$

      <p>The loss reduces to mean-squared error between the model's predicted mean
      and the analytical posterior mean. <strong>This is the setup for §6's
      parameterization choice.</strong></p>
    </div>

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'conditioning',
      toAnchorLabel: '5 — Conditioning',
      body: `The jointly-Gaussian conditioning identity gives the closed-form posterior.
             The shared-covariance KL reduces $L_{t-1}$ to squared Mahalanobis distance
             between means.`,
    })}

    <div id="viz-forward-posterior-explorer" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
