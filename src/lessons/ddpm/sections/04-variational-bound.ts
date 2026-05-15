import { renderMath, crosslinkBack, proofToggle } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-4';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'vlb');
  sec.innerHTML = `
    <div class="section-label">§4</div>
    <h2 id="vlb">The Variational Bound</h2>
    <div class="prose">
      <p>Now derive the loss. We want to maximize $\\log p_\\theta(x_0)$ for data
      $x_0 \\sim q(x_0)$ (where $q(x_0) = p_{\\text{data}}$ is the data
      distribution). Because $p_\\theta(x_0) = \\int p_\\theta(x_{0:T}) dx_{1:T}$
      is intractable, we use the variational lower bound — the standard VAE trick,
      scaled to $T$ latents.</p>

      <h3>The standard ELBO</h3>
      <p>Using $q(x_{1:T} \\mid x_0)$ as the variational distribution (it's not
      actually variational here since it's fixed, but it plays the same role):</p>

      $$-\\log p_\\theta(x_0) \\;\\leq\\; \\mathbb{E}_q\\!\\left[-\\log \\frac{p_\\theta(x_{0:T})}{q(x_{1:T} \\mid x_0)}\\right] \\;=:\\; L$$

      <p>This is the negative-ELBO form of Form 1 from ELBO/VI §4. Expand by
      definition of the products:</p>

      $$L \\;=\\; \\mathbb{E}_q\\!\\left[-\\log p(x_T) \\;-\\; \\sum_{t \\geq 1} \\log \\frac{p_\\theta(x_{t-1} \\mid x_t)}{q(x_t \\mid x_{t-1})}\\right]$$

      <p>A sum of $T + 1$ terms. We could train on this directly with Monte Carlo,
      but each term has high variance because $q(x_t \\mid x_{t-1})$ is sharply
      peaked. <strong>The paper's first algebraic trick is to rewrite this in a
      form where every term is a KL between two Gaussians</strong> — which can be
      computed in closed form (using the Cookbook), eliminating the Monte Carlo
      variance from those terms.</p>

      <h3>The variance-reduced form</h3>
      <div class="formula-box">$$L \\;=\\; \\mathbb{E}_q\\!\\left[\\underbrace{D_{\\mathrm{KL}}\\!\\big(q(x_T \\mid x_0) \\,\\|\\, p(x_T)\\big)}_{L_T} \\;+\\; \\sum_{t > 1} \\underbrace{D_{\\mathrm{KL}}\\!\\big(q(x_{t-1} \\mid x_t, x_0) \\,\\|\\, p_\\theta(x_{t-1} \\mid x_t)\\big)}_{L_{t-1}} \\;\\underbrace{-\\log p_\\theta(x_0 \\mid x_1)}_{L_0}\\right]$$</div>

      <p>Three families of terms:</p>
      <ul>
        <li><strong>$L_T$</strong>: KL between the end of the forward chain and the
        prior. With $\\beta_t$ chosen so the forward chain reaches near-Gaussian
        noise, this is <strong>tiny</strong> ($\\approx 2.9 \\times 10^{-5}$ bits/dim
        for the paper's setup) and <strong>constant in $\\theta$</strong> (the
        forward process has no parameters). Drop it.</li>
        <li><strong>$L_{t-1}$ for $t = 2, \\ldots, T$</strong>: KL between the
        <strong>tractable forward posterior</strong> $q(x_{t-1} \\mid x_t, x_0)$
        (which we'll compute in §5) and the <strong>learned reverse transition</strong>
        $p_\\theta(x_{t-1} \\mid x_t)$. Both are Gaussian → closed-form KL via the
        Gaussian Cookbook.</li>
        <li><strong>$L_0$</strong>: the final reconstruction term. Different from
        the others because the data is discrete (image pixels in $\\{0, \\ldots, 255\\}$);
        the paper handles this with a discretized Gaussian decoder (Equation 13 of
        the paper).</li>
      </ul>

      <h3>Why this is variance-reduced</h3>
      <p>In the original sum, each term involved
      $\\log p_\\theta(x_{t-1} \\mid x_t) - \\log q(x_t \\mid x_{t-1})$, which is
      the <em>ratio</em> of two Gaussians evaluated at samples drawn from a third.
      Monte Carlo estimates of such ratios have variance that grows with how
      different the distributions are.</p>

      <p>The rewritten form replaces these high-variance log-ratios with
      <strong>closed-form KL divergences between Gaussians</strong>. The expected
      values are computed analytically; the Monte Carlo only needs to sample $x_0$
      and the noise (one sample per gradient step gives good signal). This is what
      makes training viable.</p>

      ${proofToggle('Full derivation of the variance-reduced VLB', `
          <p>Start from $L = \\mathbb{E}_q[-\\log p_\\theta(x_{0:T}) / q(x_{1:T} \\mid x_0)]$.
          Expand the logs and rearrange:</p>
          $$L \\;=\\; \\mathbb{E}_q\\!\\left[-\\log p(x_T) \\;-\\; \\sum_{t > 1}\\log \\frac{p_\\theta(x_{t-1} \\mid x_t)}{q(x_t \\mid x_{t-1})} \\;-\\; \\log \\frac{p_\\theta(x_0 \\mid x_1)}{q(x_1 \\mid x_0)}\\right]$$
          <p>The key move: by Bayes' rule with $x_0$ included,</p>
          $$q(x_t \\mid x_{t-1}) \\;=\\; q(x_t \\mid x_{t-1}, x_0) \\;=\\; \\frac{q(x_{t-1} \\mid x_t, x_0) \\, q(x_t \\mid x_0)}{q(x_{t-1} \\mid x_0)}$$
          <p>(The first equality is the Markov property — $x_t$ given $x_{t-1}$
          doesn't depend on $x_0$.) Substituting this into the sum:</p>
          $$\\sum_{t > 1} \\log \\frac{p_\\theta(x_{t-1} \\mid x_t)}{q(x_t \\mid x_{t-1})} \\;=\\; \\sum_{t > 1} \\log \\frac{p_\\theta(x_{t-1} \\mid x_t)}{q(x_{t-1} \\mid x_t, x_0)} \\;+\\; \\sum_{t > 1} \\log \\frac{q(x_{t-1} \\mid x_0)}{q(x_t \\mid x_0)}$$
          <p>The second sum <strong>telescopes</strong>:</p>
          $$\\sum_{t > 1} \\log \\frac{q(x_{t-1} \\mid x_0)}{q(x_t \\mid x_0)} \\;=\\; \\log \\frac{q(x_1 \\mid x_0)}{q(x_T \\mid x_0)}$$
          <p>Plug everything back and the $\\log q(x_1 \\mid x_0)$ cancels with the
          trailing term. After grouping:</p>
          $$L \\;=\\; \\mathbb{E}_q\\!\\left[\\log \\frac{q(x_T \\mid x_0)}{p(x_T)} \\;+\\; \\sum_{t > 1} \\log \\frac{q(x_{t-1} \\mid x_t, x_0)}{p_\\theta(x_{t-1} \\mid x_t)} \\;-\\; \\log p_\\theta(x_0 \\mid x_1)\\right]$$
          <p>The first two pieces are KL divergences (under expectation over $q$,
          $\\mathbb{E}_q[\\log q/p] = D_{\\mathrm{KL}}(q \\| p)$). The third is the
          reconstruction term. We arrive at the boxed decomposition.
          $\\blacksquare$</p>
          <p>(See Appendix A of the paper for the same derivation.)</p>
        `, true)}
    </div>

    ${crosslinkBack({
      toLesson: 'elbo-vi',
      toAnchor: 'elbo-two-forms',
      toAnchorLabel: '4 — Two forms of the ELBO',
      body: `The DDPM bound is Form 1 of the ELBO (the "evidence minus KL" view),
             specialized to a $T$-step Markov chain with frozen encoder.`,
    })}

    ${crosslinkBack({
      toLesson: 'kl-jensen',
      toAnchor: 'kl-gaussians',
      toAnchorLabel: '4 — KL of Gaussians',
      body: `Every $L_{t-1}$ term is a KL between two Gaussians. The closed-form
             formula for that KL is what makes the per-step loss tractable.`,
    })}

    <div id="viz-vlb-decomposition" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
