import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-7';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'sm-equivalence');
  sec.innerHTML = `
    <div class="section-label">§7</div>
    <h2 id="sm-equivalence">The Score Matching Connection</h2>
    <p class="lead"><em>The paper's central conceptual contribution.</em></p>
    <div class="prose">
      <p>We've derived $L_{\\text{simple}}$ from variational principles. Now we'll
      show — exactly as Ho et al. show in §3.2 of the paper — that the same loss
      arises from <strong>denoising score matching</strong> at $T$ different noise
      levels.</p>

      <p>Read the simplified loss again:</p>

      $$L_{\\text{simple}}(\\theta) \\;=\\; \\mathbb{E}_{t, x_0, \\epsilon}\\!\\left[\\big\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\big\\|^2\\right], \\quad x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1 - \\bar\\alpha_t} \\epsilon$$

      <p>Compare to denoising score matching from Score Matching §5:</p>

      $$L_{\\text{DSM}}(\\theta; \\sigma) \\;=\\; \\mathbb{E}_{x, \\varepsilon}\\!\\left[\\left\\|s_\\theta(\\tilde x, \\sigma) + \\frac{\\varepsilon}{\\sigma}\\right\\|^2\\right], \\quad \\tilde x = x + \\sigma \\varepsilon$$

      <p>These are the same equation if we make two substitutions.</p>

      <h3>The dictionary</h3>
      <table class="num-table">
        <thead><tr><th>DDPM</th><th>Score Matching</th></tr></thead>
        <tbody>
          <tr><td>$x_0$</td><td>$x$</td></tr>
          <tr><td>$x_t$</td><td>$\\tilde x$</td></tr>
          <tr><td>$\\sqrt{\\bar\\alpha_t} \\, x_0$</td><td>(clean signal — same)</td></tr>
          <tr><td>$\\sqrt{1 - \\bar\\alpha_t} \\, \\epsilon$</td><td>$\\sigma \\varepsilon$</td></tr>
          <tr><td>$\\epsilon_\\theta(x_t, t)$</td><td>$-\\sqrt{1 - \\bar\\alpha_t} \\, s_\\theta(\\tilde x, \\sigma)$</td></tr>
          <tr><td>$t$</td><td>$\\sigma_t = \\sqrt{(1-\\bar\\alpha_t)/\\bar\\alpha_t}$</td></tr>
        </tbody>
      </table>

      <p>The third row identifies the noise scales: in score matching, noise has
      scale $\\sigma$. In DDPM, noise has scale $\\sqrt{1 - \\bar\\alpha_t}$.
      Setting these equal: $\\sigma = \\sqrt{1 - \\bar\\alpha_t}$.</p>

      <p>The fifth row is the key relationship:</p>

      <div class="formula-box">$$s_\\theta(x_t, t) \\;=\\; -\\frac{\\epsilon_\\theta(x_t, t)}{\\sqrt{1 - \\bar\\alpha_t}}$$</div>

      <p>The DDPM "noise prediction" network is, up to a rescaling factor, the
      <strong>score</strong> of the noise-perturbed distribution
      $q_t = q(x_t \\mid x_0)$ marginalized over $x_0$.</p>

      <h3>Why the conversion factor</h3>
      <p>The score of
      $q(x_t \\mid x_0) = \\mathcal{N}(\\sqrt{\\bar\\alpha_t} x_0, (1 - \\bar\\alpha_t) I)$
      with respect to $x_t$ is:</p>

      $$\\nabla_{x_t} \\log q(x_t \\mid x_0) \\;=\\; -\\frac{x_t - \\sqrt{\\bar\\alpha_t} x_0}{1 - \\bar\\alpha_t} \\;=\\; -\\frac{\\sqrt{1 - \\bar\\alpha_t} \\, \\epsilon}{1 - \\bar\\alpha_t} \\;=\\; -\\frac{\\epsilon}{\\sqrt{1 - \\bar\\alpha_t}}$$

      <p>Predicting $\\epsilon$ is, up to a scaling factor of
      $-\\sqrt{1 - \\bar\\alpha_t}$, the same as predicting the score.</p>

      <h3>What's "annealed" in DDPM sampling</h3>
      <p>Score Matching used a noise schedule
      $\\sigma_1 > \\sigma_2 > \\cdots > \\sigma_L$ and ran annealed Langevin:
      $T$ inner steps at each $\\sigma_\\ell$.</p>

      <p>DDPM runs <strong>one step at each of $T = 1000$ noise levels</strong>,
      in sequence. The "annealing" is implicit in the noise schedule.</p>

      <h3>Why the variances $\\sigma_t$ in the sampler</h3>
      <p>DDPM Algorithm 2 adds noise at each step:
      $x_{t-1} = \\mu_\\theta + \\sigma_t z$. This noise injection is
      <strong>exactly</strong> the Langevin diffusion noise from Score Matching §6.
      The two framings produce identical update rules — the paper-derived
      $\\sigma_t$ (either $\\beta_t$ or $\\tilde\\beta_t$) plays the role of
      $\\sqrt{2 \\eta}$ in the Langevin update.</p>

      <h3>The equivalence in one sentence</h3>
      <blockquote class="callout-blockquote">
        <p><strong>DDPM training is denoising score matching at $T$ noise levels;
        DDPM sampling is annealed Langevin dynamics.</strong></p>
      </blockquote>

      <p>This is the punchline of the paper's §3.2. Everything from §2–§6 of the
      lesson — the variational derivation, the loss decomposition, the
      $\\epsilon$-parameterization — converges here.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'score-matching',
      toAnchor: 'dsm',
      toAnchorLabel: '5 (DSM) and §7 (annealed Langevin)',
      body: `The equivalence runs both ways: the score-matching lesson framed things in
             terms of $\\sigma$, the DDPM lesson in terms of $t$. The conversion is
             $\\sigma_t = \\sqrt{(1-\\bar\\alpha_t)/\\bar\\alpha_t}$ if you treat $\\sigma$
             as standard-deviation-of-noise-given-clean-data.`,
    })}

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'mvn-density',
      toAnchorLabel: '2 — MVN density',
      body: `The score of a Gaussian
             $\\mathcal{N}(\\mu, \\Sigma)$ is $-\\Sigma^{-1}(x - \\mu)$. For
             $q(x_t \\mid x_0)$ this gives $-\\epsilon / \\sqrt{1 - \\bar\\alpha_t}$ —
             the conversion factor.`,
    })}

    <div id="viz-score-equivalence" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
