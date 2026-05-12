import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-6';
  sec.className = 'section';
  sec.innerHTML = `
    <div id="reparam-in-vae"></div>
    <div class="section-label">§6</div>
    <h2>The Reparameterization Trick in the VAE Pipeline</h2>
    <div class="prose">
      <p>The reconstruction term
      $\\mathbb{E}_{q_\\phi(z \\mid x)}[\\log p_\\theta(x \\mid z)]$ involves an
      expectation over $z \\sim q_\\phi$. To train via gradient descent we need its
      gradient with respect to <em>both</em> $\\theta$ (straightforward — $\\theta$
      only appears inside $\\log p_\\theta$) and $\\phi$ (which appears in the
      <em>distribution</em> we are sampling from).</p>

      <p>Sampling $z \\sim q_\\phi$ naively is not differentiable in $\\phi$:
      a different $\\phi$ leads to a different $z$, but the sample is a random number
      with no derivative.</p>

      <h3>The trick (from Gaussian Cookbook §4)</h3>
      <p>Rewrite the sample as a deterministic transformation of fixed noise:</p>
      $$\\boxed{\\;\\; z \\;=\\; \\mu_\\phi(x) \\;+\\; \\sigma_\\phi(x) \\odot \\varepsilon,
        \\qquad \\varepsilon \\sim \\mathcal{N}(0, I) \\;\\;}$$
      <p>Now $z$ is a deterministic function of $\\phi$ — and $\\varepsilon$ is just
      noise with no parameters. The Monte Carlo gradient estimate becomes:</p>
      $$\\nabla_\\phi \\, \\mathbb{E}_{q_\\phi}[\\log p_\\theta(x \\mid z)]
        \\;\\approx\\; \\nabla_\\phi \\log p_\\theta\\!\\big(x \\mid \\mu_\\phi(x)
        + \\sigma_\\phi(x) \\odot \\varepsilon\\big)$$
      <p>with $\\varepsilon$ sampled once per gradient step. The gradient flows through
      $\\mu_\\phi, \\sigma_\\phi$ and into the encoder weights $\\phi$ via standard backprop.
      <strong>This is what makes end-to-end gradient training of VAEs possible.</strong></p>

      <h3>One-step VAE training (pseudo-code)</h3>
      <pre class="pseudocode">for each minibatch x:
  mu, log_sigma  = encoder(x; φ)
  eps            = randn(latent_dim)               # noise
  z              = mu + exp(log_sigma) * eps        # reparameterization
  x_hat          = decoder(z; θ)
  recon_loss     = 0.5 * ||x - x_hat||² / σ_x²
  kl_loss        = 0.5 * sum(exp(2*log_sigma) + mu² - 1 - 2*log_sigma)
  loss           = recon_loss + kl_loss
  loss.backward()
  optimizer.step()</pre>
      <p>That is all. Eight lines, end-to-end differentiable.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'reparam-matrix',
      toAnchorLabel: '4',
      body: `<p>The matrix-form reparameterization trick from Gaussian Cookbook §4,
        specialized to diagonal covariance, is what makes
        <code>z = mu + exp(log_sigma) * eps</code> differentiable with respect to the
        encoder parameters.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
}
