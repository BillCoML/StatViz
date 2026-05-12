import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-5';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§5</div>
    <h2>The Closed-Form KL Term</h2>
    <div class="prose">
      <p>The KL term has a closed form because both distributions are Gaussian.
      From <a href="/StatViz/lessons/gaussian-cookbook/#kl-mvn-diag">Gaussian Cookbook §3</a>:</p>

      $$D_{\\mathrm{KL}}\\!\\big(\\mathcal{N}(\\mu, \\mathrm{diag}(\\sigma^2)) \\,\\|\\, \\mathcal{N}(0, I)\\big)
        \\;=\\; \\tfrac{1}{2}\\sum_{i=1}^{d}\\!\\left[\\sigma_i^2 + \\mu_i^2 - 1 - \\log \\sigma_i^2\\right]$$

      <p>Substituting the encoder's outputs
      $\\mu = \\mu_\\phi(x)$, $\\log\\sigma = \\log\\sigma_\\phi(x)$:</p>

      $$\\boxed{\\;\\; D_{\\mathrm{KL}}\\!\\big(q_\\phi(z \\mid x) \\,\\|\\, p(z)\\big)
        \\;=\\; \\tfrac{1}{2}\\sum_{i=1}^{d}\\!\\left[
          e^{2\\log\\sigma_{\\phi,i}(x)} \\;+\\; \\mu_{\\phi,i}(x)^2 \\;-\\; 1 \\;-\\;
          2\\log\\sigma_{\\phi,i}(x)
        \\right] \\;\\;}$$

      <p>This is a <strong>closed-form expression</strong> in the encoder output.
      Differentiating it with respect to $\\phi$ via the chain rule through the encoder
      network is standard autograd — no Monte Carlo estimate needed.
      In contrast, the reconstruction term does need Monte Carlo because the expectation
      is over $z$. The reparameterization trick (§6) handles that.</p>

      <div class="worked-example">
        <div class="worked-example-title">Worked numerical example</div>
        <p>For a 2-dim latent with $\\mu = (0.5, -0.2)$ and $\\log\\sigma = (0.1, -0.3)$:</p>
        $$D_{\\mathrm{KL}} \\;=\\; \\tfrac{1}{2}\\big[
          (e^{0.2} + 0.25 - 1 - 0.2) + (e^{-0.6} + 0.04 - 1 + 0.6)
        \\big] \\;\\approx\\; 0.2301$$
        <p>Pre-computed; verified in <code>elbo.test.ts</code>.</p>
      </div>

      <p>The one line every VAE implementation has:</p>
      <pre class="pseudocode">kl_loss = 0.5 * sum(exp(2*log_sigma) + mu**2 - 1 - 2*log_sigma)</pre>
    </div>

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'kl-mvn-diag',
      toAnchorLabel: '3',
      body: `<p>This formula is the diagonal-Gaussian-against-standard-normal special case
        derived in Gaussian Cookbook §3. It is <em>the</em> identity behind the
        <code>kl_loss</code> line in every VAE implementation you will encounter.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
}
