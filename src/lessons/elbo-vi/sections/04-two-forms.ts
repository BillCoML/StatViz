import { renderMath, callout, crosslinkForward } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-4';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§4</div>
    <h2>Two Forms of the ELBO</h2>
    <div class="prose" id="elbo-two-forms">
      <p>The ELBO has two forms that look different but are equal. Knowing
      both is essential — different parts of the literature use different
      ones, and each makes a different intuition obvious.</p>

      <h3>Form 1 — joint form (what we derived)</h3>
      $$\\mathrm{ELBO}(q) \\;=\\; \\mathbb{E}_q[\\log p(x, Z)] \\;-\\; \\mathbb{E}_q[\\log q(Z)]$$
      <p>Define the entropy of $q$ as $H(q) := -\\mathbb{E}_q[\\log q(Z)]$:</p>
      $$\\mathrm{ELBO}(q) \\;=\\; \\mathbb{E}_q[\\log p(x, Z)] \\;+\\; H(q)$$
      <p><strong>Interpretation</strong>: pick $q$ to put mass on $z$ values where the
      joint $p(x, z)$ is large, but keep $q$'s entropy high (don't collapse to a delta
      function). Trade-off between fitting the joint and staying spread out.</p>

      <h3>Form 2 — reconstruction-KL form (the VAE form)</h3>
      <p>Use $\\log p(x, z) = \\log p(x \\mid z) + \\log p(z)$:</p>
      $$\\mathbb{E}_q[\\log p(x, Z)] - \\mathbb{E}_q[\\log q(Z)]
      \\;=\\; \\mathbb{E}_q[\\log p(x \\mid Z)] + \\mathbb{E}_q[\\log p(Z)] - \\mathbb{E}_q[\\log q(Z)]$$
      <p>Group the last two terms:
      $\\mathbb{E}_q[\\log p(Z)] - \\mathbb{E}_q[\\log q(Z)] = -\\mathbb{E}_q[\\log(q(Z)/p(Z))]
      = -D_{\\mathrm{KL}}(q(z) \\,\\|\\, p(z))$.</p>
      <p>Therefore:</p>
      $$\\boxed{\\;\\; \\mathrm{ELBO}(q) \\;=\\;
      \\underbrace{\\mathbb{E}_q[\\log p(x \\mid Z)]}_{\\text{reconstruction}}
      \\;-\\;
      \\underbrace{D_{\\mathrm{KL}}(q(z) \\,\\|\\, p(z))}_{\\text{regularizer}} \\;\\;}$$
      <p><strong>Interpretation</strong>: pick $q$ to make the observation $x$ likely
      when $z \\sim q$ (good <em>reconstruction</em>), but keep $q$ close to the
      <em>prior</em> $p(z)$ (don't drift away from what the model thought $z$
      should look like).</p>
      <p>This is <strong>exactly</strong> the VAE training objective. The encoder
      $q_\\phi(z \\mid x)$ produces a distribution over latents, the decoder
      $p_\\theta(x \\mid z)$ scores how well the latent reconstructs $x$,
      and the KL term keeps the latents structured.</p>
    </div>

    ${callout('tip', 'The two interpretations are both true',
      `<p>Form 1 says: balance fitting the joint against entropy of $q$.
      Form 2 says: balance reconstruction against staying near the prior.
      They're algebraically the same — different ways of grouping the same
      three terms. Form 2 dominates the deep-learning literature because
      the prior $p(z)$ is usually a fixed simple thing (a standard Gaussian),
      so the KL is closed form.</p>`
    )}

    ${crosslinkForward({
      toLesson: 'vae',
      body: `<p>The VAE training loss is, term-for-term, Form 2 of the ELBO with the
        sign flipped. The encoder amortizes the variational distribution over examples;
        the reparameterization trick (§5 below) makes the reconstruction term
        differentiable.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
}
