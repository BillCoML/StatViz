import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-2';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§2</div>
    <h2>The Setup</h2>
    <div class="prose">
      <h3>Latent-variable models</h3>
      <p>A <strong>latent-variable model</strong> has two kinds of random variables:</p>
      <ul>
        <li>$X$ — <strong>observed</strong>. We have data points $x_1, x_2, \\ldots, x_N$.</li>
        <li>$Z$ — <strong>latent</strong> (hidden). Never observed.</li>
      </ul>
      <p>The model specifies the <strong>joint distribution</strong> $p(x, z)$, typically
      factored as a prior on the latent and a conditional on the observation:</p>
      $$p(x, z) \\;=\\; p(z) \\, p(x \\mid z)$$
      <p>Two distributions derived from the joint matter for everything we'll do:</p>
      <ul>
        <li><strong>Marginal likelihood</strong> (a.k.a. <strong>evidence</strong>):
          $$p(x) \\;=\\; \\int p(x, z) \\, dz$$
          The probability the model assigns to the observation $x$, summing over all
          possible latents.</li>
        <li><strong>Posterior</strong> over the latent given the observation:
          $$p(z \\mid x) \\;=\\; \\frac{p(x, z)}{p(x)}$$
          What we believe about $Z$ after seeing $x$.</li>
      </ul>

      <h3>The intractability</h3>
      <p>For all but the simplest models, <strong>both</strong> $p(x)$ and $p(z \\mid x)$
      are intractable. They're the same computation: $p(x)$ is the integral;
      $p(z \\mid x)$ is the integrand divided by the integral. Either way, an
      unevaluated $\\int p(x, z) \\, dz$ is sitting in the denominator.</p>
      <p>The intractability is structural, not a matter of cleverness:</p>
      <ul>
        <li>For $z \\in \\mathbb{R}^d$ with $d$ large, the integral is high-dimensional
        and lacks closed form except in special conjugate cases.</li>
        <li>For $z$ discrete with $K$ states per dimension and $D$ dimensions,
        the sum has $K^D$ terms.</li>
      </ul>

      <h3>What variational inference does</h3>
      <p>VI replaces these intractable computations with optimization:</p>
      <ol>
        <li>Pick a <strong>variational family</strong> $\\mathcal{Q}$ of tractable
        distributions $q(z)$ — say, Gaussians.</li>
        <li>Find $q^* \\in \\mathcal{Q}$ that is "close" to $p(z \\mid x)$ in some sense.</li>
        <li>Use $q^*$ as a stand-in for the posterior wherever needed.</li>
      </ol>
      <p>The two design choices are: <strong>what family $\\mathcal{Q}$ to pick</strong>,
      and <strong>how to measure "close"</strong>. The next section answers the second
      question.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'kl-jensen',
      toAnchor: 'reverse-kl',
      toAnchorLabel: '7',
      body: `<p>The choice of "close" will turn out to be <em>reverse KL divergence</em>
        — $D_{\\mathrm{KL}}(q \\,\\|\\, p_{\\text{posterior}})$, not the forward direction.
        The forward-vs-reverse distinction and its mode-seeking consequence are covered
        in KL &amp; Jensen §7.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
}
